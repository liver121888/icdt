# Create a node to advertise detection services

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ImageMsg
from PIL import Image
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import time
import json
from std_msgs.msg import String
from . import owlv2
from icdt_interfaces.srv import ObjectDetection


def map_bbox_to_original(bbox_resized, scale):
    xmin_resized, ymin_resized, xmax_resized, ymax_resized = bbox_resized

    # Map coordinates back to original image
    xmin_original = xmin_resized / scale
    xmax_original = xmax_resized / scale
    ymin_original = ymin_resized / scale
    ymax_original = ymax_resized / scale

    return [xmin_original, ymin_original, xmax_original, ymax_original]

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)  # Convert numpy float to Python float
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)  # Convert numpy int to Python int
        return super().default(obj)

def objects_to_json(objects):
    objects_data = []
    for obj in objects:
        obj_dict = {
            "label": obj.label,
            "score": obj.score,
            "bbox": obj.bbox,
            "center": obj.center,
            "bbox_3d": obj.bbox_3d if obj.bbox_3d is not None else None,
            "center_3d": obj.center_3d if obj.center_3d is not None else None
        }
        objects_data.append(obj_dict)
    
    # Return JSON as a string, using NumpyEncoder to handle numpy types
    return json.dumps(objects_data, cls=NumpyEncoder)


class Object:
    def __init__(self, label, score, box):
        self.label = label
        self.score = score

        # 2D Coordinates
        scale = 960 / 848
        scaled_box = map_bbox_to_original(box, scale)
        self.bbox = list(map(int, scaled_box))  # [x1, y1, x2, y2]
        self.center = (
            int((scaled_box[0] + scaled_box[2]) / 2),
            int((scaled_box[1] + scaled_box[3]) / 2),
        )

        # D405 Intrinsics for 3D Projection
        self.fx = 425.19189453125
        self.fy = 424.6562805175781
        self.cx = 422.978515625
        self.cy = 242.1155242919922

        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        )

        # 3D Coordinates
        self.bbox_3d = None  # [x1, y1, z1, x2, y2, z2]
        self.center_3d = None  # [x, y, z]

    def project_to_3d(self, depth_img):
        # Get depth at the center of the bounding box
        z_center = depth_img[self.center[1], self.center[0]]

        # Calculate 3D coordinates of the center
        x_center_3d = (self.center[0] - self.cx) * z_center / self.fx
        y_center_3d = (self.center[1] - self.cy) * z_center / self.fy
        self.center_3d = [x_center_3d, y_center_3d, z_center]

        # Calculate 3D coordinates for the bounding box corners
        z1 = depth_img[self.bbox[1], self.bbox[0]]  # Top-left corner
        z2 = depth_img[self.bbox[3], self.bbox[2]]  # Bottom-right corner

        # Project top-left corner
        x1_3d = (self.bbox[0] - self.cx) * z1 / self.fx
        y1_3d = (self.bbox[1] - self.cy) * z1 / self.fy

        # Project bottom-right corner
        x2_3d = (self.bbox[2] - self.cx) * z2 / self.fx
        y2_3d = (self.bbox[3] - self.cy) * z2 / self.fy

        # Set the 3D bounding box
        self.bbox_3d = [x1_3d, y1_3d, z1, x2_3d, y2_3d, z2]

    def __str__(self):
        return f"Object(label={self.label}, score={self.score}, bbox={self.bbox}, center={self.center})"


class DetectionService(Node):
    def __init__(self):
        super().__init__("detection_service")

        # ROS Setup
        self.rgb_topic = "/camera/color/image_rect_raw"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"
        self.detection_viz_topic = "/camera/color/detections"
        self.pose_array_topic = "/camera/color/pose_array"
        self.rgb_sub = self.create_subscription(
            ImageMsg, self.rgb_topic, self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            ImageMsg, self.depth_topic, self.depth_callback, 10
        )
        self.detection_pub = self.create_publisher(ImageMsg, self.detection_viz_topic, 10)
        self.pose_array_pub = self.create_publisher(PoseArray, self.pose_array_topic, 10)
        self.bridge = CvBridge()
        self.last_depth_image = None
        self.last_rgb_image = None

        # Service Setup
        self.srv = self.create_service(ObjectDetection, 'detect_objects', self.detect_objects_callback)

        # OwlV2 Setup
        self.detection_model = owlv2.ObjectDetectionModel()

        self.logger = self.get_logger()
        self.logger.info("Detection Service initialized")

    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.last_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1") / 1000.0
        self.last_depth_image = depth_image

    def detect_objects_callback(self, request, response):
        t0 = time.time()
        self.logger.info(f"Received request")
        if self.last_rgb_image is None or self.last_depth_image is None:
            response.detections = "[]"
            return response

        # Convert to PIL Image
        pil_image = Image.fromarray(self.last_rgb_image)

        # Parse requested classes
        classes = request.class_names.split()

        # Preprocess image
        inputs = self.detection_model.preprocess_image(pil_image, classes)

        # Perform detection
        outputs = self.detection_model.detect_objects(inputs)

        # Postprocess results
        unnormalized_image = self.detection_model.get_preprocessed_image(
            inputs.pixel_values
        )
        boxes, scores, labels = self.detection_model.post_process_results(
            outputs, unnormalized_image, classes
        )
        labels = [classes[label] for label in labels]

        objects = [
            Object(label, score, box)
            for label, score, box in zip(labels, scores, boxes)
        ]
        for obj in objects:
            obj.project_to_3d(self.last_depth_image)

        # Annotate Image for Visualization
        annotated_image = self.annotate_image(self.last_rgb_image, objects)
        cv_msg = self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        self.detection_pub.publish(cv_msg)

        response.detections = objects_to_json(objects)
        self.logger.info(f"Detection Service Completed in {time.time() - t0:.2f} seconds")
        return response
    
    def annotate_image(self, image, objects):
        n = len(objects)
        for i in range(n):
            cv2.rectangle(
                image,
                (int(objects[i].bbox[0]), int(objects[i].bbox[1])),
                (int(objects[i].bbox[2]), int(objects[i].bbox[3])),
                (0, 0, 255),
                2,
            )
            center = (
                int((objects[i].bbox[0] + objects[i].bbox[2]) // 2),
                int((objects[i].bbox[1] + objects[i].bbox[3]) // 2),
            )
            cv2.putText(
                image,
                objects[i].label,
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return image

    def publish_pose_array(self, objects):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "camera_color_optical_frame"

        for obj in objects:
            pose = Pose()
            
            # Set the position of the pose
            pose.position.x = obj.center_3d[0]
            pose.position.y = obj.center_3d[1]
            pose.position.z = obj.center_3d[2]

            # Optional: Set orientation (if needed, here it’s identity/no rotation)
            pose.orientation.x = 0.0
            pose.orientation.y = 1.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0

            print(f"{obj.label} Pose: {obj.center_3d}")
            pose_array.poses.append(pose)

        # Publish the pose array
        self.pose_array_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    detection_service = DetectionService()
    rclpy.spin(detection_service)
    detection_service.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
