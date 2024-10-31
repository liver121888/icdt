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
from . import owlv2


def map_bbox_to_original(bbox_resized, scale):
    xmin_resized, ymin_resized, xmax_resized, ymax_resized = bbox_resized

    # Map coordinates back to original image
    xmin_original = xmin_resized / scale
    xmax_original = xmax_resized / scale
    ymin_original = ymin_resized / scale
    ymax_original = ymax_resized / scale

    return [xmin_original, ymin_original, xmax_original, ymax_original]


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
        self.detection_topic = "/camera/color/detections"
        self.marker_array_topic = "/detections_marker_array"
        self.rgb_sub = self.create_subscription(
            ImageMsg, self.rgb_topic, self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            ImageMsg, self.depth_topic, self.depth_callback, 10
        )
        self.detection_pub = self.create_publisher(ImageMsg, self.detection_topic, 10)
        self.marker_array_pub = self.create_publisher(
            MarkerArray, self.marker_array_topic, 10
        )
        self.bridge = CvBridge()
        self.last_depth_image = None

        # Debug Flags
        self.published_detections = False

        # OwlV2 Setup
        self.detection_model = owlv2.ObjectDetectionModel()

        self.logger = self.get_logger()
        self.logger.info("Detection Service initialized")

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

    def rgb_callback(self, msg):
        if self.published_detections:
            return
        elif self.last_depth_image is None:
            return

        self.published_detections = True

        self.logger.info(f"Received image with shape: {msg.height}x{msg.width}")
        # Convert to OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(cv_image)

        # Preprocess image
        classes = ["can", "plate", "ball", "block", "box"]
        inputs = self.detection_model.preprocess_image(pil_image, classes)

        # Perform detection
        self.logger.info("Performing detection")
        t0 = time.time()
        outputs = self.detection_model.detect_objects(inputs)
        t1 = time.time()
        self.logger.info(f"Detection time: {t1 - t0} seconds")

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

        annotated_image = self.annotate_image(cv_image, objects)

        cv_msg = self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        self.detection_pub.publish(cv_msg)
        self.publish_marker_array(objects)

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1") / 1000.0
        self.last_depth_image = depth_image

    def publish_marker_array(self, objects):
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()

        for idx, obj in enumerate(objects):
            marker = Marker()
            marker.header.stamp = current_time
            marker.header.frame_id = "camera_color_optical_frame"
            marker.ns = "object_markers"
            marker.id = idx  # Unique ID for each marker
            marker.type = Marker.SPHERE  # You can use other types, e.g., CUBE, ARROW
            marker.action = Marker.ADD

            # Set the pose of the marker
            marker.pose.position.x = obj.center_3d[0]
            marker.pose.position.y = obj.center_3d[1]
            marker.pose.position.z = obj.center_3d[2]

            # Optional: Orientation (if needed, here it’s identity/no rotation)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Define scale (size of the marker in each dimension)
            marker.scale.x = 0.1  # Adjust for appropriate size
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Define color (RGBA)
            marker.color.r = 1.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Fully opaque

            print(f"{obj.label} Marker: {obj.center_3d}")
            marker_array.markers.append(marker)

        # Publish the marker array
        self.marker_array_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    detection_service = DetectionService()
    rclpy.spin(detection_service)
    detection_service.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
