# Create a node to advertise detection services

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ImageMsg
from PIL import Image
import numpy as np
from cv_bridge import CvBridge
import cv2
import time
from . import owlv2

class DetectionService(Node):
    def __init__(self):
        super().__init__('detection_service')

        # ROS Setup
        self.rgb_topic = '/camera/color/image_rect_raw'
        self.detection_topic = '/camera/color/detections'
        self.rgb_sub = self.create_subscription(ImageMsg, self.rgb_topic, self.rgb_callback, 10)
        self.detection_pub = self.create_publisher(ImageMsg, self.detection_topic, 10)
        self.bridge = CvBridge()

        # Debug Flags
        self.published_detections = False

        # OwlV2 Setup
        self.detection_model = owlv2.ObjectDetectionModel()

        self.logger = self.get_logger()
        self.logger.info("Detection Service initialized")


    def annotate_image(self, image, boxes, labels):
        n = len(labels)
        for i in range(n):
            cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 0, 255), 2)
            center = (int((boxes[i][0] + boxes[i][2]) // 2), int((boxes[i][1] + boxes[i][3]) // 2))
            cv2.putText(image, labels[i], center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image


    def rgb_callback(self, msg):
        if self.published_detections:
            return
        self.published_detections = True

        self.logger.info(f"Received image with shape: {msg.height}x{msg.width}")
        # Convert to OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(cv_image)

        # Preprocess image
        classes = ['can', 'plate', 'ball', 'block', 'box']
        inputs = self.detection_model.preprocess_image(pil_image, classes)

        # Perform detection
        self.logger.info("Performing detection")
        t0 = time.time()
        outputs = self.detection_model.detect_objects(inputs)
        t1 = time.time()
        self.logger.info(f"Detection time: {t1 - t0} seconds")

        # Postprocess results
        unnormalized_image = self.detection_model.get_preprocessed_image(inputs.pixel_values)
        boxes, scores, labels = self.detection_model.post_process_results(outputs, unnormalized_image, classes)
        labels = [classes[label] for label in labels]
        cv_image = np.array(unnormalized_image)

        annotated_image = self.annotate_image(cv_image, boxes, labels)

        cv_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'rgb8')
        self.detection_pub.publish(cv_msg)


def main(args=None):
    rclpy.init(args=args)
    detection_service = DetectionService()
    rclpy.spin(detection_service)
    detection_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()