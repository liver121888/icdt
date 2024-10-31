import rclpy
from rclpy.node import Node
import json
import numpy as np
from icdt_interfaces.srv import ObjectDetection


# Custom JSON Decoder to Parse JSON into DetectedObject Instances
class NumpyDecoder(json.JSONDecoder):
    """Custom JSON Decoder to convert lists back to numpy arrays where appropriate."""
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if 'bbox' in obj and 'center' in obj:  # Likely an Object structure
            return DetectedObject(
                label=obj['label'],
                score=obj['score'],
                bbox=np.array(obj['bbox']),
                center=np.array(obj['center']),
                bbox_3d=np.array(obj['bbox_3d']) if obj['bbox_3d'] else None,
                center_3d=np.array(obj['center_3d']) if obj['center_3d'] else None,
            )
        return obj


# Individual Detection Object
class DetectedObject:
    """Class to hold individual detection details."""
    def __init__(self, label, score, bbox, center, bbox_3d=None, center_3d=None):
        self.label = label
        self.score = score
        self.bbox = bbox
        self.center = center
        self.bbox_3d = bbox_3d
        self.center_3d = center_3d

    def get_pose(self):
        """Returns the 3D center pose of the detected object."""
        return self.center_3d

    def __eq__(self, other):
        """Override equality to allow detection search by label in DetectedObjectsCollection."""
        return isinstance(other, DetectedObject) and self.label == other.label


# Collection of Detection Objects with Lookup Functionality
class DetectedObjectsCollection:
    """Collection class to manage and search detected objects."""
    def __init__(self, objects):
        self.objects = objects

    def __contains__(self, label):
        """Enable 'if object in detections' syntax."""
        return any(obj.label == label for obj in self.objects)

    def find(self, label):
        """Find a specific detected object by label."""
        for obj in self.objects:
            if obj.label == label:
                return obj
        return None


# DetectionClient Class for Handling Detection Requests
class DetectionClient:
    def __init__(self, node: Node):
        self.client = node.create_client(ObjectDetection, 'detect_objects')
        while not self.client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('Waiting for service detect_objects to be available...')
        self.request = ObjectDetection.Request()
        self.node = node
        self.classes = ["can", "plate", "ball", "block", "box"]  # List of classes to detect

    def send_detection_request(self):
        """Sends a detection request and returns a collection of detected objects."""
        self.request.class_names = " ".join(self.classes)
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            detections = json.loads(future.result().detections, cls=NumpyDecoder)
            return DetectedObjectsCollection(detections)
        else:
            self.node.get_logger().error("Service call failed")
            return DetectedObjectsCollection([])


# RobotInterfaceNode as Central Interface for Robotics Functions
class RobotInterfaceNode(Node):
    def __init__(self):
        super().__init__('robot_interface')
        self.detection_client = DetectionClient(self)  # Initialize the detection client
        # Future clients can be added here as members

    def detect_objects(self):
        """Calls the DetectionClient to detect objects and returns a collection of detected objects."""
        if self.detection_client:
            return self.detection_client.send_detection_request()
        else:
            self.get_logger().error("Detection client is not initialized.")
            return DetectedObjectsCollection([])


# Main Interactive Loop
def main(args=None):
    rclpy.init(args=args)

    # Initialize the RobotInterfaceNode instance
    robot_interface = RobotInterfaceNode()

    try:
        while rclpy.ok():
            # Get user input for target object
            target_label = input("Object: ").strip().lower()
            
            # Call the detect_objects method from RobotInterfaceNode
            detections = robot_interface.detect_objects()

            # Check if the target label is in the detections and output location
            if target_label in detections:
                detected_object = detections.find(target_label)
                if detected_object:
                    print(f"{detected_object.label} found at {detected_object.get_pose()}")
            else:
                print(f"{target_label} not found.")
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        robot_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
