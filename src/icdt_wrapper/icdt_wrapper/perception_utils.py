import rclpy
from rclpy.node import Node
import json
import numpy as np
from icdt_interfaces.srv import ObjectDetection
from geometry_msgs.msg import PoseStamped


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
                bbox_3d=np.array(obj['bbox_3d']) if obj['bbox_3d'] else None, # [x1, y1, z1, x2, y2, z2]
                center_3d=np.array(obj['center_3d']) if obj['center_3d'] else None, # [x, y, z]
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
        self.x_width = bbox[1] - bbox[0]
        self.y_width = bbox[3] - bbox[2]
        self.z_width = bbox_3d[5] - bbox_3d[2]

    def __eq__(self, other):
        """Override equality to allow detection search by label in DetectedObjectsCollection."""
        return isinstance(other, DetectedObject) and self.label == other.label


# Collection of Detection Objects with Lookup Functionality
class DetectedObjectsCollection:
    """Collection class to manage and search detected objects."""
    def __init__(self, objects):
        self.objects = objects
        self._index = 0

    def __iter__(self):
        """Make the collection iterable."""
        self._index = 0
        return self

    def __next__(self):
        """Enable iteration with next()."""
        if self._index < len(self.objects):
            result = self.objects[self._index]
            self._index += 1
            return result
        raise StopIteration

    def __contains__(self, label):
        """Enable 'if object in detections' syntax."""
        return any(obj.label == label for obj in self.objects)

    def find(self, label):
        """Find a specific detected object by label."""
        for obj in self.objects:
            if obj.label == label:
                return obj
        return None
    
    def get_scene_description(self):
        """Generate a description of the scene by iterating over each object."""
        descriptions = []
        for obj in self.objects:
            if obj.center_3d is not None:
                description = f"There is a [{obj.label}] at [{obj.center_3d[0]:.2f}, {obj.center_3d[1]:.2f}, {obj.center_3d[2]:.2f}]"
            else:
                description = f"There is a [{obj.label}] without 3D center information"
            descriptions.append(description)
        return "\n".join(descriptions)


# DetectionClient Class for Handling Detection Requests
class DetectionClient:
    def __init__(self, node: Node, detection_service: str):
        self.client = node.create_client(ObjectDetection, detection_service)
        while not self.client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('Waiting for service detect_objects to be available...')
        self.request = ObjectDetection.Request()
        self.node = node
        self.classes = ["can", "plate", "ball", "block", "box", "wallet", "napkin"]  # List of classes to detect

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
