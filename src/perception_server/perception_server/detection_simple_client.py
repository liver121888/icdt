import rclpy
from rclpy.node import Node
from icdt_interfaces.srv import ObjectDetection

class ObjectDetectionClient(Node):
    def __init__(self):
        super().__init__('object_detection_client')
        self.client = self.create_client(ObjectDetection, 'detect_objects')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service detect_objects to be available...')
        self.request = ObjectDetection.Request()

    def send_request(self, classes):
        # Convert list of classes to a space-separated string
        self.request.class_names = " ".join(classes)
        
        # Call the service and wait for the result
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            # Print the JSON-formatted detection result
            self.get_logger().info(f"Detection result: {future.result().detections}")
        else:
            self.get_logger().error("Service call failed")


def main(args=None):
    rclpy.init(args=args)
    client = ObjectDetectionClient()
    
    # Define the classes to send
    classes = ["can", "plate", "ball", "block", "box"]
    
    # Send the request
    client.send_request(classes)
    
    # Shutdown the node
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
