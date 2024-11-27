#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import numpy as np
import time
from icdt_interfaces.srv import ObjectDetection, MotionPlanning

from icdt_wrapper.perception_utils import *
from icdt_wrapper.llm import LLMRouter, SYSTEM_PROMPT


# RobotInterfaceNode as Central Interface for Robotics Functions
class RobotInterfaceNode(Node):
    def __init__(self):
        super().__init__('robot_interface')
        self.detection_client = DetectionClient(self)  # Initialize the detection client
        self.motion_planning_cli = self.create_client(MotionPlanning, '/lbr/motion_planning')
        while not self.motion_planning_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')


        # LLM Stuff
        self.llm_router = LLMRouter("gpt-4o", SYSTEM_PROMPT)

    def detect_objects(self):
        """Calls the DetectionClient to detect objects and returns a collection of detected objects."""
        if self.detection_client:
            return self.detection_client.send_detection_request()
        else:
            self.get_logger().error("Detection client is not initialized.")
            return DetectedObjectsCollection([])
        
    def move_robot(self, target_pose, dummy=False):
        """Move robot from current pose to target pose."""
        if dummy:
            self.get_logger().info("[Dummy] moving robot to target pose")
            return
        
        motion_planning_req = MotionPlanning.Request()
        motion_planning_req.goal = target_pose
        self.get_logger().info("Sending goal to motion planning service")

        # Call the service and wait for the result
        future = self.motion_planning_cli.call_async(motion_planning_req)
        rclpy.spin_until_future_complete(self, future)

        # Check if we received a response
        if future.done():
            response = future.result()
            self.get_logger().info(f'Result of motion_planning service: {response.success}')
        else:
            self.get_logger().error("Failed to receive a response from motion planning service")

    def move_home(self, dummy=False):
        """Move Robot to Home Position."""
        if dummy:
            self.get_logger().info("[Dummy] moving robot to home position")
            return

        home_pose = PoseStamped()
        home_pose.header.frame_id = "link_0"
        home_pose.pose.position.x = -0.4
        home_pose.pose.position.y = 0.0
        home_pose.pose.position.z = 0.36
        home_pose.pose.orientation.x = 0.0
        home_pose.pose.orientation.y = 1.0
        home_pose.pose.orientation.z = 0.0
        home_pose.pose.orientation.w = 0.0

        self.move_robot(home_pose)


# Main Interactive Loop
def main(args=None):
    rclpy.init(args=args)

    # Initialize the RobotInterfaceNode instance
    robot_interface = RobotInterfaceNode()

    try:
        while rclpy.ok():
            # Get user input for target object
            user_prompt ="""
            [Scene Description]
            """
            user_task = input("Prompt: ").strip()
            detections = robot_interface.detect_objects()
            scene_description = detections.get_scene_description()
            user_prompt += scene_description
            user_prompt += "\n[Instruction]\n"
            user_prompt += user_task

            llm_output = robot_interface.llm_router(user_prompt)
            reasoning, valid_code = llm_output
            print(f"Reasoning: {reasoning}")
            if valid_code:
                print(f"Valid Code: {valid_code}")
                if_execute = input("Execute Code? (Y/N): ").strip().lower()
                if if_execute == "y":
                    exec(valid_code)
                time.sleep(2)
                robot_interface.move_home()



            
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        robot_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()