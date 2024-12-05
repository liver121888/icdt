#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import json
import numpy as np
import time
from icdt_interfaces.srv import ObjectDetection, MotionPlanning
from franka_msgs.action import Move
from icdt_wrapper.perception_utils import *
from icdt_wrapper.llm import LLMRouter, SYSTEM_PROMPT
from geometry_msgs.msg import PoseStamped

# Base RobotInterface Class
class RobotInterface(Node):
    def __init__(self, node_name, motion_planning_service, detection_service):
        super().__init__(node_name)
        self.motion_planning_cli = self.create_client(MotionPlanning, motion_planning_service)
        while not self.motion_planning_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Motion planning service not available, waiting again...')

        self.detection_client = DetectionClient(self, detection_service)

        self.llm_router = LLMRouter("gpt-4o", SYSTEM_PROMPT)

    def move_robot(self, goal_pose, dummy=False):
        """Move robot from current pose to target pose."""
        if dummy:
            self.get_logger().info("[Dummy] moving robot to target pose")
            return

        target_pose = PoseStamped()
        target_pose.header.frame_id = "world"
        target_pose.pose = goal_pose.pose

        motion_planning_req = MotionPlanning.Request()
        motion_planning_req.goal = target_pose
        self.get_logger().info("Sending goal to motion planning service")

        future = self.motion_planning_cli.call_async(motion_planning_req)
        rclpy.spin_until_future_complete(self, future)

        if future.done():
            response = future.result()
            self.get_logger().info(f'Result of motion_planning service: {response.success}')
        else:
            self.get_logger().error("Failed to receive a response from motion planning service")

    def detect_objects(self):
        """Calls the Object Detection service to detect objects and returns the results."""
        if self.detection_client:
            return self.detection_client.send_detection_request()
        else:
            self.get_logger().error("Detection client is not initialized.")
            return DetectedObjectsCollection([])

# Derived Class for LBR Robot
class LBRInterface(RobotInterface):
    def __init__(self):
        super().__init__('lbr_interface', '/lbr/lbr_motion_planning', '/detect_objects_d405')

    def move_home(self, dummy=False):
        """Move LBR Robot to Home Position."""
        if dummy:
            self.get_logger().info("[Dummy] moving LBR robot to home position")
            return

        home_pose = PoseStamped()
        home_pose.header.frame_id = "world"
        home_pose.pose.position.x = -0.4
        home_pose.pose.position.y = 0.0
        home_pose.pose.position.z = 0.36
        home_pose.pose.orientation.x = 0.0
        home_pose.pose.orientation.y = 1.0
        home_pose.pose.orientation.z = 0.0
        home_pose.pose.orientation.w = 0.0

        self.move_robot(home_pose)

# Derived Class for Franka Robot
class FrankaInterface(RobotInterface):
    def __init__(self):
        super().__init__('franka_interface', '/franka_motion_planning', '/franka_object_detection')
        self.gripper_action_client = ActionClient(self, Move, '/fr3_gripper/move')
        while not self.gripper_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Gripper action server not available, waiting again...')

    def move_home(self, dummy=False):
        """Move Franka Robot to Home Position."""
        if dummy:
            self.get_logger().info("[Dummy] moving Franka robot to home position")
            return

        home_pose = PoseStamped()
        home_pose.header.frame_id = "world"
        home_pose.pose.position.x = -1.0
        home_pose.pose.position.y = 0.0
        home_pose.pose.position.z = 0.485
        home_pose.pose.orientation.x = -1.0
        home_pose.pose.orientation.y = 0.0
        home_pose.pose.orientation.z = 0.0
        home_pose.pose.orientation.w = 0.0

        self.move_robot(home_pose)

    def move_gripper(self, target_width, speed):
        """Move the Robot Gripper to the target width."""
        goal_msg = Move.Goal()
        goal_msg.width = target_width
        goal_msg.speed = speed

        self.get_logger().info(f'Sending gripper goal: width={target_width} m, speed={speed} m/s')
        self.gripper_action_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda feedback_msg: self.get_logger().info(
                f'Feedback: Current width={feedback_msg.current_width:.3f} m'
            )
        ).add_done_callback(
            lambda future: self.get_logger().info(
                'Goal accepted' if future.result().accepted else 'Goal rejected'
            )
        )

    def open_gripper(self):
        """Open the Robot Gripper."""
        self.move_gripper(target_width=0.07, speed=0.1)

    def close_gripper(self):
        """Close the Robot Gripper."""
        self.move_gripper(target_width=0.01, speed=0.1)

# Main Interactive Loop
def main(args=None):
    rclpy.init(args=args)

    # Initialize the RobotInterfaceNode instance
    robot_interface = LBRInterface()

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

            # dummy
            # dummy_code = """
            # target_label = 'can'    
            # if target_label in detections:
            #     detected_object = detections.find(target_label)
            #     if detected_object:
            #         print(f"{detected_object.label} found at {detected_object.center_3d}")
            #         print(f"Moving to {detected_object.center_3d}")
            #         robot_interface.move_robot(detected_object.get_pose())
            # else:
            #     print(f"{target_label} not found.")
            # """
            # dummy_code = """robot_interface.open_gripper()"""
            # llm_output = ["Dummy Reasoning", dummy_code]

            reasoning, valid_code = llm_output
            print(f"Reasoning: {reasoning}")
            if valid_code:
                print(f"Valid Code: {valid_code}")
                if_execute = input("Execute Code? (Y/N): ").strip().lower()
                if if_execute == "y":
                    exec(valid_code)
                time.sleep(2)
                # robot_interface.move_home()

    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        robot_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
