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


# RobotInterfaceNode as Central Interface for Robotics Functions
class RobotInterfaceNode(Node):
    def __init__(self):
        super().__init__('robot_interface')
        self.detection_client = DetectionClient(self)  # Initialize the detection client
        # lbr
        # self.motion_planning_cli = self.create_client(MotionPlanning, '/lbr_motion_planning')
        # franka
        self.motion_planning_cli = self.create_client(MotionPlanning, '/franka_motion_planning')
        while not self.motion_planning_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        self.gripper_action_client = ActionClient(self, Move, '/fr3_gripper/move')
        while not self.gripper_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('action server not available, waiting again...')

        # LLM Stuff
        self.llm_router = LLMRouter("gpt-4o", SYSTEM_PROMPT)

    def detect_objects(self):
        """Calls the DetectionClient to detect objects and returns a collection of detected objects."""
        if self.detection_client:
            return self.detection_client.send_detection_request()
        else:
            self.get_logger().error("Detection client is not initialized.")
            return DetectedObjectsCollection([])
        
    def move_robot(self, goal_pose, dummy=False):
        """Move robot from current pose to target pose."""
        if dummy:
            self.get_logger().info("[Dummy] moving robot to target pose")
            return
        
        # change frame to fix pilz error
        target_pose = PoseStamped()
        target_pose.header.frame_id = "world"
        target_pose.pose.position.x = goal_pose.pose.position.x
        target_pose.pose.position.y = goal_pose.pose.position.y
        target_pose.pose.position.z = goal_pose.pose.position.z
        target_pose.pose.orientation.x = goal_pose.pose.orientation.x
        target_pose.pose.orientation.y = goal_pose.pose.orientation.y
        target_pose.pose.orientation.z = goal_pose.pose.orientation.z
        target_pose.pose.orientation.w = goal_pose.pose.orientation.w

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

        # lbr
        # home_pose = PoseStamped()
        # home_pose.header.frame_id = "world"
        # home_pose.pose.position.x = -0.4
        # home_pose.pose.position.y = 0.0
        # home_pose.pose.position.z = 0.36
        # home_pose.pose.orientation.x = 0.0
        # home_pose.pose.orientation.y = 1.0
        # home_pose.pose.orientation.z = 0.0
        # home_pose.pose.orientation.w = 0.0

        # franka
        home_pose = PoseStamped()
        home_pose.header.frame_id = "world"
        # home_pose.pose.position.x = 0.31
        # offset -1.3 meter, 0.31 - 1.3 ~= -1.0
        home_pose.pose.position.x = -1.0
        home_pose.pose.position.y = 0.0
        home_pose.pose.position.z = 0.485
        home_pose.pose.orientation.x = -1.0
        home_pose.pose.orientation.y = 0.0
        home_pose.pose.orientation.z = 0.0
        home_pose.pose.orientation.w = 0.0

        self.move_robot(home_pose)

    def open_gripper(self):
        """Open the Robot Gripper."""
        # 0.035 on each side is open, target width is 0.07
        self.move_gripper(target_width=0.07, speed=0.1)

    def close_gripper(self):
        """Close the Robot Gripper."""
        self.move_gripper(target_width=0.01, speed=0.1)

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
    
            # llm_output = robot_interface.llm_router(user_prompt)

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
            dummy_code = """robot_interface.move_home()"""
            llm_output = ["Dummy Reasoning", dummy_code]

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
