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
# from icdt_wrapper.llm import LLMRouter, SYSTEM_PROMPT
from icdt_wrapper.llm import LLMRouter, LBR_SYSTEM_PROMPT as SYSTEM_PROMPT
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
        home_pose.pose.position.x = -0.22
        home_pose.pose.position.y = 0.0
        home_pose.pose.position.z = 0.19
        home_pose.pose.orientation.x = 0.0
        home_pose.pose.orientation.y = 1.0
        home_pose.pose.orientation.z = 0.0
        home_pose.pose.orientation.w = 0.0

        self.move_robot(home_pose)

    def move_to_scan_pose(self, dummy=False):
        """Move LBR Robot to Scan Pose."""
        if dummy:
            self.get_logger().info("[Dummy] moving LBR robot to scan pose")
            return

        scan_pose = PoseStamped()
        scan_pose.header.frame_id = "world"
        scan_pose.pose.position.x = -0.4
        scan_pose.pose.position.y = 0.0
        scan_pose.pose.position.z = 0.19
        scan_pose.pose.orientation.x = 0.0
        scan_pose.pose.orientation.y = 1.0
        scan_pose.pose.orientation.z = 0.0
        scan_pose.pose.orientation.w = 0.0

        self.move_robot(scan_pose)

    def get_pose(self, detected_object):
        """Get the pose of the detected object."""
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = detected_object.center_3d[0]
        pose.pose.position.y = detected_object.center_3d[1]
        pose.pose.position.z = detected_object.center_3d[2]
        pose.pose.orientation.x = -0.7071
        pose.pose.orientation.y = 0.7071
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        return pose
    
    def sweep_left(self, sweep_pose):
        """Sweep the robot end effector in a straight line."""

        sweep_hover_pose = PoseStamped()
        sweep_hover_pose.header.frame_id = "world"
        sweep_hover_pose.pose.position.x = sweep_pose.pose.position.x
        sweep_hover_pose.pose.position.y = sweep_pose.pose.position.y + 0.1
        sweep_hover_pose.pose.position.z = sweep_pose.pose.position.z
        sweep_hover_pose.pose.orientation = sweep_pose.pose.orientation

        self.move_robot(sweep_hover_pose)

        start_sweep_pose = PoseStamped()
        start_sweep_pose.header.frame_id = "world"
        start_sweep_pose.pose.position.x = sweep_hover_pose.pose.position.x
        start_sweep_pose.pose.position.y = sweep_hover_pose.pose.position.y
        actual_z = sweep_hover_pose.pose.position.z - 0.025
        actual_z = actual_z * 0.25
        start_sweep_pose.pose.position.z = actual_z
        start_sweep_pose.pose.orientation = sweep_hover_pose.pose.orientation

        self.get_logger().info(f"start sweep pose z: {start_sweep_pose.pose.position.z}")
        self.move_robot(start_sweep_pose)

        end_sweep_pose = PoseStamped()
        end_sweep_pose.header.frame_id = "world"
        end_sweep_pose.pose.position.x = start_sweep_pose.pose.position.x
        end_sweep_pose.pose.position.y = start_sweep_pose.pose.position.y - 0.35
        end_sweep_pose.pose.position.z = actual_z
        end_sweep_pose.pose.orientation = start_sweep_pose.pose.orientation

        self.get_logger().info(f"end sweep pose z: {end_sweep_pose.pose.position.z}")
        self.move_robot(end_sweep_pose)

    def sweep_right(self, sweep_pose):
        """Sweep the robot end effector in a straight line."""

        sweep_hover_pose = PoseStamped()
        sweep_hover_pose.header.frame_id = "world"
        sweep_hover_pose.pose.position.x = sweep_pose.pose.position.x
        sweep_hover_pose.pose.position.y = sweep_pose.pose.position.y - 0.1
        sweep_hover_pose.pose.position.z = sweep_pose.pose.position.z
        sweep_hover_pose.pose.orientation = sweep_pose.pose.orientation

        self.move_robot(sweep_hover_pose)

        start_sweep_pose = PoseStamped()
        start_sweep_pose.header.frame_id = "world"
        start_sweep_pose.pose.position.x = sweep_hover_pose.pose.position.x
        start_sweep_pose.pose.position.y = sweep_hover_pose.pose.position.y
        actual_z = sweep_hover_pose.pose.position.z - 0.025
        actual_z = actual_z * 0.25
        start_sweep_pose.pose.position.z = actual_z
        start_sweep_pose.pose.orientation = sweep_hover_pose.pose.orientation

        self.move_robot(start_sweep_pose)

        end_sweep_pose = PoseStamped()
        end_sweep_pose.header.frame_id = "world"
        end_sweep_pose.pose.position.x = start_sweep_pose.pose.position.x
        end_sweep_pose.pose.position.y = start_sweep_pose.pose.position.y + 0.35
        end_sweep_pose.pose.position.z = actual_z
        end_sweep_pose.pose.orientation = start_sweep_pose.pose.orientation

        self.move_robot(end_sweep_pose)


# Derived Class for Franka Robot
class FrankaInterface(RobotInterface):
    def __init__(self):
        super().__init__('franka_interface', '/franka_motion_planning', '/detect_objects_d415')
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
        home_pose.pose.position.x = -0.95
        home_pose.pose.position.y = -0.01
        home_pose.pose.position.z = 0.512
        home_pose.pose.orientation.x = 1.0
        home_pose.pose.orientation.y = 0.0
        home_pose.pose.orientation.z = 0.0
        home_pose.pose.orientation.w = 0.0

        self.move_robot(home_pose)

    def move_to_scan_pose(self, dummy=False):
        """Move Franka Robot to Scan Pose."""
        if dummy:
            self.get_logger().info("[Dummy] moving Franka robot to scan pose")
            return
        
        scan_pose = PoseStamped()
        scan_pose.header.frame_id = "world"
        scan_pose.pose.position.x = -0.635
        scan_pose.pose.position.y = -0.01
        scan_pose.pose.position.z = 0.512
        scan_pose.pose.orientation.x = 1.0
        scan_pose.pose.orientation.y = 0.0
        scan_pose.pose.orientation.z = 0.0
        scan_pose.pose.orientation.w = 0.0

        self.move_robot(scan_pose)

    def get_pose(self, detected_object):
        """Get the pose of the detected object."""
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = detected_object.center_3d[0]
        pose.pose.position.y = detected_object.center_3d[1]
        pose.pose.position.z = detected_object.center_3d[2]
        pose.pose.orientation.x = 1.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        return pose

    def move_gripper(self, target_width, speed):
        """Move the Robot Gripper to the target width."""
        goal_msg = Move.Goal()
        goal_msg.width = target_width
        goal_msg.speed = speed

        self.get_logger().info(f'Sending gripper goal: width={target_width} m, speed={speed} m/s')
        goal_future = self.gripper_action_client.send_goal_async(goal_msg)

        # Wait for the future to complete
        rclpy.spin_until_future_complete(self, goal_future)
        # Check the result
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Gripper goal was rejected")
            return
        # Get the result future
        result_future = goal_handle.get_result_async()
        # Wait for the result to complete
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Gripper goal was executed")


    def open_gripper(self):
        """Open the Robot Gripper."""
        self.move_gripper(target_width=0.07, speed=0.5)

    def close_gripper(self):
        """Close the Robot Gripper."""
        self.move_gripper(target_width=0.0, speed=0.5)

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
            robot_interface.move_to_scan_pose()
            detections = robot_interface.detect_objects()
            scene_description = detections.get_scene_description()
            print(f"Scene Description: {scene_description}")
            user_prompt += scene_description
            user_prompt += "\n[Instruction]\n"
            user_prompt += user_task
    
            llm_output = robot_interface.llm_router(user_prompt)

            # dummy code for testing
            # dummy_code = """robot_interface.close_gripper()\nrobot_interface.open_gripper()"""
            # llm_output = ["Dummy Reasoning", dummy_code]

            reasoning, valid_code = llm_output
            print(f"Reasoning: {reasoning}")
            if valid_code:
                print(f"Valid Code: {valid_code}")
                if_execute = input("Execute Code? (Y/N): ").strip().lower()
                if if_execute == "y":
                    exec(valid_code)
                time.sleep(2)

    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        robot_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
