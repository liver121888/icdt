#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from rclpy.duration import Duration
from icdt_interfaces.srv import MotionPlanning

class Wrapper(Node):

    def __init__(self):
        super().__init__('icdt_wrapper')

        self.motion_planning_cli = self.create_client(MotionPlanning, '/lbr/motion_planning')
        while not self.motion_planning_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.motion_planning_req = MotionPlanning.Request()

        self.get_clock().sleep_for(Duration(seconds=10))
        self.get_logger().info('Start execution')
        self.execution()

    def run_llm(self, prompt):
        return "dummy", prompt

    def execution(self):
        # exe just once for now
        # while True:

        # prompt = """objects = detect_objects(image)
        # target_object = 'box'
        # if target_object in objects:
        #     move_robot(target_object.pose)"""

        prompt = """self.move_robot(Pose())"""

        llm_output = self.run_llm(prompt)
        reasoning, valid_code = llm_output
        self.get_logger().info(f"Reasoning: {reasoning}")
        if valid_code:
            eval(valid_code)

    # move robot from current pose to target pose
    def move_robot(self, target_pose):

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "link_0"
        goal_pose.pose.position.x = -0.4
        goal_pose.pose.position.y = 0.25
        goal_pose.pose.position.z = 0.36
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 1.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 0.0

        self.motion_planning_req.goal = goal_pose
        # we want it blocking
        self.get_logger().info("Sending goal to motion planning service")
        # response = self.motion_planning_cli.call(self.motion_planning_req)
        # self.get_logger().info(
        #     'Result of motion_planning service: %d', response.success)

        # Call the service and wait for the result
        future = self.motion_planning_cli.call_async(self.motion_planning_req)
        rclpy.spin_until_future_complete(self, future)

        # Check if we received a response
        if future.done():
            response = future.result()
            self.get_logger().info(f'Result of motion_planning service: {response.success}')
        else:
            self.get_logger().error("Failed to receive a response from motion planning service")



def main(args=None):
    rclpy.init(args=args)
    wrapper = Wrapper()
    rclpy.spin(wrapper)
    wrapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()