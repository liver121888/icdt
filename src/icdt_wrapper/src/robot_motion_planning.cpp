/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2021, PickNik Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PickNik Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/moveit_cpp/planning_component.h>
#include "icdt_interfaces/srv/motion_planning.hpp"

class MoveItPlanningNode : public rclcpp::Node
{
public:

  MoveItPlanningNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
      .automatically_declare_parameters_from_overrides(true)
      .allow_undeclared_parameters(true))
    : Node("robot_motion_planning", options)
  {
    RCLCPP_INFO(LOGGER, "automatically_declare_parameters_from_overrides: %d", Node::get_node_options().automatically_declare_parameters_from_overrides());
    RCLCPP_INFO(LOGGER, "allow_undeclared_parameters: %d", Node::get_node_options().allow_undeclared_parameters());
  }

  std::shared_ptr<MoveItPlanningNode> shared_from_this()
  {
    return std::static_pointer_cast<MoveItPlanningNode>(Node::shared_from_this());
  }

  void initialize()
  {

    using std::placeholders::_1;
    using std::placeholders::_2;

    static const std::string prefix = this->get_parameter("prefix").as_string();

    MotionPlanningService = this->create_service<icdt_interfaces::srv::MotionPlanning>(
        prefix + "_motion_planning", std::bind(&MoveItPlanningNode::planMotion, this, _1, _2));

    // Initialize MoveItCpp API
    moveit_cpp::MoveItCpp::Options moveit_cpp_options(this->shared_from_this());
    moveit_cpp_ = std::make_shared<moveit_cpp::MoveItCpp>(this->shared_from_this(), moveit_cpp_options);

    planning_group_ = this->get_parameter("planning_group").as_string();
    
    robot_model_ = moveit_cpp_->getRobotModel();
    goal_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    joint_model_group_ = std::shared_ptr<const moveit::core::JointModelGroup>(goal_state_->getJointModelGroup(planning_group_));
    planning_component_ = std::make_shared<moveit_cpp::PlanningComponent>(planning_group_, moveit_cpp_);

  }

  void planMotion(const std::shared_ptr<icdt_interfaces::srv::MotionPlanning::Request> request,
            std::shared_ptr<icdt_interfaces::srv::MotionPlanning::Response> response)
  {

    // Set parameters required by the planning component
    moveit_cpp::PlanningComponent::PlanRequestParameters plan_params;
    plan_params.planner_id = this->get_parameter(PLAN_REQUEST_PARAM_NS + "planner_id").as_string();
    plan_params.planning_pipeline = this->get_parameter(PLAN_REQUEST_PARAM_NS + "planning_pipeline").as_string();
    plan_params.planning_attempts = this->get_parameter(PLAN_REQUEST_PARAM_NS + "planning_attempts").as_int();
    plan_params.planning_time = this->get_parameter(PLAN_REQUEST_PARAM_NS + "planning_time").as_double();
    plan_params.max_velocity_scaling_factor =
        this->get_parameter(PLAN_REQUEST_PARAM_NS + "max_velocity_scaling_factor").as_double();
    plan_params.max_acceleration_scaling_factor =
        this->get_parameter(PLAN_REQUEST_PARAM_NS + "max_acceleration_scaling_factor").as_double();

    // update planning scene with current state
    moveit_cpp_->getPlanningSceneMonitor()->updateSceneWithCurrentState();

    // Set start state to current state
    planning_component_->setStartStateToCurrentState();

    // Copy goal constraint into planning component
    auto goalPose = request->goal;

    // RCLCPP_INFO(LOGGER, "Goal Pose Position: x: %f, y: %f, z: %f", goalPose.pose.position.x, goalPose.pose.position.y,
    //             goalPose.pose.position.z);

    // RCLCPP_INFO(LOGGER, "Goal Pose Orientation: x: %f, y: %f, z: %f, w: %f", goalPose.pose.orientation.x,
    //             goalPose.pose.orientation.y, goalPose.pose.orientation.z, goalPose.pose.orientation.w);

    bool ik_success = goal_state_->setFromIK(joint_model_group_.get(), goalPose.pose);

    RCLCPP_INFO(LOGGER, "IK success: %d", ik_success);
    
    if (!ik_success)
    {
      RCLCPP_ERROR(LOGGER, "IK failed");
      response->success = false;
      return;
    }
    else
    {
      RCLCPP_INFO(LOGGER, "IK success");
    }

    std::vector<double> joint_values;
    goal_state_->copyJointGroupPositions(joint_model_group_.get(), joint_values);
    for (size_t i = 0; i < joint_values.size(); ++i)
    {
      RCLCPP_INFO(LOGGER, "Joint %ld: %f", i+1, joint_values[i]);
    }

    moveit_msgs::msg::Constraints goal_constraints =
        kinematic_constraints::constructGoalConstraints(*(goal_state_.get()), joint_model_group_.get(), 1e-4);

    // if don't use IK
    // moveit_msgs::msg::Constraints goal_constraints =
    //     kinematic_constraints::constructGoalConstraints("link_tool", goalPose);

    planning_component_->setGoal({ goal_constraints });

    // Plan motion
    auto plan_solution = planning_component_->plan(plan_params);
    if (plan_solution.error_code == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      // isblocking or not
      planning_component_->execute(true);
      rclcpp::sleep_for(std::chrono::seconds(1));
      response->success = true;
    }
    else
    {
      response->success = false;
    }
  }

private:
  const rclcpp::Logger LOGGER = rclcpp::get_logger("robot_motion_planning");
  std::shared_ptr<moveit_cpp::MoveItCpp> moveit_cpp_;
  // const std::string PLANNING_SCENE_MONITOR_NS = "planning_scene_monitor_options.";
  // const std::string PLANNING_PIPELINES_NS = "planning_pipelines.";
  const std::string PLAN_REQUEST_PARAM_NS = "plan_request_params.";
  const std::string UNDEFINED = "<undefined>";
  rclcpp::Service<icdt_interfaces::srv::MotionPlanning>::SharedPtr MotionPlanningService;

  std::shared_ptr<const moveit::core::JointModelGroup> joint_model_group_;

  // Robot model
  std::shared_ptr<const moveit::core::RobotModel> robot_model_;

  // Planning group
  std::string planning_group_;

  // Goal from IK calculation
  std::shared_ptr<moveit::core::RobotState> goal_state_;
  std::shared_ptr<moveit_cpp::PlanningComponent> planning_component_;

};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveItPlanningNode>();

  // Call initialize after creating the shared pointer instance
  node->initialize();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}