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
  MoveItPlanningNode() : Node("robot_motion_planning")
  {
  }

  std::shared_ptr<MoveItPlanningNode> shared_from_this()
  {
    return std::static_pointer_cast<MoveItPlanningNode>(Node::shared_from_this());
  }

  void initialize()
  {
    // Declare planning scene parameters
    {
      this->declare_parameter<std::string>(PLANNING_SCENE_MONITOR_NS + "name", UNDEFINED);
      this->declare_parameter<std::string>(PLANNING_SCENE_MONITOR_NS + "robot_description", UNDEFINED);
      this->declare_parameter<std::string>(PLANNING_SCENE_MONITOR_NS + "joint_state_topic", UNDEFINED);
      this->declare_parameter<std::string>(PLANNING_SCENE_MONITOR_NS + "attached_collision_object_topic", UNDEFINED);
      this->declare_parameter<std::string>(PLANNING_SCENE_MONITOR_NS + "publish_planning_scene_topic", UNDEFINED);
      this->declare_parameter<std::string>(PLANNING_SCENE_MONITOR_NS + "monitored_planning_scene_topic", UNDEFINED);
      this->declare_parameter<double>(PLANNING_SCENE_MONITOR_NS + "wait_for_initial_state_timeout", 10.0);
    }

    // Declare planning group parameters
    {
      this->declare_parameter<std::string>("planning_group", UNDEFINED);
    }

    // Declare planning pipeline parameters
    {
      this->declare_parameter<std::vector<std::string>>(PLANNING_PIPELINES_NS + "pipeline_names",
                                                        std::vector<std::string>({ UNDEFINED }));
      this->declare_parameter<std::string>(PLANNING_PIPELINES_NS + "namespace", UNDEFINED);
    }

    // Declare planning pipeline OMPL parameters
    {
      this->declare_parameter<std::vector<std::string>>(
          "ompl.arm.planner_configs",
          std::vector<std::string>(
              { "SBLkConfigDefault",         "ESTkConfigDefault",   "LBKPIECEkConfigDefault",   "BKPIECEkConfigDefault",
                "KPIECEkConfigDefault",      "RRTkConfigDefault",   "RRTConnectkConfigDefault", "RRTstarkConfigDefault",
                "TRRTkConfigDefault",        "PRMkConfigDefault",   "PRMstarkConfigDefault",    "FMTkConfigDefault",
                "BFMTkConfigDefault",        "PDSTkConfigDefault",  "STRIDEkConfigDefault",     "BiTRRTkConfigDefault",
                "LBTRRTkConfigDefault",      "BiESTkConfigDefault", "ProjESTkConfigDefault",    "LazyPRMkConfigDefault",
                "LazyPRMstarkConfigDefault", "SPARSkConfigDefault", "SPARStwokConfigDefault",   "TrajOptDefault" }));

      this->declare_parameter<std::string>("ompl.planner_configs.SBLkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.ESTkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.LBKPIECEkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.BKPIECEkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.KPIECEkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.RRTkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.RRTConnectkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.RRTstarkConfigDefault.type", UNDEFINED);
      this->declare_parameter<std::string>("ompl.planner_configs.TRRTkConfigDefault.type", "geometric::TRRT");
      this->declare_parameter<std::string>("ompl.planner_configs.PRMkConfigDefault.type", "geometric::PRM");
      this->declare_parameter<std::string>("ompl.planner_configs.PRMstarkConfigDefault.type", "geometric::PRMstar");
      this->declare_parameter<std::string>("ompl.planner_configs.FMTkConfigDefault.type", "geometric::FMT");
      this->declare_parameter<std::string>("ompl.planner_configs.BFMTkConfigDefault.type", "geometric::BFMT");
      this->declare_parameter<std::string>("ompl.planner_configs.PDSTkConfigDefault.type", "geometric::PDST");
      this->declare_parameter<std::string>("ompl.planner_configs.STRIDEkConfigDefault.type", "geometric::STRIDE");
      this->declare_parameter<std::string>("ompl.planner_configs.BiTRRTkConfigDefault.type", "geometric::BiTRRT");
      this->declare_parameter<std::string>("ompl.planner_configs.LBTRRTkConfigDefault.type", "geometric::LBTRRT");
      this->declare_parameter<std::string>("ompl.planner_configs.BiESTkConfigDefault.type", "geometric::BiEST");
      this->declare_parameter<std::string>("ompl.planner_configs.ProjESTkConfigDefault.type", "geometric::ProjEST");
      this->declare_parameter<std::string>("ompl.planner_configs.LazyPRMkConfigDefault.type", "geometric::LazyPRM");
      this->declare_parameter<std::string>("ompl.planner_configs.LazyPRMstarkConfigDefault.type",
                                           "geometric::LazyPRMstar");
      this->declare_parameter<std::string>("ompl.planner_configs.SPARSkConfigDefault.type", "geometric::SPARS");
      this->declare_parameter<std::string>("ompl.planner_configs.SPARStwokConfigDefault.type", "geometric::SPARStwo");
      this->declare_parameter<std::string>("ompl.planner_configs.TrajOptDefault.type", "geometric::TrajOpt");
      this->declare_parameter<std::string>("ompl.arm.projection_evaluator", "joints(A1, A2, A3, A4, A5, A6, A7)");

      this->declare_parameter<std::vector<std::string>>("ompl.planning_plugins",
                                                        std::vector<std::string>({ UNDEFINED }));
      this->declare_parameter<std::string>("ompl.planning_plugin", UNDEFINED);

      this->declare_parameter<std::string>("ompl.request_adapters", UNDEFINED);
      this->declare_parameter<std::string>("ompl.response_adapters", UNDEFINED);
      this->declare_parameter<double>("ompl.start_state_max_bounds_error", 0.1);
    }

    // Declare pilz_industrial_motion_planner parameters
    {
      this->declare_parameter<std::string>("pilz_industrial_motion_planner.capabilities", UNDEFINED);
      this->declare_parameter<std::string>("pilz_industrial_motion_planner.default_planner_config", UNDEFINED);
      this->declare_parameter<std::vector<std::string>>("pilz_industrial_motion_planner.planning_plugins",
                                                        std::vector<std::string>({ UNDEFINED }));
      this->declare_parameter<std::string>("pilz_industrial_motion_planner.planning_plugin", UNDEFINED);
      this->declare_parameter<std::string>("pilz_industrial_motion_planner.request_adapters", UNDEFINED);
      this->declare_parameter<double>("pilz_industrial_motion_planner.cartesian_limits.max_trans_vel", 1.0);
      this->declare_parameter<double>("pilz_industrial_motion_planner.cartesian_limits.max_trans_acc", 2.25);
      this->declare_parameter<double>("pilz_industrial_motion_planner.cartesian_limits.max_trans_dec", -5.0);
      this->declare_parameter<double>("pilz_industrial_motion_planner.cartesian_limits.max_rot_vel", 1.57);
    }

    // For IK calculation
    {
      this->declare_parameter<std::string>("robot_description_kinematics.arm.kinematics_solver", "pick_ik/"
                                                                                                 "PickIkPlugin");
      this->declare_parameter<double>("robot_description_kinematics.arm.kinematics_solver_timeout", 0.05);
      this->declare_parameter<int>("robot_description_kinematics.arm.kinematics_solver_attempts", 3);
      this->declare_parameter<std::string>("robot_description_kinematics.arm.mode", "global");
      this->declare_parameter<double>("robot_description_kinematics.arm.position_scale", 1.0);
      this->declare_parameter<double>("robot_description_kinematics.arm.rotation_scale", 0.5);
      this->declare_parameter<double>("robot_description_kinematics.arm.position_threshold", 0.001);
      this->declare_parameter<double>("robot_description_kinematics.arm.orientation_threshold", 0.01);
      this->declare_parameter<double>("robot_description_kinematics.arm.cost_threshold", 0.001);
      this->declare_parameter<double>("robot_description_kinematics.arm.minimal_displacement_weight", 0.0);
      this->declare_parameter<double>("robot_description_kinematics.arm.gd_step_size", 0.0001);
    }

    // Declare PlanRequestParameters
    {
      this->declare_parameter<std::string>(PLAN_REQUEST_PARAM_NS + "planner_id", UNDEFINED);
      this->declare_parameter<std::string>(PLAN_REQUEST_PARAM_NS + "planning_pipeline", UNDEFINED);
      this->declare_parameter<int>(PLAN_REQUEST_PARAM_NS + "planning_attempts", 1);
      this->declare_parameter<double>(PLAN_REQUEST_PARAM_NS + "planning_time", 1.0);
      this->declare_parameter<double>(PLAN_REQUEST_PARAM_NS + "max_velocity_scaling_factor", 1.0);
      this->declare_parameter<double>(PLAN_REQUEST_PARAM_NS + "max_acceleration_scaling_factor", 1.0);
    }

    // Trajectory Execution Functionality (required by the MoveItPlanningPipeline but not used within hybrid planning)
    this->declare_parameter<std::string>("moveit_controller_manager", UNDEFINED);
    // this->declare_parameter<bool>("allow_trajectory_execution", true);
    // this->declare_parameter<bool>("moveit_manage_controllers", true);

    using std::placeholders::_1;
    using std::placeholders::_2;
    MotionPlanningService = this->create_service<icdt_interfaces::srv::MotionPlanning>(
        "motion_planning", std::bind(&MoveItPlanningNode::planMotion, this, _1, _2));

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

    RCLCPP_INFO(LOGGER, "Goal Pose Position: x: %f, y: %f, z: %f", goalPose.pose.position.x, goalPose.pose.position.y,
                goalPose.pose.position.z);

    RCLCPP_INFO(LOGGER, "Goal Pose Orientation: x: %f, y: %f, z: %f, w: %f", goalPose.pose.orientation.x,
                goalPose.pose.orientation.y, goalPose.pose.orientation.z, goalPose.pose.orientation.w);

    bool ik_success = goal_state_->setFromIK(joint_model_group_.get(), goalPose.pose);

    RCLCPP_INFO(LOGGER, "IK success: %d", ik_success);
    
    if (!ik_success)
    {
      response->success = false;
      return;
    }

    std::vector<double> joint_values;
    goal_state_->copyJointGroupPositions(joint_model_group_.get(), joint_values);
    for (size_t i = 0; i < joint_values.size(); ++i)
    {
      RCLCPP_INFO(LOGGER, "Joint %ld: %f", i+1, joint_values[i]);
    }

    moveit_msgs::msg::Constraints goal_constraints =
        kinematic_constraints::constructGoalConstraints(*(goal_state_.get()), joint_model_group_.get());

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
  const std::string PLANNING_SCENE_MONITOR_NS = "planning_scene_monitor_options.";
  const std::string PLANNING_PIPELINES_NS = "planning_pipelines.";
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