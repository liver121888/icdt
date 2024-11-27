import os
import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

from lbr_description import LBRDescriptionMixin

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return file.read()
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("lbr_bringup"),
                        "launch",
                        "sim.launch.py",
                    ]
                )
            )
        )
    )

    robot_description = LBRDescriptionMixin.param_robot_description(sim=True)

    robot_description_semantic = {
        "robot_description_semantic": load_file("med7_moveit_config", "config/med7.srdf")
    }

    ld.add_action(
        Node(
            package="icdt_wrapper",
            executable="robot_motion_planning",
            name="lbr_robot_motion_planning",
            namespace="/lbr",
            output="screen",
            parameters=[
                PathJoinSubstitution(
                    [
                        FindPackageShare("icdt_wrapper"),
                        "config",
                        "lbr_robot_motion_planning.yaml",
                    ]
                ),
                {"use_sim_time": True},
                robot_description,
                robot_description_semantic
            ],
            remappings=[
                    ("lbr/attached_collision_object", "/attached_collision_object"),
                    ("lbr/joint_states", "/joint_states"),
                    ("lbr/monitored_planning_scene", "/monitored_planning_scene"),
                    ("lbr/planning_scene", "/planning_scene"),
                    ("/display_planned_path", "/lbr/display_planned_path"),
                    ("/display_contacts", "/lbr/display_contacts"),
                    ("/trajectory_execution_event", "/lbr/trajectory_execution_event"),
                    ("/joint_trajectory_controller/joint_trajectory", "/lbr/joint_trajectory_controller/joint_trajectory"),
                    ("lbr/collision_object", "/collision_object"),
            ],
        )
    )

    return ld