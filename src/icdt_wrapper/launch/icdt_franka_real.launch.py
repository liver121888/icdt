import os
import xacro
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    PathJoinSubstitution, 
    Command, 
    FindExecutable
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
                        FindPackageShare("franka_bringup"),
                        "launch",
                        "franka_real_moveit.launch.py",
                    ]
                )
            )
        )
    )
    
    franka_xacro_filepath = os.path.join(get_package_share_directory(
        'franka_description'), 'robots', "fr3", "fr3"+'.urdf.xacro')
    robot_description_config = xacro.process_file(franka_xacro_filepath,
                                           mappings={
                                            'arm_id': "fr3",
                                            'hand': "true",
                                            'ros2_control': 'true',
                                            'gazebo': 'true',
                                            'ee_id': 'franka_hand',
                                            'gazebo_effort': 'true',
                                            'use_fake_hardware': "false",
                                            'fake_sensor_commands': "false",
                                           }
                                           ).toprettyxml(indent='  ')
    robot_description = {'robot_description': robot_description_config}

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'srdf',
        'fr3_arm.srdf.xacro'
    )

    robot_description_semantic_config = Command(
        [FindExecutable(name='xacro'), ' ',
         franka_semantic_xacro_file, ' hand:=true']
    )

    robot_description_semantic = {'robot_description_semantic': ParameterValue(
    robot_description_semantic_config, value_type=str)}

    ld.add_action(
        Node(
            package="icdt_wrapper",
            executable="robot_motion_planning",
            name="franka_robot_motion_planning",
            namespace="",
            output="screen",
            parameters=[
                PathJoinSubstitution(
                    [
                        FindPackageShare("icdt_wrapper"),
                        "config",
                        "franka_robot_motion_planning.yaml",
                    ]
                ),
                {"use_sim_time": False},
                robot_description,
                robot_description_semantic
            ],
        )
    )

    return ld