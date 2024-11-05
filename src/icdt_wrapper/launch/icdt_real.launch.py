import os
import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution, LaunchConfiguration
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
                        "real.launch.py",
                    ]
                )
            ),
        )
    )


    arg_name = DeclareLaunchArgument('name',             
                default_value=PathJoinSubstitution([
                FindPackageShare('paradocs_control'),  # Finds the install/share directory for your package
                TextSubstitution(text='config/eih_cam1')  # Appends the relative path to your file
            ]),)

    handeye_publisher = Node(package='easy_handeye2', executable='handeye_publisher', name='handeye_publisher', parameters=[{
        'name': LaunchConfiguration('name'),
    }])

    ld.add_action(arg_name)
    ld.add_action(handeye_publisher)

    arg_name_d435 = DeclareLaunchArgument('nameD435',             
                default_value=PathJoinSubstitution([
                FindPackageShare('paradocs_control'),  # Finds the install/share directory for your package
                TextSubstitution(text='config/eih_cam2')  # Appends the relative path to your file
            ]),)

    handeye_publisher_d435 = Node(package='easy_handeye2', executable='handeye_publisher', name='handeye_publisher_d435', parameters=[{
        'name': LaunchConfiguration('nameD435'),
    }])

    ld.add_action(arg_name_d435)
    ld.add_action(handeye_publisher_d435) 

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("paradocs_control"),
                        "launch",
                        "static_obstacles.launch.py",
                    ]
                )
            ),
        )
    )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("paradocs_control"),
                        "launch",
                        "rs_dual_camera_launch.py",
                    ]
                )
            ),
            launch_arguments={
                'serial_no1': '_128422270653',
                'serial_no2': '_242322073889',
                'camera_name1': 'camera',
                'camera_name2': 'D435',
                'camera_namespace1': '',
                'camera_namespace2': '',
            }.items(),
        )
    )


    # now testing aruco_pose, uncomment when doing the drill_pose
    drill_pose_transformer = Node(package='paradocs_control', executable='drill_pose_transformer.py', name='pose_transformer')
    ld.add_action(drill_pose_transformer)

    # serial_writer = Node(package='serialcomm', executable='serialwriter_exec', name='serial_writer')
    # ld.add_action(serial_writer)

    robot_description = LBRDescriptionMixin.param_robot_description(sim=True)

    robot_description_semantic = {
        "robot_description_semantic": load_file("med7_moveit_config", "config/med7.srdf")
    }

    robot_motion_planning = Node(
        package="icdt_wrapper",
        executable="robot_motion_planning",
        name="robot_motion_planning",
        namespace="/lbr",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("icdt_wrapper"),
                    "config",
                    "robot_motion_planning.yaml",
                ]
            ),
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
    ld.add_action(robot_motion_planning)

    detection_node = Node(
        package="perception_server",
        executable="detection_service",
        name="detection_service",
        output="screen",
    )
    ld.add_action(detection_node)


    return ld
