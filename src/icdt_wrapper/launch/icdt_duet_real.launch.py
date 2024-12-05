import os
import xacro
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument
)
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    PathJoinSubstitution, 
    Command, 
    LaunchConfiguration,
    TextSubstitution,
    FindExecutable
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
# from moveit_configs_utils import MoveItConfigsBuilder

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

    # franka
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
            ),
            launch_arguments={"use_rviz": "false"}.items(),
        )
    )

    # lbr
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
            launch_arguments={"rviz": "false"}.items(),
        )
    )
    
    # franka
    franka_xacro_filepath = os.path.join(get_package_share_directory(
        'franka_description'), 'robots', "fr3", "fr3"+'.urdf.xacro')
    franka_robot_description_config = xacro.process_file(franka_xacro_filepath,
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
    franka_robot_description = {'robot_description': franka_robot_description_config}

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'srdf',
        'fr3_arm.srdf.xacro'
    )

    franka_robot_description_semantic_config = Command(
        [FindExecutable(name='xacro'), ' ',
         franka_semantic_xacro_file, ' hand:=true']
    )

    franka_robot_description_semantic = {'robot_description_semantic': ParameterValue(
    franka_robot_description_semantic_config, value_type=str)}

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
                franka_robot_description,
                franka_robot_description_semantic
            ],
        )
    )


    # lbr
    robot_description = LBRDescriptionMixin.param_robot_description(sim=False)

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
                {"use_sim_time": False},
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

    # moveit_config = (
    #     # robot_name: dual_arm_panda
    #     MoveItConfigsBuilder("icdt_duet")
    #     .robot_description(file_path="config/panda.urdf.xacro")
    #     .robot_description_semantic(file_path="config/panda.srdf")
    #     .trajectory_execution(file_path="config/moveit_controllers.yaml")
    #     .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
    #     .to_moveit_configs()
    # )

    # franka_robot_description,
    # franka_robot_description_semantic,
    # robot_description,
    # robot_description_semantic,
    # ompl_planning_pipeline_config,
    # kinematics_yaml,

    kinematics_yaml = load_yaml(
        'icdt_wrapper', 'config/kinematics.yaml'
    )

    planning_yaml = load_yaml(
        'icdt_wrapper', 'config/planning.yaml'
    )

    rviz_file = os.path.join(get_package_share_directory('icdt_wrapper'), 'config',
                             'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['--display-config', rviz_file],
        parameters=[
            # moveit_config.robot_description,
            # moveit_config.robot_description_semantic,
            kinematics_yaml,
            planning_yaml,
            robot_description,
            robot_description_semantic,
            franka_robot_description,
            franka_robot_description_semantic,
            # ompl_planning_pipeline_config,
            # kinematics_yaml,
        ],
        ros_arguments=['--log-level', 'warn'],
    )

    ld.add_action(rviz_node)

    # cameras
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

    arg_name_d415 = DeclareLaunchArgument('nameD415',             
                default_value=PathJoinSubstitution([
                FindPackageShare('franka_bringup'),  # Finds the install/share directory for your package
                TextSubstitution(text='config/eih_cam1')  # Appends the relative path to your file
            ]),)

    handeye_publisher_d415 = Node(package='easy_handeye2', executable='handeye_publisher', name='handeye_publisher_d415', parameters=[{
        'name': LaunchConfiguration('nameD415'),
    }])

    ld.add_action(arg_name_d415)
    ld.add_action(handeye_publisher_d415) 

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
                # 'serial_no2': '_242322073889',
                'serial_no2': '_143122064672',
                'camera_name1': 'camera',
                'camera_name2': 'D415',
                'camera_namespace1': '',
                'camera_namespace2': '',
            }.items(),
        )
    )


    return ld