from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("franka_bringup"),
                        "launch",
                        "franka_sim_moveit.launch.py",
                    ]
                )
            )
        )
    )

    ld.add_action(
        Node(
            package="icdt_wrapper",
            executable="wrapper",
            name="icdt_wrapper",
            output="screen",
        )
    )

    return ld