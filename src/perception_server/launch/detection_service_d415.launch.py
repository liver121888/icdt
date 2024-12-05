import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    detection_node_1 = Node(
        package='perception_server',
        executable='detection_service',
        name='d415',
        parameters=[
            {'node_name': 'd415'},
            {'rgb_topic': '/D415/color/image_raw'},
            {'depth_topic': '/D415/aligned_depth_to_color/image_raw'},
            {'detection_viz_topic': '/D415/color/detections'},
            {'pose_array_topic': '/D415/color/pose_array'},
            {'camera_frame_id': 'D415_color_optical_frame'},
            {'world_frame_id': 'world'},
            {'width': 1280},
            {'intrinsics': [
                910.8426,
                908.3326,
                636.2789,
                358.2963
            ]},
            {'orientation': [
                1.0,
                0.0,
                0.0,
                0.0
            ]}
        ]
    )

    return LaunchDescription([detection_node_1])