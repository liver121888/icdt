import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    detection_node_1 = Node(
        package='perception_server',
        executable='detection_service',
        name='d405',
        parameters=[
            {'node_name': 'd405'},
            {'rgb_topic': '/camera/color/image_rect_raw'},
            {'depth_topic': '/camera/aligned_depth_to_color/image_raw'},
            {'detection_viz_topic': '/camera/color/detections'},
            {'pose_array_topic': '/camera/color/pose_array'},
            {'camera_frame_id': 'camera_color_optical_frame'},
            {'world_frame_id': 'world'},
            {'width': 848},
            {'intrinsics': [
                425.19189453125,
                424.6562805175781,
                422.978515625,
                242.1155242919922
            ]},
            {'orientation': [
                0.0,
                1.0,
                0.0,
                0.0
            ]}
        ]
    )

    detection_node_2 = Node(
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
    

    return LaunchDescription([detection_node_1, detection_node_2])