from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pid_tune',
            executable='pid_tune_differential.py',
            name='tune_pid_differential',
            parameters=[
                {'lpid_ui_enable': True},
                {'rpid_ui_enable': True},
            ],
        ),
    ])
