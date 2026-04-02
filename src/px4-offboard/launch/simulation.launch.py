import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'px4_offboard'
    
    # Paths
    px4_dir = os.path.expanduser('~/PX4-Autopilot')
    
    # 1. Micro-XRCE-DDS Agent
    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )
    
    # 2. PX4 SITL (with Gazebo)
    # Using 'make' command as requested. PX4_GZ_WORLD is set to forest.
    px4_sitl = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500_lidar_2d'],
        cwd=px4_dir,
        additional_env={'PX4_GZ_WORLD': 'forest'},
        output='screen'
    )
    
    # 3. ROS-GZ Bridge for LiDAR and Clock
    # Topic based on the user's existing logic and Gazebo world name.
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/forest/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )

    return LaunchDescription([
        micro_xrce_agent,
        px4_sitl,
        ros_gz_bridge
    ])
