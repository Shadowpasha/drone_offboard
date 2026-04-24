#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, TwistStamped, Point
from std_msgs.msg import String
from std_msgs.msg import Empty as Empty_msg
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
import math
import time
import random
import threading
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import os
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus
from rclpy.clock import Clock
import multiprocessing
from .visualizer import start_visualizer

class RealDroneEnv(gym.Env):
    def __init__(self):
        # QoS profile for LiDAR
        qos_profile_laser = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
       
        rclpy.init()
        self.node = rclpy.create_node("real_drone_env")
        self.goal_range = 3
        
        # QoS profiles for Publishers
        qos_profile_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher_offboard_mode = self.node.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.node.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_vehicle_command = self.node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile_pub)
        self.goal_marker_pub = self.node.create_publisher(Marker, '/goal_marker', 10)
        
        # QoS for PX4 subscribers (Best Effort, Volatile)
        qos_profile_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odometry_sub = self.node.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.vehicle_odometry_callback,
            qos_profile_sub,
        )
        self.status_sub_px4 = self.node.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile_sub,
        )
        self.status_sub_px4_v1 = self.node.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            qos_profile_sub,
        )

        # Real LiDAR topic
        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            "/scan",
            self.get_laser_scan,
            qos_profile_laser
        )
        self.node.get_logger().info(f"LiDAR subscription created on topic: /scan")

        self.pose = Pose()
        self.vel = Twist()
        self.goal = [2.0, 0.0] # Default goal 2m ahead
         
        self.prev_distance = 0.0
        self.distance = 0.0
        self.goal_reached = False
        self.pitch = 0.0
        self.roll = 0.0
        self.trueYaw = 0.0
        self.raw_ned_yaw = float('nan')
        self.goal_heading = 0.0
        self.pos_received = False

        self.done = False
        self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]),(2,),dtype= np.float64) 
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(134,), dtype= np.float64)
        self.laser_done_cnt = 0
        
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        self.et = threading.Thread(target=self.node_spin)
        self.et.daemon = True
        self.et.start()
        
        self._is_closed = False
        self.extracted_row = np.ones(128) * 12.0 # Initialize with max distance

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.offboard_setpoint_counter = 0

        # Maintain offboard mode
        self.timer = self.node.create_timer(0.05, self.cmdloop_callback)
        
        self.target_pos = np.zeros(3) # ENU target setpoint [East, North, Up]
        self.last_action = np.zeros(2) 
        
        # Takeoff/Landing parameters (Smooth/Exponential)
        self.takeoff_speed = 0.05 # m/s (maximum)
        self.takeoff_acceleration = 0.001 # m/s^2
        self.current_z_setpoint = 0.0
        self.dt = 0.05 

        # Visualizer Setup
        self.viz_queue = multiprocessing.Queue(maxsize=1)
        self.viz_proc = multiprocessing.Process(target=start_visualizer, args=(self.viz_queue,))
        self.viz_proc.daemon = True
        self.viz_proc.start()
    def cmdloop_callback(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.publisher_vehicle_command.publish(msg)

    def land(self):
        self.node.get_logger().info("Mission complete. Descending slowly before landing...")
        start_height = self.vehicle_local_position[2]
        self.current_z_setpoint = start_height
        current_descent_speed = 0.01 
        
        target_height = 0.15 # Slightly higher for safety on real drone
        while (self.current_z_setpoint > target_height) and not self._is_closed:
            if current_descent_speed < self.takeoff_speed:
                current_descent_speed += self.takeoff_acceleration
            
            self.current_z_setpoint -= current_descent_speed
            if self.current_z_setpoint < target_height:
                self.current_z_setpoint = target_height
            
            pos_cmd = TrajectorySetpoint()
            pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            pos_cmd.position = [self.vehicle_local_position[1], self.vehicle_local_position[0], -self.current_z_setpoint]
            pos_cmd.yaw = self.locked_ned_yaw
            self.publisher_trajectory.publish(pos_cmd)
            time.sleep(self.dt)
            
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.node.get_logger().info("Final approach. Landing command sent.")

    def get_laser_scan(self, msg):
        self.laser_done_cnt += 1
        ranges = np.array(msg.ranges)
        # 1. Clip and handle NaN/Inf — max range is 12m (matching training env)
        max_range = 12.0
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=max_range)
        ranges[ranges == 0] = max_range
        ranges = np.clip(ranges, 0.0, max_range)
        
        # 2. Map 270° FOV into 360° virtual scan
        # Model expects 128 bins over 360° (2.8125° per bin)
        # SITL LiDAR has 270° FOV -> covers exactly 96 bins (270 / 2.8125)
        virtual_scan_360 = np.ones(128) * max_range # Default to max range (blind spot)
        
        # Downsample real 270° data to 96 bins
        num_points = len(ranges)
        if num_points > 0:
            # Interpolate the 1080 (or whatever) points into 96 bins
            resampled_270 = np.interp(np.linspace(0, num_points, 96), np.arange(num_points), ranges)
            
            # Place resampled data into the virtual scan
            # Center of 96 bins is index 48. We want this at index 64 (Forward)
            # So the 96 bins go from index 64-48=16 to 64+48=112
            virtual_scan_360[16:112] = resampled_270

        # 3. Standard Normalization (Matches Training)
        robot_radius = 0.5
        min_ranges = virtual_scan_360
        clearances = np.clip(min_ranges - robot_radius, 0.0, max_range)
        self.extracted_row = clearances / max_range

    def vehicle_odometry_callback(self, msg):
        # 1. Update position (ENU: msg.position[0]=North, msg.position[1]=East, msg.position[2]=Down)
        # NED msg: [North, East, Down]
        self.vehicle_local_position[0] = msg.position[1] # East
        self.vehicle_local_position[1] = msg.position[0] # North
        self.vehicle_local_position[2] = -msg.position[2] # Up
        
        self.pose.position.x = self.vehicle_local_position[0]
        self.pose.position.y = self.vehicle_local_position[1]
        self.pose.position.z = self.vehicle_local_position[2]
        
        # 2. Update velocity (ENU)
        self.vehicle_local_velocity[0] = msg.velocity[1] # vy
        self.vehicle_local_velocity[1] = msg.velocity[0] # vx
        self.vehicle_local_velocity[2] = -msg.velocity[2] # vz

        # 3. Calculate yaw from quaternion (NED: [w, x, y, z])
        q = msg.q
        self.raw_ned_yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        
        # Convert NED yaw (North-Clockwise) to ENU yaw (East-CounterClockwise)
        # matches training and train_env_disp_mem.py
        self.trueYaw = (math.pi / 2.0) - self.raw_ned_yaw
        # Simple wrap
        while self.trueYaw > math.pi: self.trueYaw -= 2.0 * math.pi
        while self.trueYaw < -math.pi: self.trueYaw += 2.0 * math.pi
        
        self.pos_received = True
        
        # Calculate distance and heading
        self.distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))
        self.goal_heading = math.atan2((self.goal[1] - self.pose.position.y),self.goal[0]-self.pose.position.x)
        
        if self.distance < 0.3:
            self.done = True
            self.goal_reached = True

    def node_spin(self):
        try:
            self.executor.spin()
        except rclpy.executors.ExternalShutdownException:
            pass

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def reset(self,seed=None,options=None):
        if options is not None and "goal_pos" in options:
             self.goal = options["goal_pos"]
        
        # Wait for data
        wait_start = time.time()
        while (math.isnan(self.raw_ned_yaw) or not self.pos_received) and (time.time() - wait_start) < 10.0:
            time.sleep(0.1)
        
        self.start_east = self.pose.position.x
        self.start_north = self.pose.position.y
        self.start_yaw = self.trueYaw
        self.locked_ned_yaw = self.raw_ned_yaw
        
        # Handle Offboard/Arm
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        
        pos_cmd = TrajectorySetpoint()
        pos_cmd.yaw = self.locked_ned_yaw
        
        # Smooth Takeoff
        target_altitude = 1.5
        self.current_z_setpoint = 0.0
        current_speed = 0.001 
        
        self.node.get_logger().info(f"Starting smooth takeoff to {target_altitude}m...")
        
        while self.current_z_setpoint < target_altitude:
            if current_speed < self.takeoff_speed:
                current_speed += self.takeoff_acceleration
            
            self.current_z_setpoint += current_speed
            if self.current_z_setpoint > target_altitude:
                self.current_z_setpoint = target_altitude
                
            pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            pos_cmd.position = [self.start_north, self.start_east, -self.current_z_setpoint]
            self.publisher_trajectory.publish(pos_cmd)
            
            if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            
            time.sleep(self.dt)

        self.node.get_logger().info("Altitude reached. Holding for stability (2s)...")
        hold_start = time.time()
        while (time.time() - hold_start) < 2.0:
            pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            self.publisher_trajectory.publish(pos_cmd)
            time.sleep(self.dt)
              
        # Capture the actual stabilized position to base the goal on
        self.start_east = self.pose.position.x
        self.start_north = self.pose.position.y
        self.start_yaw = self.trueYaw
        self.locked_ned_yaw = self.raw_ned_yaw # Re-lock for consistency
        
        self.target_pos = np.array([self.start_east, self.start_north, 1.5]) 
        self.last_action = np.zeros(2)
        
        # Observations
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        heading_norm = heading_diff / math.pi
        
        dx_global = self.goal[0] - self.pose.position.x
        dy_global = self.goal[1] - self.pose.position.y
        dev_x_local = dx_global * math.cos(self.trueYaw) + dy_global * math.sin(self.trueYaw)
        dev_y_local = -dx_global * math.sin(self.trueYaw) + dy_global * math.cos(self.trueYaw)

        self.goal_data = np.array([self.last_action[0], self.last_action[1], self.distance / 11.0, heading_norm, dev_x_local / 8.0, dev_y_local / 8.0], dtype=np.float64)
        state =  np.append(self.extracted_row, self.goal_data)
        
        self.done = False
        self.goal_reached = False
        return (state, {})

    def step(self, action):
        reward = 0.0
        truncated = False

        # Action scale matches training env (action[0]*0.05, action[1]*0.05)
        move_fwd = float(action[0]) * 0.05
        move_lat = float(action[1]) * 0.05
        target_up = 1.5 
        
        # Standard ENU body->world rotation (matching train_env_disp_mem.py)
        # current_yaw is now ENU East-CCW
        current_yaw = self.trueYaw
        delta_east  = move_fwd * math.cos(current_yaw) - move_lat * math.sin(current_yaw)
        delta_north = move_fwd * math.sin(current_yaw) + move_lat * math.cos(current_yaw)

        target_east = self.target_pos[0] + delta_east
        target_north = self.target_pos[1] + delta_north

        vel_cmd = TrajectorySetpoint()
        vel_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        vel_cmd.position = [target_north, target_east, -target_up]
        vel_cmd.yaw = self.locked_ned_yaw
        self.publisher_trajectory.publish(vel_cmd)
        
        self.target_pos = np.array([target_east, target_north, target_up])
        self.last_action = action
        time.sleep(0.05)
        
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        heading_norm = heading_diff / math.pi
        
        dx_global = self.goal[0] - self.pose.position.x
        dy_global = self.goal[1] - self.pose.position.y
        dev_x_local = dx_global * math.cos(self.trueYaw) + dy_global * math.sin(self.trueYaw)
        dev_y_local = -dx_global * math.sin(self.trueYaw) + dy_global * math.cos(self.trueYaw)

        self.goal_data = np.array([self.last_action[0], self.last_action[1], self.distance / 11.0, heading_norm, dev_x_local / 8.0, dev_y_local / 8.0], dtype=np.float64)
        state =  np.append(self.extracted_row, self.goal_data)

        # Update Visualizer
        if not self.viz_queue.full():
            self.viz_queue.put((self.extracted_row, self.distance, heading_diff, action, dev_x_local, dev_y_local))

        if not self.done:
            reward = (self.prev_distance - self.distance)
            self.prev_distance = self.distance
        else:
            reward = 100.0 if self.goal_reached else -100.0
        
        return state, reward, self.done, truncated, {"reached":self.goal_reached}
  
    def close(self):
        if self._is_closed: return
        self._is_closed = True
        self.node.destroy_node()
        rclpy.shutdown()

    def render(self): pass
