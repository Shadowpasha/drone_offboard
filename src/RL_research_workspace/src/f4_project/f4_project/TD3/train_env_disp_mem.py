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
import cv2
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus
from rclpy.clock import Clock
import subprocess
import multiprocessing
try:
    from .visualizer import start_visualizer
except ImportError:
    from visualizer import start_visualizer

class DroneEnv(gym.Env):
    def __init__(self, sim=True):
        self.sim = sim
        
        # QoS profile for LiDAR
        qos_profile_laser = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
       
        rclpy.init()
        # Node name based on sim mode
        node_name = "training" if self.sim else "real_drone_env"
        self.node = rclpy.create_node(node_name)
        self.goal_range = 3
        
        if self.sim:
            self.obstacle_range = 4.5
            self.num_obstacles = 4
        
        # QoS profiles for Publishers
        qos_profile_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher_offboard_mode = self.node.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.node.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_vehicle_command = self.node.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', qos_profile_pub)
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
            "fmu/out/vehicle_odometry",
            self.vehicle_odometry_callback,
            qos_profile_sub,
        )
        self.status_sub_px4 = self.node.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile_sub,
        )
        self.status_sub_px4_v1 = self.node.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            qos_profile_sub,
        )

        # Lidar topic selection based on sim mode
        if self.sim:
            lidar_topic = "/world/forest/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"
        else:
            lidar_topic = "/scan"
            
        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            lidar_topic,
            self.get_laser_scan,
            qos_profile_laser
        )
        self.node.get_logger().info(f"LiDAR subscription created on topic: {lidar_topic}")

        self.pose = Pose()
        self.vel = Twist()
        
        if self.sim:
            self.first_reset = True
            self.goal = [random.uniform(-3.5, 3.5), random.uniform(-4.0, 4.0)]
            self.max_steps = 1000
        else:
            self.goal = [2.0, 0.0] # Default goal 2m ahead
         
        self.prev_distance = 0.0
        self.distance = 0.0
        self.goal_reached = False
        self.done = False

        self.pitch = 0.0
        self.roll = 0.0
        self.trueYaw = 0.0
        self.raw_ned_yaw = float('nan')
        self.goal_heading = 0.0
        self.pos_received = False

        self.action_space = spaces.Box(np.array([-1,-1]), np.array([1,1]), (2,), dtype=np.float64) 
        # Observation: 60 laser rays + 10 goal/state data = 70 (matches MuJoCo deployment env)
        self.observation_space = spaces.Box(-8.0, 8.0, shape=(70,), dtype=np.float64)
        self.laser_done_cnt = 0
        self.ep_time = 0
        
        if self.sim:
            self.tree_locations = []
            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            self.tree_locations.append(pose)
            
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        self.et = threading.Thread(target=self.node_spin)
        self.et.daemon = True
        self.et.start()
        
        self._is_closed = False
        # 60 laser rays matching MuJoCo env (normalized [0,1] by ray_range=12m)
        self.extracted_row = np.ones(60) * 1.0 # Initialize with max normalized distance
        self.raw_ranges = np.ones(1080) * 12.0 # Initialize high-res raw ranges

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.offboard_setpoint_counter = 0
        self.pre_arm_active = False  # Set by reset() to trigger the arm sequence via cmdloop

        if self.sim:
            self.spawned_obstacles = []
            self.goal_marker_spawned = False

        # Maintain offboard mode
        self.timer = self.node.create_timer(0.04, self.cmdloop_callback)
        
        self.target_pos = np.zeros(3) # ENU target setpoint [East, North, Up]
        self.lock = threading.Lock()
        self.last_action = np.zeros(2) 
        
        self.takeoff_speed = 0.1
        self.takeoff_acceleration = 0.01
        self.current_z_setpoint = 0.0
        self.dt = 0.04 
        
        if not self.sim:
            self.play_area_limit = 5.0 # ±5m from start (10x10m area)

        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0
        self.origin_z_fixed = None
        self.origin_set = False

        # Visualizer Setup
        self.viz_queue = multiprocessing.Queue(maxsize=1)
        self.viz_proc = multiprocessing.Process(target=start_visualizer, args=(self.viz_queue,))
        self.viz_proc.daemon = True
        self.viz_proc.start()

    def cmdloop_callback(self):
        # Always publish OffboardControlMode (required every <500ms or PX4 drops offboard)
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

        # Always co-publish a TrajectorySetpoint — PX4 requires BOTH to enter offboard mode.
        # Before armed/offboard: hold at current NED position.
        # After armed/offboard: track self.target_pos (set by reset() and step()).
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)

        if self.pre_arm_active:
            # Pre-arm phase (triggered by reset()): stream current-position setpoints,
            # then send arm + offboard commands after 10 iterations (canonical pattern
            # from https://github.com/Jaeyoung-Lim/px4-offboard/blob/master/px4_offboard/offboard_control.py)
            traj_msg.position = [
                float(self.pose.position.y),   # NED North
                float(self.pose.position.x),   # NED East
                float(-self.pose.position.z),  # NED Down
            ]
            if hasattr(self, 'locked_ned_yaw') and not math.isnan(self.raw_ned_yaw):
                traj_msg.yaw = self.raw_ned_yaw
            elif not math.isnan(self.raw_ned_yaw):
                traj_msg.yaw = self.raw_ned_yaw
            self.publisher_trajectory.publish(traj_msg)

            if self.offboard_setpoint_counter == 10:
                # Send arm + offboard mode commands exactly once after 10 setpoints streamed
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            if self.offboard_setpoint_counter < 200:  # Keep retrying up to ~10s
                self.offboard_setpoint_counter += 1
                # Re-send arm commands every 50 ticks (~2.5s) in case of lost packet
                if self.offboard_setpoint_counter > 10 and self.offboard_setpoint_counter % 50 == 0:
                    if not (self.arming_state == VehicleStatus.ARMING_STATE_ARMED and
                            self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        else:
            # Normal operation: track target_pos (set by reset() after takeoff or step())
            if self.origin_set:
                with self.lock:
                    target_pos_copy = self.target_pos.copy()
                traj_msg.position = [
                    float(target_pos_copy[1]),   # NED North
                    float(target_pos_copy[0]),   # NED East
                    float(-target_pos_copy[2]),  # NED Down
                ]
                # Set unused fields to NaN so PX4 ignores them (pure position control)
                traj_msg.velocity = [float('nan'), float('nan'), float('nan')]
                traj_msg.acceleration = [float('nan'), float('nan'), float('nan')]
                if hasattr(self, 'locked_ned_yaw'):
                    traj_msg.yaw = self.locked_ned_yaw
                self.publisher_trajectory.publish(traj_msg)

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
        self.node.get_logger().info("Emergency / Mission complete. Sending land command.")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.node.get_logger().info("Landing command sent.")

    def get_laser_scan(self, msg):
        self.laser_done_cnt += 1
        ranges = np.array(msg.ranges)
        
        # 1. Clip and handle NaN/Inf — max range is 12m (matching training env)
        max_range = 12.0
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=max_range)
        ranges[ranges == 0] = max_range
        
        # Save the high-res raw ranges AFTER cleaning up invalid noise, but BEFORE downsampling
        self.raw_ranges = ranges.copy()
        
        ranges = np.clip(ranges, 0.0, max_range)
        
        # 2. Map 270° FOV into 360° virtual scan (60 bins, matching MuJoCo deployment env)
        # Model expects 60 bins over 360° (6° per bin)
        # LiDAR has 270° FOV -> covers 45 of the 60 bins (270/360 * 60 = 45)
        virtual_scan_360 = np.ones(60) * max_range # Default to max range (blind spot = no obstacle)
        
        # Downsample real 270° data to 45 bins
        num_points = len(ranges)
        if num_points > 0:
            # Interpolate the full-resolution scan into 45 bins covering 270°
            resampled_270 = np.interp(np.linspace(0, num_points, 45), np.arange(num_points), ranges)
            
            # Place resampled data into the virtual 360 scan.
            # Center of 45 bins is index 22. We want Forward (0°) at index 30 (half of 60).
            # So the 45 bins go from index 30-22=8 to 8+45=53
            virtual_scan_360[8:53] = resampled_270

        # 3. Normalize by ray_range (dist / self.ray_range)
        self.extracted_row = np.clip(virtual_scan_360 / max_range, 0.0, 1.0)

    def vehicle_odometry_callback(self, msg):
        self.vehicle_local_position[0] = msg.position[1] # East
        self.vehicle_local_position[1] = msg.position[0] # North
        self.vehicle_local_position[2] = -msg.position[2] # Up
        
        self.pose.position.x = self.vehicle_local_position[0]
        self.pose.position.y = self.vehicle_local_position[1]
        self.pose.position.z = self.vehicle_local_position[2]
        
        self.vehicle_local_velocity[0] = msg.velocity[1] # vy
        self.vehicle_local_velocity[1] = msg.velocity[0] # vx
        self.vehicle_local_velocity[2] = -msg.velocity[2] # vz

        q = msg.q
        self.raw_ned_yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        
        self.trueYaw = (math.pi / 2.0) - self.raw_ned_yaw
        self.trueYaw = (self.trueYaw + math.pi) % (2 * math.pi) - math.pi
        
        self.pos_received = True

        self.distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))
        self.goal_heading = math.atan2((self.goal[1] - self.pose.position.y),self.goal[0]-self.pose.position.x)
        
        if self.distance < 0.5:
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
        if self.sim:
            if options is not None and "goal_pos" in options:
                 goal_pos = options["goal_pos"]
            else:
                 goal_pos = None
        else:
            if options is not None and "goal_pos" in options and options["goal_pos"] is not None:
                 goal_pos = options["goal_pos"]
            else:
                 goal_pos = None

        # Wait for data
        wait_start = time.time()
        while (math.isnan(self.raw_ned_yaw) or not self.pos_received) and (time.time() - wait_start) < 10.0:
            time.sleep(0.1)
        
        # Spin-wait until we have valid position data from PX4
        while not self.pos_received and (time.time() - wait_start) < 10.0:
            time.sleep(0.1)
        
        # --- Trigger the canonical pre-arm sequence via cmdloop_callback ---
        # Following https://github.com/Jaeyoung-Lim/px4-offboard/blob/master/px4_offboard/offboard_control.py:
        # Reset the counter so cmdloop streams 10 setpoints at current pos, then arms.
        self.offboard_setpoint_counter = 0
        self.origin_set = False  # Prevent cmdloop from tracking target_pos until arm confirmed
        self.pre_arm_active = True
        self.node.get_logger().info("Pre-arm sequence started via cmdloop (streaming setpoints + arm)...")

        # Wait for arm + offboard confirmation (cmdloop_callback handles all publishing)
        arm_wait_start = time.time()
        while (time.time() - arm_wait_start) < 15.0:
            if (self.arming_state == VehicleStatus.ARMING_STATE_ARMED and
                    self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                break
            time.sleep(0.1)

        self.pre_arm_active = False  # Stop pre-arm mode in cmdloop

        if not (self.arming_state == VehicleStatus.ARMING_STATE_ARMED and
                self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            self.node.get_logger().error("FAILED to arm/enter offboard after 15s. Aborting reset.")
        else:
            self.node.get_logger().info("Armed and in offboard mode.")

        # --- Capture origin now that we are confirmed armed + in offboard ---
        self.start_east = self.pose.position.x
        self.start_north = self.pose.position.y
        self.start_yaw = self.trueYaw
        self.locked_ned_yaw = self.raw_ned_yaw

        self.origin_x = self.start_east
        self.origin_y = self.start_north
        if self.sim:
            if self.origin_z_fixed is None:
                self.origin_z_fixed = self.pose.position.z
            self.origin_z = self.origin_z_fixed
        else:
            self.origin_z = self.pose.position.z
        # NOTE: origin_set is NOT set True here yet — it will be set after target_pos
        # is primed with the takeoff altitude, to avoid cmdloop firing with target_pos=zeros.
        self.node.get_logger().info(f"Origin captured: [{self.origin_x:.2f}, {self.origin_y:.2f}, {self.origin_z:.2f}]")

        if self.sim:
            self.goal_reached = False
            if len(self.tree_locations) > 0:
                self.tree_locations[0].position.x = self.start_east
                self.tree_locations[0].position.y = self.start_north
            self.randomize_trees(goal_pos=goal_pos)
        else:
            if goal_pos is not None:
                local_fwd = float(goal_pos[0])
                local_left = float(goal_pos[1])
            else:
                local_fwd = 2.0
                local_left = 0.0
            offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
            offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
            self.goal = [self.start_east + offset_east, self.start_north + offset_north]
            self.node.get_logger().info(f"Goal set to: [{self.goal[0]:.2f}, {self.goal[1]:.2f}] (Relative offset: [{local_fwd:.2f}, {local_left:.2f}])")

        # --- Takeoff: set target_pos to 1m above origin; cmdloop will track it ---
        target_altitude = 1.7
        # NED Z = -(ENU Up origin_z) - altitude (more negative = higher in NED)
        ned_z_ground = -self.origin_z
        ned_z_target = ned_z_ground - target_altitude
        self.node.get_logger().info(f"Taking off to {target_altitude}m (NED Z target={ned_z_target:.2f}, origin_z={self.origin_z:.2f})...")
        # IMPORTANT: set target_pos BEFORE origin_set=True to avoid cmdloop
        # firing in the window between origin_set=True and target_pos being assigned,
        # which would send a ground-level setpoint ([0,0,0] NED) to PX4.
        with self.lock:
            self.target_pos = np.array([self.start_east, self.start_north, self.origin_z + target_altitude])
        self.origin_set = True  # NOW cmdloop will begin tracking target_pos safely
        self.last_action = np.zeros(2)

        # Wait for the drone to reach takeoff altitude (cmdloop handles setpoint publishing)
        takeoff_wait_start = time.time()
        while (time.time() - takeoff_wait_start) < 10.0:
            current_alt = self.pose.position.z
            if abs(current_alt - (self.origin_z + target_altitude)) < 0.3:
                self.node.get_logger().info(f"Takeoff altitude reached: {current_alt:.2f}m")
                break
            time.sleep(0.1)
        else:
            self.node.get_logger().warn(f"Takeoff altitude not confirmed within 10s (current z={self.pose.position.z:.2f}m). Continuing anyway.")
        
        # Distance and goal heading calculation
        self.distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))
        self.goal_heading = math.atan2((self.goal[1] - self.pose.position.y),self.goal[0]-self.pose.position.x)
        self.prev_distance = self.distance
        
        # Observations
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        dx_global = self.goal[0] - self.pose.position.x
        dy_global = self.goal[1] - self.pose.position.y
        dev_x_local = dx_global * math.cos(self.trueYaw) + dy_global * math.sin(self.trueYaw)
        dev_y_local = -dx_global * math.sin(self.trueYaw) + dy_global * math.cos(self.trueYaw)

        # Get velocity in body frame (ENU)
        vx_body = self.vehicle_local_velocity[1] * math.cos(self.trueYaw) + self.vehicle_local_velocity[0] * math.sin(self.trueYaw)
        vy_body = -self.vehicle_local_velocity[1] * math.sin(self.trueYaw) + self.vehicle_local_velocity[0] * math.cos(self.trueYaw)
        
        # 10-element goal data: matches MuJoCo _get_obs exactly, relative coordinates
        # [action[0], action[1], distance, heading_error, vx_body, vy_body, pos_x, pos_y, roll, pitch]
        self.goal_data = np.array([
            self.last_action[0], 
            self.last_action[1], 
            self.distance,         # raw distance
            heading_diff,          # heading error in radians [-pi, pi]
            vx_body,               # body-frame forward velocity
            vy_body,               # body-frame lateral velocity
            self.pose.position.x - self.start_east,  # relative position X
            self.pose.position.y - self.start_north,  # relative position Y
            self.roll,             # roll angle
            self.pitch,            # pitch angle
        ], dtype=np.float64)
        # State dim = 60 (laser) + 10 (goal/state) = 70 (matches MuJoCo deployment env)
        state =  np.append(self.extracted_row, self.goal_data)
        
        self.ep_time = 0
        self.done = False
        self.goal_reached = False
        
        if self.sim:
            # Update Visualizer in simulation
            self.viz_queue.put((self.extracted_row, self.distance, heading_diff, dev_x_local, dev_y_local, self.last_action, vx_body, vy_body, self.pose.position.x, self.pose.position.y))
        else:
            if not self.viz_queue.full():
                self.viz_queue.put((self.extracted_row, self.distance, heading_diff, dev_x_local, dev_y_local, self.last_action, vx_body, vy_body, self.pose.position.x, self.pose.position.y))

        return (state, {})

    def step(self, action):
        reward = 0.0
        truncated = False

        # Action scale matches train_env_disp_mem.py (action[0]*0.00975, action[1]*0.00625)
        move_fwd = float(action[0]) * 0.00975
        move_lat = float(action[1]) * 0.00625
        target_up = 1.7 
        
        # Standard ENU body->world rotation
        current_yaw = self.trueYaw
        delta_east  = move_fwd * math.cos(current_yaw) - move_lat * math.sin(current_yaw)
        delta_north = move_fwd * math.sin(current_yaw) + move_lat * math.cos(current_yaw)

        with self.lock:
            target_east = self.target_pos[0] + delta_east
            target_north = self.target_pos[1] + delta_north

            if not self.sim:
                # Limit movement to 10x10 area centered at takeoff in real flight
                target_east = np.clip(target_east, self.start_east - self.play_area_limit, self.start_east + self.play_area_limit)
                target_north = np.clip(target_north, self.start_north - self.play_area_limit, self.start_north + self.play_area_limit)

            self.target_pos = np.array([target_east, target_north, self.origin_z + target_up])

        self.last_action = action
        
        time.sleep(0.04)
        
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        dx_global = self.goal[0] - self.pose.position.x
        dy_global = self.goal[1] - self.pose.position.y
        dev_x_local = dx_global * math.cos(self.trueYaw) + dy_global * math.sin(self.trueYaw)
        dev_y_local = -dx_global * math.sin(self.trueYaw) + dy_global * math.cos(self.trueYaw)

        # Get velocity in body frame (ENU)
        vx_body = self.vehicle_local_velocity[1] * math.cos(self.trueYaw) + self.vehicle_local_velocity[0] * math.sin(self.trueYaw)
        vy_body = -self.vehicle_local_velocity[1] * math.sin(self.trueYaw) + self.vehicle_local_velocity[0] * math.cos(self.trueYaw)
        
        # 10-element goal data: matches MuJoCo _get_obs exactly, relative coordinates
        # [action[0], action[1], distance, heading_error, vx_body, vy_body, pos_x, pos_y, roll, pitch]
        self.goal_data = np.array([
            self.last_action[0], 
            self.last_action[1], 
            self.distance,         # raw distance
            heading_diff,          # heading error in radians [-pi, pi]
            vx_body,               # body-frame forward velocity
            vy_body,               # body-frame lateral velocity
            self.pose.position.x - self.start_east,  # relative position X
            self.pose.position.y - self.start_north,  # relative position Y
            self.roll,             # roll angle
            self.pitch,            # pitch angle
        ], dtype=np.float64)
        # State dim = 60 (laser) + 10 (goal/state) = 70 (matches MuJoCo deployment env)
        state =  np.append(self.extracted_row, self.goal_data)

        # Update Visualizer with kinematics
        if self.sim:
            self.viz_queue.put((self.extracted_row, self.distance, heading_diff, dev_x_local, dev_y_local, action, vx_body, vy_body, self.pose.position.x, self.pose.position.y))
        else:
            if not self.viz_queue.full():
                self.viz_queue.put((self.extracted_row, self.distance, heading_diff, dev_x_local, dev_y_local, action, vx_body, vy_body, self.pose.position.x, self.pose.position.y))

        if self.sim:
            if self.ep_time >= self.max_steps:
                self.done = True
                truncated = True
            self.ep_time += 1

        if not self.done:
            # Progress toward goal (matches training env)
            reward = 3.0 * (self.prev_distance - self.distance)
            self.prev_distance = self.distance
        else:
            if self.goal_reached:
                reward = 300.0
            else:
                reward = -50.0
        
        return state, reward, self.done, truncated, {"reached":self.goal_reached}

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        
        self.node.get_logger().info("Closing Environment and ROS2 Node...")
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        
        if hasattr(self, 'et'):
            self.et.join(timeout=1.0)
            
        if hasattr(self, 'node'):
            self.node.destroy_node()
            
        try:
            rclpy.shutdown()
        except Exception:
            pass

        if hasattr(self, 'viz_proc') and self.viz_proc.is_alive():
            self.viz_proc.terminate()
            self.viz_proc.join(timeout=1.0)

    def render(self):
        pass

    def check_pos(self,x,y):
        pos_ok = True
        for model in self.tree_locations:
            if( model.position.x + 1.8 > x > model.position.x - 1.8 and model.position.y + 1.8 > y > model.position.y - 1.8):
                pos_ok = False
        return pos_ok
    
    def check_pos_goal(self,x,y):
        pos_ok = True
        for model in self.tree_locations:
            if( model.position.x + 0.7 > x > model.position.x - 0.7 and model.position.y + 0.7 > y > model.position.y - 0.7):
                pos_ok = False
        return pos_ok

    def spawn_ring(self, name, x, y, yaw):
        if not self.sim: return
        sdf_path = "/home/anas/drone_sitl_ws/src/RL_research_workspace/src/f4_project/urdf/ring_obstacle.sdf"
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', sdf_path,
            '-name', name,
            '-x', str(x),
            '-y', str(y),
            '-z', '0.0',
            '-Y', str(yaw)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                self.spawned_obstacles.append(name)
                print(f"DEBUG: Spawned {name} via ros_gz_sim at ({x:.2f}, {y:.2f})")
            else:
                print(f"ERROR: Failed to spawn {name}. \nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        except Exception as e:
            print(f"ERROR running ros_gz_sim for {name}: {e}")

    def spawn_goal_marker(self, x, y):
        if not self.sim: return
        name = "goal_disk"
        sdf_content = f"""
        <?xml version="1.0" ?>
        <sdf version="1.6">
          <model name="{name}">
            <static>true</static>
            <link name="link">
              <pose>{x} {y} 0.05 0 0 0</pose>
              <visual name="visual">
                <geometry>
                  <cylinder><radius>0.4</radius><length>0.1</length></cylinder>
                </geometry>
                <material>
                  <ambient>0.1 0.8 0.1 1</ambient>
                  <diffuse>0.1 0.8 0.1 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>""".replace('\n', '').replace('"', '\\"')
        cmd = [
            'gz', 'service', '-s', '/world/forest/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '300',
            '--req', f'sdf: "{sdf_content}", name: "{name}"'
        ]
        try:
            if self.goal_marker_spawned:
                subprocess.run(['gz', 'service', '-s', '/world/forest/remove', '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean', '--timeout', '50', '--req', f'name: "{name}"'], capture_output=True, check=False)
            subprocess.run(cmd, capture_output=True, check=False)
            self.goal_marker_spawned = True
            print(f"DEBUG: Goal disk spawned at ({x:.2f}, {y:.2f})")
        except:
            pass

    def clear_trees(self):
        if not self.sim: return
        for name in self.spawned_obstacles:
             try:
                 subprocess.run(['gz', 'service', '-s', '/world/forest/remove', '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean', '--timeout', '100', '--req', f'name: "{name}"'], capture_output=False)
             except:
                 pass
        self.spawned_obstacles = []

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.goal_marker_pub.publish(marker)

    def randomize_trees(self, goal_pos=None):
        if not self.sim: return
        print("randomizing obstacles along the path")
        self.clear_trees()
        
        # 1. Determine Goal Position first
        if goal_pos is not None:
             local_fwd = float(goal_pos[0])
             local_left = float(goal_pos[1])
             
             offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
             offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
             
             self.goal = [self.start_east + offset_east, self.start_north + offset_north]
        else:
            goal_ok = False
            while not goal_ok:
                local_fwd = random.uniform(3.0, 5.0) 
                local_left = random.uniform(-2.0, 2.0)
                
                offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
                offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
                
                self.goal = [self.start_east + offset_east, self.start_north + offset_north]
                goal_ok = self.check_pos_goal(self.goal[0],self.goal[1])

        # 2. Spawn 5 random ring obstacles
        num_obstacles = 5
        spawned_count = 0
        attempts = 0
        while spawned_count < num_obstacles and attempts < 50:
             attempts += 1
             local_fwd = random.uniform(-4.0, 4.0)
             local_left = random.uniform(-4.0, 4.0)
             
             offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
             offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
             
             ox = self.start_east + offset_east
             oy = self.start_north + offset_north
             yaw = random.uniform(-math.pi, math.pi)
             
             dist_from_start = math.sqrt((ox - self.start_east)**2 + (oy - self.start_north)**2)
             dist_from_goal = math.sqrt((ox - self.goal[0])**2 + (oy - self.goal[1])**2)
             
             if dist_from_start > 1.0 and dist_from_goal > 1.0:
                  obs_name = f"ring_{spawned_count}"
                  self.spawn_ring(obs_name, ox, oy, yaw)
                  spawned_count += 1
                  
        time.sleep(0.1)
        self.publish_goal_marker(self.goal[0], self.goal[1])
        self.spawn_goal_marker(self.goal[0], self.goal[1])
        self.prev_distance = math.sqrt(math.pow(self.goal[0] - self.start_east, 2) + math.pow(self.goal[1] - self.start_north, 2))

class DroneGazeboEnv(DroneEnv):
    def __init__(self):
        super().__init__(sim=True)

class RealDroneEnv(DroneEnv):
    def __init__(self):
        super().__init__(sim=False)
