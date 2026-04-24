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
from gazebo_msgs.msg import ContactsState # Keeping msg, removing srvs
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


class DroneGazeboEnv(gym.Env):
    def __init__(self):

        qos_profile_laser = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
       
        rclpy.init()
        self.node = rclpy.create_node("training")
        self.goal_range = 3
        self.obstacle_range = 4.5
        self.num_obstacles = 4
        
        # QoS profiles
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
        
        self.sub_contact = self.node.create_subscription(ContactsState,"/simple_drone/bumper",self.get_contact, 1)
        
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

        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            "/world/forest/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan",
            self.get_laser_scan,
            qos_profile_laser
        )
        print(f"DEBUG: LiDAR subscription created on topic: {self.lidar_sub.topic_name}")

        self.pose = Pose()
        self.vel = Twist()
        self.first_reset  = True
        self.goal = [random.uniform(-3.5, 3.5),random.uniform(-4.0, 4.0)]
         
        self.prev_distance = 0.0  # Real value set in reset()
        self.distance = 0.0
        self.goal_reached = False
        self.done = False
        self.max_steps = 1000

        self.image_counter = 0
        self.pitch = 0.0
        self.roll = 0.0
        self.trueYaw = 0.0
        self.raw_ned_yaw = float('nan')
        self.goal_heading = 0.0
        self.control_mode_position = True # Always use position control

        self.intial = True

        self.done = False

        self.contact = ContactsState()
        self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]),(2,),dtype= np.float64) 
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(134,), dtype= np.float64)
        self.laser_done_cnt = 0
        self.ep_time = time.time()
        self.previous_error = 0.0
        self.tree_locations = []
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        self.tree_locations.append(pose)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        self.et = threading.Thread(target=self.node_spin)
        self.et.start()
        
        self._is_closed_env = False
        self.laser_ranges = np.zeros(10)
        self.laser_ranges_top = np.zeros(10)
        self.laser_ranges_bottom = np.zeros(10)
        self.laser_ranges_360 = np.zeros(20)
        self.goal_data = np.zeros(6)
        self.extracted_row = np.ones(128) * 1.0 # Initialize with safe distance

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.last_local_pos_update = 0.0
        self.pos_received = False

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        # offboard_setpoint_counter initialization
        self.offboard_setpoint_counter = 0

        # List to keep track of spawned obstacle names for cleanup
        self.spawned_obstacles = []

        # Create a timer to publish control mode and maintain offboard
        self.timer = self.node.create_timer(0.05, self.cmdloop_callback)
        self.goal_marker_spawned = False
        
        self.target_pos = np.zeros(3) # ENU target setpoint [East, North, Up]
        self.last_action = np.zeros(2) # [last_action_x, last_action_y]
        self.world_size = 10.0 # Normalization constant for world-frame distances
        self._is_closed = False
        
        # Takeoff/Landing parameters
        self.takeoff_speed = 0.05 # m/s (maximum)
        self.takeoff_acceleration = 0.001 # m/s^2
        self.current_z_setpoint = 0.0
        self.dt = 0.05 # Loop rate in reset/land

        # Visualization setup
        self.viz_queue = multiprocessing.Queue()
        self.viz_process = multiprocessing.Process(target=start_visualizer, args=(self.viz_queue,), daemon=True)
        self.viz_process.start()


    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        
        # Always Position Control for safety
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
        
        # Current height (Up is positive in ENU)
        start_height = self.vehicle_local_position[2]
        self.current_z_setpoint = start_height
        current_descent_speed = 0.01 # Start slow
        
        # Ramp down to 10cm above ground
        target_height = 0.1
        while (self.current_z_setpoint > target_height) and not self._is_closed_env:
            # Accelerate descent
            if current_descent_speed < self.takeoff_speed:
                current_descent_speed += self.takeoff_acceleration
            
            self.current_z_setpoint -= current_descent_speed
            if self.current_z_setpoint < target_height:
                self.current_z_setpoint = target_height
            
            # Create Setpoint (NED: Down is negative Z)
            pos_cmd = TrajectorySetpoint()
            pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            pos_cmd.position = [self.vehicle_local_position[1], self.vehicle_local_position[0], -self.current_z_setpoint]
            pos_cmd.yaw = self.locked_ned_yaw
            self.publisher_trajectory.publish(pos_cmd)
            
            time.sleep(self.dt)
            
        # Final autonomous landing
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.node.get_logger().info("Final approach. Landing command sent.")

    def get_laser_scan(self, msg):
        if self.laser_done_cnt == 0:
            self.node.get_logger().info("First LiDAR message received!")
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
        
        # Log occasionally
        if self.laser_done_cnt % 100 == 0:
             self.node.get_logger().debug(f"LiDAR Update #{self.laser_done_cnt}: Min Norm = {np.min(self.extracted_row):.3f}")

    def spawn_ring(self, name, x, y, yaw):
        """Spawn the ring obstacle at (x, y) orientation yaw in Gazebo Sim."""
        sdf_path = "/home/anas/drone_sitl_ws/src/RL_research_workspace/src/f4_project/urdf/ring_obstacle.sdf"
        
        # Use ros_gz_sim to correctly parse the file and place it at (x, y, yaw)
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
        """Spawn a visual green disk at the goal location in Gazebo Sim."""
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
        </sdf>""".replace('\n', '').replace('"', '\\"') # Flatten for CLI
        
        cmd = [
            'gz', 'service', '-s', '/world/forest/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '300',
            '--req', f'sdf: "{sdf_content}", name: "{name}"'
        ]
        
        try:
            # Only attempt to remove if we successfully spawned it previously
            if self.goal_marker_spawned:
                subprocess.run(['gz', 'service', '-s', '/world/forest/remove', '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean', '--timeout', '50', '--req', f'name: "{name}"'], capture_output=True, check=False)
            
            subprocess.run(cmd, capture_output=True, check=False)
            self.goal_marker_spawned = True
            print(f"DEBUG: Goal disk spawned at ({x:.2f}, {y:.2f})")
        except:
            pass


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
        
        # Convert NED yaw (North-Clockwise, 0 at North) to ENU yaw (East-CounterClockwise, 0 at East)
        # Matching the robust method from real_drone_env.py
        self.trueYaw = (math.pi / 2.0) - self.raw_ned_yaw
        self.trueYaw = (self.trueYaw + math.pi) % (2 * math.pi) - math.pi
        
        self.pos_received = True

        # Calculate distance and heading
        self.distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))
        self.goal_heading = math.atan2((self.goal[1] - self.pose.position.y),self.goal[0]-self.pose.position.x)
        
        if(abs(self.distance) < 0.5):
            self.done = True
            self.goal_reached = True

        # Reset checks (roll/pitch limit)
        if ( self.pitch > 1.57 or self.pitch < -1.57):
             # Logic to handle flip, maybe just end episode or try reset
             pass 
        elif(self.roll > 1.57 or self.roll < -1.57):
             pass

    def get_image(self,msg):
        # Unused now but kept for compatibility if needed
        pass
    
    def node_spin(self):
        self.executor.spin()

    def get_contact(self,msg):
        self.contact = msg

        if(len(self.contact.states) > 0):
            if(self.contact.states[0].collision2_name != "ground_plane::link::collision"):
                self.done = True

    def velocity_cb(self,msg):
        msg = Pose()
        self.vel = msg

    def position_cb(self,msg):
        # Unused, kept as placeholder or remove
        pass

    def calculate_observation(self,data):
        ranges = list(data.ranges)
        return ranges

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def reset(self,seed=None,options=None):
        if options is not None and "goal_pos" in options:
             goal_pos = options["goal_pos"]
        else:
             goal_pos = None

        self.control_mode_position = True
        
        # Wait for ROS2 subscribers to populate valid attitude data
        time.sleep(1.0)
        
        # Spin-wait until we have valid NED yaw from PX4 (not NaN)
        wait_start = time.time()
        while math.isnan(self.raw_ned_yaw) and (time.time() - wait_start) < 5.0:
            time.sleep(0.1)
            
        # Spin-wait until we have valid position data from PX4
        while not self.pos_received and (time.time() - wait_start) < 10.0:
            time.sleep(0.1)
        
        # Capture the drone's current position and yaw as the local relative origin
        self.start_east = self.pose.position.x
        self.start_north = self.pose.position.y
        self.start_yaw = self.trueYaw
        
        # Lock the NED yaw at this exact moment — this is what PX4 will hold
        self.locked_ned_yaw = self.raw_ned_yaw
        
        print(f"DEBUG RESET: start_yaw(ENU)={math.degrees(self.start_yaw):.1f}deg, locked_ned_yaw={math.degrees(self.locked_ned_yaw):.1f}deg")
        print(f"DEBUG RESET: start_pos=({self.start_east:.2f}, {self.start_north:.2f})")

        # Handle Offboard/Arm/Takeoff
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        
        # Position Command to take off straight up from current location
        pos_cmd = TrajectorySetpoint()
        pos_cmd.yaw = self.locked_ned_yaw
        
        # Smooth Takeoff: Ramp altitude from 0 to 1.5m
        target_altitude = 1.5
        self.current_z_setpoint = 0.0
        current_speed = 0.001 # Starting slow
        
        print(f"Starting smooth takeoff to {target_altitude}m...")
        
        # Loop until the ACTUAL altitude (from odometry) reaches the target
        while self.vehicle_local_position[2] < (target_altitude - 0.1):
            # Accelerate the setpoint
            if current_speed < self.takeoff_speed:
                current_speed += self.takeoff_acceleration
            
            self.current_z_setpoint += current_speed
            if self.current_z_setpoint > target_altitude:
                self.current_z_setpoint = target_altitude
                
            pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            # NED: Down is -Z
            pos_cmd.position = [self.start_north, self.start_east, -self.current_z_setpoint]
            self.publisher_trajectory.publish(pos_cmd)
            
            if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            
            time.sleep(self.dt)

        print("Altitude reached. Holding for stability (2s)...")
        # Final stability hold
        hold_start = time.time()
        while (time.time() - hold_start) < 2.0:
            pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            self.publisher_trajectory.publish(pos_cmd)
            time.sleep(self.dt)
              
        print("Takeoff complete. Stabilizing...")
        time.sleep(1.0)

        # Update start position and yaw to the actual stabilized values
        # (Restoring this fix to ensure the goal is relative to current position)
        self.start_east = self.pose.position.x
        self.start_north = self.pose.position.y
        self.start_yaw = self.trueYaw
        self.locked_ned_yaw = self.raw_ned_yaw

        self.goal_reached = False
        self.overshoot = False
        
        # Update exclusion zone to current spot before randomizing
        if len(self.tree_locations) > 0:
            self.tree_locations[0].position.x = self.start_east
            self.tree_locations[0].position.y = self.start_north
        
        self.randomize_trees(goal_pos=goal_pos)
        
        # Track initial target setpoint (matches takeoff height)
        self.target_pos = np.array([self.start_east, self.start_north, 1.5]) # ENU
        self.last_action = np.zeros(2)
        
        laser_combination_level = self.extracted_row
        self.closest_laser = np.min(laser_combination_level)
        self.original_distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))

        self.prev_distance = self.distance
        
        # Construct 6-dim goal data: [Last Action X, Last Action Y, Dist/15, Heading/pi, DevX/15, DevY/15]
        # Heading diff normalized by pi (wrap to [-pi, pi] first)
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        heading_norm = heading_diff / math.pi
        
        # Deviation from GLOBAL GOAL
        dx_global = self.goal[0] - self.pose.position.x
        dy_global = self.goal[1] - self.pose.position.y
        
        # Transform offsets into the Drone's LOCAL Body Frame (matching training env)
        # dev_x = Forward offset, dev_y = Lateral (Left) offset
        dev_x_local = dx_global * math.cos(self.trueYaw) + dy_global * math.sin(self.trueYaw)
        dev_y_local = -dx_global * math.sin(self.trueYaw) + dy_global * math.cos(self.trueYaw)

        # Normalization Patch: 14.14 for dist (10*1.414), 10.0 for world offsets (matches 10x10 world)
        self.goal_data = np.array([
            self.last_action[0], 
            self.last_action[1], 
            self.distance / 14.14, 
            heading_norm, 
            dev_x_local / 10.0, 
            dev_y_local / 10.0
        ], dtype=np.float64)
# State dim = 128 (laser) + 6 (goal info) = 134
        state =  np.append(self.extracted_row,self.goal_data)
        self.previous_error = 0.0
        
        self.ep_time = 0
        self.done = False
        self.goal_reached = False
        self.contact = ContactsState()

        # Update Visualizer
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        self.viz_queue.put((self.extracted_row, self.distance, heading_diff, dev_x_local, dev_y_local, self.last_action))

        return (state, {})

    def step(self, action):
        reward = 0.0
        truncated = False

        # Action scale matches training env: action[0]*0.05 = forward, action[1]*0.05 = lateral
        # Training env: dx = action[0]*0.05, dy = action[1]*0.05
        move_fwd = float(action[0]) * 0.01
        move_lat = float(action[1]) * 0.01

        # Fixed altitude (ENU, positive up)
        target_up = 1.5

        # trueYaw is ENU yaw (CCW-positive from East).
        current_yaw = self.trueYaw
        # Correct ENU body->world rotation:
        #   East  = fwd*cos(yaw) - lat*sin(yaw)
        #   North = fwd*sin(yaw) + lat*cos(yaw)
        delta_east  = move_fwd * math.cos(current_yaw) - move_lat * math.sin(current_yaw)
        delta_north = move_fwd * math.sin(current_yaw) + move_lat * math.cos(current_yaw)        # Purely integrate target position (never read current pos, preventing runaway feedback)
        # self.target_pos is cleanly initialized at self.start_east/north in reset()
        target_east = self.target_pos[0] + delta_east
        target_north = self.target_pos[1] + delta_north

        # Create Setpoint (NED frame required for PX4)
        vel_cmd = TrajectorySetpoint()
        vel_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        vel_cmd.position = [target_north, target_east, -target_up]
        
        # Maintain the initial takeoff heading strictly
        vel_cmd.yaw = self.locked_ned_yaw

        self.publisher_trajectory.publish(vel_cmd)
        
        # Track commanded setpoint and action for observation
        self.target_pos = np.array([target_east, target_north, target_up])
        self.last_action = action

        time.sleep(0.05)

        
        laser_combination_level = self.extracted_row
        self.closest_laser = np.min(laser_combination_level)
        
        # Goal observation: heading and local-frame offsets (ENU body frame)
        heading_diff = self.goal_heading - self.trueYaw
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        heading_norm = heading_diff / math.pi

        # Deviation from GLOBAL GOAL projected into drone's ENU body frame
        # dev_x_local = Forward offset (positive = goal is ahead)
        # dev_y_local = Left offset (positive = goal is to the left)
        dx_global = self.goal[0] - self.pose.position.x
        dy_global = self.goal[1] - self.pose.position.y
        dev_x_local =  dx_global * math.cos(self.trueYaw) + dy_global * math.sin(self.trueYaw)
        dev_y_local = -dx_global * math.sin(self.trueYaw) + dy_global * math.cos(self.trueYaw)

        # print(f"X: {dev_x_local}, Y: {dev_y_local}")

        self.goal_data = np.array([
            self.last_action[0],
            self.last_action[1],
            self.distance / 14.14,
            heading_norm,
            dev_x_local / 10.0,
            dev_y_local / 10.0
        ], dtype=np.float64)

        
        state =  np.append(self.extracted_row,self.goal_data)
        # Update Visualizer
        self.viz_queue.put((self.extracted_row, self.distance, heading_diff, dev_x_local, dev_y_local, self.last_action))

        if(self.ep_time >= self.max_steps):
            self.done = True
            truncated = True
        self.ep_time += 1

        # print(self.distance)


        if not self.done:
                # Progress toward goal (matches training env)
                reward = 3.0 * (self.prev_distance - self.distance)
                # Simple reward: remove proximity penalties or "braveness" terms
                self.prev_distance = self.distance
        else:
            if(self.goal_reached):
                reward = 300.0
            else:
                reward = -50.0
        
        return state, reward, self.done, truncated, {"reached":self.goal_reached}
  
    def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        
        print("Closing Environment and ROS2 Node...")
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        
        if hasattr(self, 'et'):
            self.et.join(timeout=2.0)
            
        if hasattr(self, 'node'):
            self.node.destroy_node()
            
        try:
            rclpy.shutdown()
        except Exception:
            pass

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
    

    def clear_trees(self):
        # Removing entities in Gazebo Sim via ROS2 is done through 'gz service' 
        # as there isn't always a standard bridge for 'remove' unless configured.
        # Here we use the spawned_obstacles list to keep track.
        # To avoid complex ROS2-SDF-GZ service mapping for 'remove', 
        # we'll just rename and overwrite or rely on the user to clean up if many.
        # But most reliable way for SITL is to use subprocess for 'gz service'
        import subprocess
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
                # Use slightly larger range for goal if none provided
                local_fwd = random.uniform(3.0, 5.0) 
                local_left = random.uniform(-2.0, 2.0)
                
                offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
                offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
                
                self.goal = [self.start_east + offset_east, self.start_north + offset_north]
                goal_ok = self.check_pos_goal(self.goal[0],self.goal[1])

        # 2. Spawn 5 random ring obstacles in an 8x8 area around the drone
        
        num_obstacles = 5
        spawned_count = 0
        attempts = 0
        
        while spawned_count < num_obstacles and attempts < 50:
             attempts += 1
             # Sample local coordinates within an 8x8 area
             local_fwd = random.uniform(-4.0, 4.0)
             local_left = random.uniform(-4.0, 4.0)
             
             # Rotate offsets by the drone's starting yaw to align the area
             offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
             offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
             
             ox = self.start_east + offset_east
             oy = self.start_north + offset_north
             
             # Random orientation (fully random 360 degrees)
             yaw = random.uniform(-math.pi, math.pi)
             
             dist_from_start = math.sqrt((ox - self.start_east)**2 + (oy - self.start_north)**2)
             dist_from_goal = math.sqrt((ox - self.goal[0])**2 + (oy - self.goal[1])**2)
             
             # Enforce safe margins from start and goal
             if dist_from_start > 1.0 and dist_from_goal > 1.0:
                  obs_name = f"ring_{spawned_count}"
                  self.spawn_ring(obs_name, ox, oy, yaw)
                  spawned_count += 1
                  
        time.sleep(0.1)

        # Publish marker in ROS2 and Gazebo
        self.publish_goal_marker(self.goal[0], self.goal[1])
        self.spawn_goal_marker(self.goal[0], self.goal[1])
        
        # Distance to goal is now distance to actual start
        self.prev_distance = math.sqrt(math.pow(self.goal[0] - self.start_east, 2) + math.pow(self.goal[1] - self.start_north, 2))
