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
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition, OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus
from rclpy.clock import Clock

class DroneGazeboEnv(gym.Env):
    def __init__(self):

        qos_profile_laser = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.SYSTEM_DEFAULT,
            history=DurabilityPolicy.SYSTEM_DEFAULT,
            depth=10
        )
       
        rclpy.init()
        self.node = rclpy.create_node("training")
        self.goal_range = 3
        self.obstacle_range = 4.5
        self.num_obstacles = 5
        
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

        self.attitude_sub_px4 = self.node.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            qos_profile_sub,
        )
        self.local_position_sub_px4 = self.node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )

        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            "/world/forest/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan",
            self.get_laser_scan,
            qos_profile_laser
        )

        self.pose = Pose()
        self.vel = Twist()
        self.first_reset  = True
        self.goal = [random.uniform(-3.5, 3.5),random.uniform(-4.0, 4.0)]
         
        self.prev_distance = math.sqrt(math.pow(self.goal[0],2) + math.pow(self.goal[1],2))
        self.prev_closest_laser = 5.0
        self.distance = self.prev_distance
        self.goal_reached = False
        self.overshoot = False
        self.penalty = 0.0

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
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(70,), dtype= np.float64)
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
        
        self.close = False
        self.laser_ranges = np.zeros(10)
        self.laser_ranges_top = np.zeros(10)
        self.laser_ranges_bottom = np.zeros(10)
        self.laser_ranges_360 = np.zeros(20)
        self.goal_data = np.zeros(6)
        self.extracted_row = np.ones(64) * 1.0 # Initialize with safe distance

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.last_local_pos_update = 0.0

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.offboard_setpoint_counter = 0

        # Create a timer to publish control mode and maintain offboard
        self.timer = self.node.create_timer(0.05, self.cmdloop_callback)


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
        # Sends NAV_LAND command (21)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        print("Landing command sent.")

    def get_laser_scan(self, msg):
        ranges = np.array(msg.ranges)
        # Handle Inf and Nan
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        
        # Downsample to 64 points
        ranges_img = ranges.reshape(1, -1).astype(np.float32)
        resized_ranges = cv2.resize(ranges_img, (64, 1), interpolation=cv2.INTER_AREA)
        
        self.extracted_row = resized_ranges.flatten()
        
        # Normalization for Lidar rays (actual distance / 12.0) and clip to [0, 1]
        self.extracted_row = self.extracted_row / 12.0
        self.extracted_row = np.clip(self.extracted_row, 0.0, 1.0)
        
        # Check for NaN again just in case
        np.nan_to_num(self.extracted_row, copy=False, nan=1.0) # max range normalized is 1.0


    def vehicle_attitude_callback(self, msg):
        # Calculate raw MAVLink NED yaw straight from the PX4 quaternion [w, x, y, z]
        self.raw_ned_yaw = math.atan2(2.0 * (msg.q[0] * msg.q[3] + msg.q[1] * msg.q[2]), 1.0 - 2.0 * (msg.q[2]**2 + msg.q[3]**2))
        
        # NED-> ENU transformation for RL state logic
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)
        
        # Update self.pose orientation for existing logic
        self.pose.orientation.w = self.vehicle_attitude[0]
        self.pose.orientation.x = self.vehicle_attitude[1]
        self.pose.orientation.y = self.vehicle_attitude[2]
        self.pose.orientation.z = self.vehicle_attitude[3]
        
        # Update Euler angles for existing logic
        orientation_q = [self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]
        self.pitch, self.roll, self.trueYaw = euler_from_quaternion(orientation_q)


    def vehicle_local_position_callback(self, msg):
        # NED->ENU transformation for position and velocity
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz
        
        # Update self.pose position for existing logic
        self.pose.position.x = self.vehicle_local_position[0]
        self.pose.position.y = self.vehicle_local_position[1]
        self.pose.position.z = self.vehicle_local_position[2]
        
        # Update self.vel for existing logic
        self.vel.linear.x = self.vehicle_local_velocity[0]
        self.vel.linear.y = self.vehicle_local_velocity[1]
        self.vel.linear.z = self.vehicle_local_velocity[2]

        # Calculate distance and heading (reused from position_cb)
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

    def reset(self,seed=None,options=None):
        if options is not None and "goal_pos" in options:
             goal_pos = options["goal_pos"]
        else:
             goal_pos = None

        self.control_mode_position = True
        
        # Wait a tiny bit longer to ensure ROS2 subscribers have populated initial data
        time.sleep(1.0)

        # Capture the drone's current position and yaw as the local relative origin
        self.start_east = self.pose.position.x
        self.start_north = self.pose.position.y
        self.start_yaw = self.trueYaw

        # Handle Offboard/Arm/Takeoff
        # First send setpoints
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        
        # Position Command to take off straight up from current location
        pos_cmd = TrajectorySetpoint()
        pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        # NED array: [North, East, Down]
        pos_cmd.position = [self.start_north, self.start_east, -1.5] 
        
        # Maintain perfect heading by feeding PX4 its own current raw NED yaw. 
        # If no data has arrived yet, use NaN to tell PX4 to natively ignore yaw control.
        pos_cmd.yaw = self.raw_ned_yaw
        yaw_display = pos_cmd.yaw if not math.isnan(pos_cmd.yaw) else 0.0
        
        print(f"Takeoff locked to: N:{self.start_north:.2f}, E:{self.start_east:.2f}, YAW:{yaw_display:.2f} rad")
        self.publisher_trajectory.publish(pos_cmd)

        # Wait until we are close to 0,0
        start_wait = time.time()
        # Initial wait for mode switch
        time.sleep(1.0)
        
        # Loop to ensure we reach the start position
        while (time.time() - start_wait) < 8.0:
             # Check XY distance relative to start
             dist_to_start = math.sqrt((self.vehicle_local_position[0] - self.start_east)**2 + (self.vehicle_local_position[1] - self.start_north)**2)
             # Check Z distance (should be close to 1.5m -> -1.5 in NED = 1.5 in ENU)
             # self.vehicle_local_position is ENU
             dist_z = abs(self.vehicle_local_position[2] - 1.5)
             
             if dist_to_start < 0.5 and dist_z < 0.5:
                 break
             
             # Keep publishing setpoint (needed for offboard mode validity)
             pos_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
             self.publisher_trajectory.publish(pos_cmd)
             
             # Also re-arm/re-mode if disarmed unexpectedly
             if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                  self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                  self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                  
             time.sleep(0.1)
             
        print("Takeoff target reached. Stabilizing...")
        time.sleep(2.0)

        self.goal_reached = False
        self.overshoot = False
        
        # Update exclusion zone to current spot before randomizing
        if len(self.tree_locations) > 0:
            self.tree_locations[0].position.x = self.start_east
            self.tree_locations[0].position.y = self.start_north
        
        self.randomize_trees(goal_pos=goal_pos)
        
        laser_combination_level = self.extracted_row
        self.closest_laser = np.min(laser_combination_level)
        self.original_distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))

        self.prev_distance = self.distance
        
        # Construct 6-dim goal data: [Last Action X, Last Action Y, Dist/15, Heading/pi, DevX/15, DevY/15]
        # Heading diff normalized by pi (wrap to [-pi, pi] first)
        heading_diff = self.goal_heading - self.trueYaw
        while heading_diff > math.pi: heading_diff -= 2 * math.pi
        while heading_diff < -math.pi: heading_diff += 2 * math.pi
        heading_norm = heading_diff / math.pi
        
        # Standard frame transformation (Ego-centric)
        dx = self.goal[0] - self.pose.position.x
        dy = self.goal[1] - self.pose.position.y
        
        target_dev_x = (dx * math.cos(self.trueYaw) + dy * math.sin(self.trueYaw)) / 15.0
        target_dev_y = (-dx * math.sin(self.trueYaw) + dy * math.cos(self.trueYaw)) / 15.0
        
        self.goal_data = np.array([0.0, 0.0, self.distance / 15.0, heading_norm, target_dev_x, target_dev_y])

        state =  np.append(laser_combination_level,self.goal_data)
        self.previous_error = 0.0
        
        self.ep_time = 0
        self.done = False
        self.goal_reached = False
        self.contact = ContactsState()

        return (state, {})

    def step(self, action):
        reward = 0.0
        truncated = False

        # Holonomic Translational Control
        # Action[0] is forward/backward -> mapped to [-0.4, 0.4]
        move_fwd = float(action[0]) * 0.4
        # Action[1] is left/right -> mapped to [-0.4, 0.4]
        # Using rotation matrix where positive action[1] is lateral LEFT
        move_lat = float(action[1]) * 0.4
        
        # Current Position (ENU)
        current_east = self.vehicle_local_position[0]
        current_north = self.vehicle_local_position[1]
        target_up = 1.5 # Fixed altitude ENU
        
        # Current Yaw (ENU)
        current_yaw = self.trueYaw
        
        # Rotate movement into ENU frame
        # Forward axis: (cos(yaw), sin(yaw))
        # Lateral axis (Left): (-sin(yaw), cos(yaw))
        delta_east = move_fwd * math.cos(current_yaw) - move_lat * math.sin(current_yaw)
        delta_north = move_fwd * math.sin(current_yaw) + move_lat * math.cos(current_yaw)

        target_east = current_east + delta_east
        target_north = current_north + delta_north
        target_yaw = current_yaw # Maintain heading

        # Create Setpoint (NED frame required for PX4)
        vel_cmd = TrajectorySetpoint()
        vel_cmd.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        vel_cmd.position = [target_north, target_east, -target_up]
        
        # Maintain heading identically to takeoff
        vel_cmd.yaw = self.raw_ned_yaw

        self.publisher_trajectory.publish(vel_cmd)

        time.sleep(0.05)

        
        laser_combination_level = self.extracted_row
        self.closest_laser = np.min(laser_combination_level)
        
        # Construct 6-dim goal data: [Last Action X, Last Action Y, Dist/15, Heading/pi, DevX/15, DevY/15]
        heading_diff = self.goal_heading - self.trueYaw
        while heading_diff > math.pi: heading_diff -= 2 * math.pi
        while heading_diff < -math.pi: heading_diff += 2 * math.pi
        heading_norm = heading_diff / math.pi
        
        # Standard frame transformation (Ego-centric)
        dx = self.goal[0] - self.pose.position.x
        dy = self.goal[1] - self.pose.position.y
        
        target_dev_x = (dx * math.cos(self.trueYaw) + dy * math.sin(self.trueYaw)) / 15.0
        target_dev_y = (-dx * math.sin(self.trueYaw) + dy * math.cos(self.trueYaw)) / 15.0
        
        self.goal_data = np.array([action[0], action[1], self.distance / 15.0, heading_norm, target_dev_x, target_dev_y])
        state =  np.append(laser_combination_level,self.goal_data)
        # print(state)

        if(self.ep_time > 500):
            self.done = True
            truncated = True
        self.ep_time+=1

        if not self.done:
                reward = (self.prev_distance - self.distance)
                self.prev_distance = self.distance
        else:
            if(self.goal_reached):
                reward = 100.0
            else:
                reward = -100.0
        
        return state, reward, self.done, truncated, {"reached":self.goal_reached}
  

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
        # No entity spawning, so nothing to clear
        pass
    
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
        print("randomizing")   
        if goal_pos is not None:
             local_fwd = float(goal_pos[0])
             local_left = float(goal_pos[1])
             
             offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
             offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
             
             self.goal = [self.start_east + offset_east, self.start_north + offset_north]
        else:
            goal_ok = False
            while not goal_ok:
                local_fwd = random.uniform(-3.0, 3.0)
                local_left = random.uniform(-3.0, 3.0)
                
                offset_east = local_fwd * math.cos(self.start_yaw) - local_left * math.sin(self.start_yaw)
                offset_north = local_fwd * math.sin(self.start_yaw) + local_left * math.cos(self.start_yaw)
                
                self.goal = [self.start_east + offset_east, self.start_north + offset_north]
                goal_ok = self.check_pos_goal(self.goal[0],self.goal[1])

        time.sleep(0.1)

        # Publish marker
        self.publish_goal_marker(self.goal[0], self.goal[1])
        
        # Distance to goal is now distance to actual start
        self.prev_distance = math.sqrt(math.pow(self.goal[0] - self.start_east, 2) + math.pow(self.goal[1] - self.start_north, 2))
