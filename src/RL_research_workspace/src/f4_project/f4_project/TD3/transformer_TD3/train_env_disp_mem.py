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
import math
import time
import random
import threading
from px4_msgs.msg import VehicleLocalPosition
from gazebo_msgs.srv import SetEntityState,SpawnEntity,DeleteEntity,GetEntityState
from gazebo_msgs.msg import ContactsState, EntityState
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
# from pid import PID
import os
from std_srvs.srv import Empty

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
        
        self.vel_pub = self.node.create_publisher(Twist,'/simple_drone/cmd_vel', 10)
        self.take_pub = self.node.create_publisher(Empty_msg,'/simple_drone/takeoff', 10)

        self.unpause_proxy = self.node.create_client(Empty,"/unpause_physics")
        self.pause_proxy = self.node.create_client(Empty,"/pause_physics")
        self.reset_world_proxy = self.node.create_client(Empty,"/reset_world")
        self.reset_proxy = self.node.create_client(SetEntityState,"/gazebo/set_entity_state")
        self.model_state = self.node.create_client(GetEntityState,"/gazebo/get_entity_state")

        self.sub_contact = self.node.create_subscription(ContactsState,"/simple_drone/bumper",self.get_contact, 1)
        
        self.pos_sub = self.node.create_subscription(Pose,"/simple_drone/gt_pose",self.position_cb, 10)
        self.vel_sub = self.node.create_subscription(Twist,"/simple_drone/gt_vel",self.velocity_cb, 10)
        # self.sub_disparity  = self.node.create_subscription(Image,"/camera/depth/image_raw",self.get_laser, 10)
        self.sub_disparity_360  = self.node.create_subscription(LaserScan,"/simple_drone/scan",self.get_laser_360, qos_profile_laser)
        self.sub_disparity  = self.node.create_subscription(LaserScan,"/realsense_scan",self.get_laser, qos_profile_laser)
        self.sub_disparity_top  = self.node.create_subscription(LaserScan,"/realsense_scan_top",self.get_laser_top, qos_profile_laser)
        self.sub_disparity_bottom  = self.node.create_subscription(LaserScan,"/realsense_scan_bottom",self.get_laser_bottom, qos_profile_laser)

        # self.done_sub  = self.node.create_subscription(String,"/position_reached",self.get_done, qos_profile=1)
        # self.stop_pub = self.node.create_publisher(String,"/stop_control",1)

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
        #Reset Settings
        self.reset_msg = SetEntityState.Request()
        self.reset_msg.state.name = "/simple_drone"
        self.reset_msg.state.pose.position.x = 0.0
        self.reset_msg.state.pose.position.y = 0.0
        self.reset_msg.state.pose.position.z = 1.5
        self.image_counter = 0

        self.intial = True

       
        self.done = False
        self.del_model_prox = self.node.create_client(DeleteEntity,"delete_entity")
        self.spawn_model_client = self.node.create_client(SpawnEntity,"spawn_entity")

        self.contact = ContactsState()
        self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]),(2,),dtype= np.float64) 
        # self.spaces = {
        #         'laser': spaces.Box(0.0, 1.4, shape=(30,), dtype= np.float64),
        #         'goal': spaces.Box(-8.0, 8.0,shape=(4,), dtype= np.float64)
        #             }
        self.observation_space = spaces.Box(-8.0, 8.0, shape=(34,), dtype= np.float64)
        self.laser_done_cnt = 0
        self.ep_time = time.time()
        # self.disparity_img = Image()
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
        self.goal_data = np.zeros(4)

    def get_laser(self,msg):
        laser_ranges = msg.ranges
        final_laser_range = []
        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 0.2
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        # self.laser_ranges = np.roll(self.laser_ranges,1,axis=0)
        self.laser_ranges = final_laser_range

    def get_laser_top(self,msg):
        laser_ranges = msg.ranges
        final_laser_range = []
        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 0.2
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        # self.laser_ranges_top = np.roll(self.laser_ranges_top,1,axis=0)
        self.laser_ranges_top = final_laser_range

    def get_laser_bottom(self,msg):
        laser_ranges = msg.ranges
        final_laser_range = []
        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 0.2
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        # self.laser_ranges_bottom = np.roll(self.laser_ranges_bottom,1,axis=0)
        self.laser_ranges_bottom = final_laser_range

    def get_laser_360(self,msg):
        laser_ranges = msg.ranges
        final_laser_range = []
        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isinf(final_laser_range)] = 5.0
        final_laser_range = final_laser_range/5.0
        # self.laser_ranges_360 = np.roll(self.laser_ranges_360,1,axis=0)
        self.laser_ranges_360 = final_laser_range

    
    def node_spin(self):
        self.executor.spin()

    def get_contact(self,msg):
        self.contact = msg

        if(len(self.contact.states) > 0):
            if(self.contact.states[0].collision2_name != "ground_plane::link::collision"):
                self.done = True

        # print("contact")
                
    def velocity_cb(self,msg):
        msg = Pose()
        self.vel = msg

    def position_cb(self,msg):
        self.pose = msg
        orientation_q = [0.0,0.0,0.0,0.0]
        orientation_q[0] = msg.orientation.x
        orientation_q[1] = msg.orientation.y
        orientation_q[2] = msg.orientation.z
        orientation_q[3] = msg.orientation.w

        #trueYaw is the drones current yaw value
        self.pitch, self.roll, self.trueYaw = euler_from_quaternion(orientation_q)
        self.distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))
        self.goal_heading = math.atan2((self.goal[1] - self.pose.position.y),self.goal[0]-self.pose.position.x)
        if(abs(self.distance) < 0.5):
            self.done = True
            self.goal_reached = True

        if ( self.pitch > 1.57 or self.pitch < -1.57):
            for i in range(100):
                self.reset_msg.state.pose.position.x = self.pose.position.x
                self.reset_msg.state.pose.position.y = self.pose.position.y
                self.reset_proxy.wait_for_service(timeout_sec=0.2)
                future = self.reset_proxy.call_async(self.reset_msg)
                time.sleep(0.1)
        elif(self.roll > 1.57 or self.roll < -1.57):
            for i in range(100):
                self.reset_msg.state.pose.position.x = self.pose.position.x
                self.reset_msg.state.pose.position.y = self.pose.position.y
                self.reset_proxy.wait_for_service(timeout_sec=0.2)
                future = self.reset_proxy.call_async(self.reset_msg)
                time.sleep(0.1)

        
    def calculate_observation(self,data):
        ranges = list(data.ranges)
        return ranges

    def reset(self,seed=None,options=None):

        time.sleep(0.1)
        empty = Empty_msg()
        self.take_pub.publish(empty)
        # Resets the state of the environment and returns an initial observation.
        self.clear_trees()

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.linear.z = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

        time.sleep(2.5)
        future = self.reset_proxy.call_async(self.reset_msg)

        self.goal_reached = False
        self.overshoot = False
        

        time.sleep(2.0)

        drone_pos = Pose()
        drone_pos.position.x = 0.0
        drone_pos.position.y = 0.0
        self.tree_locations[0] = drone_pos
        self.randomize_trees()
        
        laser_combination_level = np.append(self.laser_ranges,self.laser_ranges_360)
        # laser_combination_level = self.laser_ranges
        # laser_combination =  np.append(self.laser_ranges_top,self.laser_ranges, self.laser_ranges_bottom, self.laser_ranges_360)
        self.closest_laser = np.min(laser_combination_level)
        # print(laser_combination)
        self.original_distance = math.sqrt(math.pow((self.goal[0] - self.pose.position.x),2) + math.pow((self.goal[1] - self.pose.position.y),2))

        # self.goal_data = np.roll(self.goal_data,1,axis=0)
        self.goal_data = np.array([0.0,0.0, self.distance, self.goal_heading - self.trueYaw])

        self.prev_distance = self.distance

        state =  np.append(laser_combination_level,self.goal_data)
        self.previous_error = 0.0

        while not self.unpause_proxy.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

        try:
            self.unpause_proxy.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        time.sleep(1.0)

        while not self.pause_proxy.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

        try:
            pass
            self.pause_proxy.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")


        self.ep_time = 0
        self.done = False
        self.goal_reached = False
        self.contact = ContactsState()

        return (state, {})

    def step(self, action):
        reward = 0.0
        truncated = False

        vel_cmd = Twist()
        # vel_cmd.linear.x = (action[0] + 1.01) * 0.05
        vel_cmd.linear.x = float(((action[0])+ 1.0)*0.2)
        vel_cmd.linear.y = 0.0
        vel_cmd.linear.z = 0.0
        vel_cmd.angular.z = float((action[1]))*0.25
        self.vel_pub.publish(vel_cmd)

        while not self.unpause_proxy.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

        try:
            self.unpause_proxy.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        time.sleep(0.1)

        while not self.pause_proxy.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

        try:
            pass
            self.pause_proxy.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        laser_combination_level = np.append(self.laser_ranges,self.laser_ranges_360)
        # laser_combination_level = self.laser_ranges
        # laser_combination =  np.append(self.laser_ranges_top,self.laser_ranges, self.laser_ranges_bottom, self.laser_ranges_360)
        self.closest_laser = np.min(laser_combination_level)
        # print(laser_combination)
        # self.goal_data = np.roll(self.goal_data,1,axis=0)
        self.goal_data = np.array([action[0],action[1], self.distance, self.goal_heading - self.trueYaw])
        # print(self.goal_heading - self.trueYaw)
        state =  np.append(laser_combination_level,self.goal_data)
        # print(state)

        if(self.ep_time > 200):
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
        
        # models = self.model_states
        for model in self.tree_locations:
            if( model.position.x + 1.8 > x > model.position.x - 1.8 and model.position.y + 1.8 > y > model.position.y - 1.8):
                pos_ok = False

        return pos_ok
    
    def check_pos_goal(self,x,y):
        pos_ok = True
        # pose = Pose()
        # pose.position.x = 99.0
        # pose.position.y = 99.0 
        # self.tree_locations[0] = pose
    # models = self.model_states
        for model in self.tree_locations:
            if( model.position.x + 0.7 > x > model.position.x - 0.7 and model.position.y + 0.7 > y > model.position.y - 0.7):
                pos_ok = False
        # pose.position.x = 0.0
        # pose.position.y = 0.0 
        # self.tree_locations[0] = pose
        return pos_ok
    

    def clear_trees(self):
        for i in range(1,6):
            delete_req = DeleteEntity.Request()
            delete_req.name = "pine_tree_" + str(i)
            future = self.del_model_prox.call_async(delete_req)

            if(len(self.tree_locations) > 1):
                self.tree_locations.pop()
            time.sleep(0.05)

        
        self.del_model_prox.wait_for_service(timeout_sec=1)
        delete_req = DeleteEntity.Request()
        delete_req.name = "goal"
        future = self.del_model_prox.call_async(delete_req)
    
    def randomize_trees(self):
        
        print("randomizing")
        for i in range(1,6):
            tree_ok = False
            tree_x = 0.0
            tree_y = 0.0
            while not tree_ok:
                tree_x = random.uniform(-4.5,4.5)
                tree_y = random.uniform(-4.5,4.5)
                tree_ok = self.check_pos(tree_x,tree_y)
            
            tree = random.randint(0,1)
            if(tree == 0):
                path = "/root/ros2_ws/src/f4-project/urdf/Tree.urdf"
            else:
                path = "/root/ros2_ws/src/f4-project/urdf/Tree_wider.urdf"

            quaternion = quaternion_from_euler(0.0, 0.0, random.randint(0,360))
            quat = Quaternion()
       
            quat.x = quaternion[0]
            quat.y = quaternion[1]
            quat.z = quaternion[2]
            quat.w = quaternion[3]

            spawn_req = SpawnEntity.Request()
            spawn_req.name="pine_tree_" + str(i)
            pose = Pose()
            pose.position.x = tree_x
            pose.position.y = tree_y
            pose.position.z = 2.0
            pose.orientation = quat

            self.tree_locations.append(pose)
            spawn_req.initial_pose = pose
            spawn_req.robot_namespace='/tree'
            spawn_req.reference_frame = 'world'
            spawn_req.xml = open(path,'r').read()

            future = self.spawn_model_client.call_async(spawn_req)
  
            
        goal_ok = False
        while not goal_ok:
            self.goal = [random.uniform(-3.0,3.0),random.uniform(-3.0,3.0)]
            goal_ok = self.check_pos_goal(self.goal[0],self.goal[1])


        time.sleep(0.7)

        spawn_req = SpawnEntity.Request()
        spawn_req.name="goal"
        pose = Pose()
        pose.position.x = self.goal[0]
        pose.position.y = self.goal[1]
        pose.position.z = 0.05

        spawn_req.initial_pose = pose
        spawn_req.robot_namespace="/goal"
        spawn_req.reference_frame = 'world'
        spawn_req.xml = open("/root/ros2_ws/src/f4-project/urdf/goal.urdf",'r').read()

        self.spawn_model_client.wait_for_service(timeout_sec=2)

        future = self.spawn_model_client.call_async(spawn_req)
        
        self.prev_distance = math.sqrt(math.pow(self.goal[0],2) + math.pow((self.goal[1]),2))

        time.sleep(2.0)




