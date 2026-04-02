#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, TwistStamped, Point
from std_msgs.msg import String
from sensor_msgs.msg import Image
import math
import time
import random
import threading
from cv_bridge import CvBridge
import cv2 as cv
from px4_msgs.msg import VehicleLocalPosition
from gazebo_msgs.srv import SetEntityState,SpawnEntity,DeleteEntity,GetEntityState
from gazebo_msgs.msg import ContactsState, EntityState
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from pid import PID
import os

class DroneGazeboEnv(gym.Env):
    def __init__(self):

        qos_profile_drone = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10
        )

        qos_profile_laser = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10
        )
       
        rclpy.init()
        self.node = rclpy.create_node("training")
        # self.node_2 = rclpy.create_node("training_env")

        # executor.add_node(self.node_2)
        
        self.vel_pub = self.node.create_publisher(Twist,'/cmd_vel', 10)

        # self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = self.node.create_client(SetEntityState,"/set_entity_state")
        self.model_state = self.node.create_client(GetEntityState,"/gazebo/get_entity_state")

        # self.set_model_pub = rospy.Publisher("/gazebo/set_model_state",ModelState,queue_size=10)
        # self.delete_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        # self.spawn_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # self.sub_laser  = rospy.Subscriber("/disparity", DisparityImage, callback = self.get_laser)
        
        self.sub_contact = self.node.create_subscription(ContactsState,"/bumper",self.get_contact, 1)
        
        self.pos_sub = self.node.create_subscription(PoseStamped,"/mavros/local_position/pose",self.position_cb, qos_profile_drone)
        self.vel_sub = self.node.create_subscription(TwistStamped,"/mavros/local_position/velocity_body",self.velocity_cb, qos_profile_drone)
        # self.sub_disparity  = self.node.create_subscription(Image,"/camera/depth/image_raw",self.get_laser, 10)
        self.sub_disparity_360  = self.node.create_subscription(LaserScan,"/scan",self.get_laser_360, qos_profile_laser)
        self.sub_disparity  = self.node.create_subscription(LaserScan,"/realsense_scan",self.get_laser, qos_profile_laser)
        self.sub_disparity_top  = self.node.create_subscription(LaserScan,"/realsense_scan_top",self.get_laser_top, qos_profile_laser)
        self.sub_disparity_bottom  = self.node.create_subscription(LaserScan,"/realsense_scan_bottom",self.get_laser_bottom, qos_profile_laser)

        # self.states_sub = self.node.create_subscription(ModelStates,"/gazebo/model_states",self.get_model_states, 10)
        
        # vel_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, callback = self.velocity_cb)
        self.stop_pub = self.node.create_publisher(String,"/stop_control",1)
        self.pose = PoseStamped()
        self.vel = TwistStamped()
        self.first_reset  = True
        # self.goals = [[3.0,-3.0],[5.0,5.0],[6.0,-2.0],[2.0,-2.0],[6.0,4.0],[2.0,2.0]]
        # self.goal = self.goals[random.randint(0,4)]

        self.goal = [random.uniform(1.5,10.0),random.uniform(-4.0,4.0)]
         
        self.prev_distance = math.sqrt(math.pow(self.goal[0],2) + math.pow(self.goal[1],2))
        self.distance = self.prev_distance
        self.goal_reached = False
        self.overshoot = False
        self.penalty = 0.0
        #Reset Settings
        self.reset_msg = SetEntityState.Request()
        self.reset_msg.state.name = "iris_rplidar"
        self.reset_msg.state.pose.position.x = 0.0
        self.reset_msg.state.pose.position.y = 0.0
        self.reset_msg.state.pose.position.z = 1.5
        self.image_counter = 0

       
        self.done = False
        self.del_model_prox = self.node.create_client(DeleteEntity,"delete_entity")
        self.spawn_model_client = self.node.create_client(SpawnEntity,"spawn_entity")

        self.contact = ContactsState()
        self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]),(2,),dtype= np.float64) 
        self.spaces = {
                'laser': spaces.Box(0.0, 5.5, shape=(3,160), dtype= np.float64),
                'goal': spaces.Box(-20.0,20.0,shape=(3,7), dtype= np.float64)
                    }
        self.observation_space = gym.spaces.Dict(self.spaces)
        self.laser_done_cnt = 0
        self.ep_time = time.time()
        self.disparity_img = Image()
        self.previous_error = 0.0
        # self.model_states = ModelStates()
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
        self.laser_ranges = np.zeros((3,40))
        self.laser_ranges_top = np.zeros((3,40))
        self.laser_ranges_bottom = np.zeros((3,40))
        self.laser_ranges_360 = np.zeros((3,40))
        self.goal_data = np.zeros((3,7))
        self.pid_distance = PID(Kp=0.001,Ki=0.00001,Kd=0.000015,Ku=1.0,sample_time=0.2,output_limits=(-0.7,0.7),integeral_limits=(-0.3,0.3))
        self.pid_obstacle = PID(Kp=0.0001,Ki=0.0001,Kd=0.0000015,Ku=1.0,sample_time=0.2,output_limits=(-0.7,0.7),integeral_limits=(-0.3,0.3))
        # rclpy.spin(self.node)

        # self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))
    def get_laser(self,msg):
        # self.disparity_img = msg
        # msg = LaserScan()
        laser_ranges = msg.ranges
        final_laser_range = []
        # for i in range (len(laser_ranges)):
        #     if i%10 == 0:
        #         final_laser_range.append(laser_ranges[i])

        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 0.2
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        self.laser_ranges = np.roll(self.laser_ranges,1,axis=0)
        self.laser_ranges[0] = final_laser_range

    def get_laser_top(self,msg):
        # self.disparity_img = msg
        # msg = LaserScan()
        laser_ranges = msg.ranges
        final_laser_range = []
        # for i in range (len(laser_ranges)):
        #     if i%10 == 0:
        #         final_laser_range.append(laser_ranges[i])

        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 0.2
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        self.laser_ranges_top = np.roll(self.laser_ranges_top,1,axis=0)
        self.laser_ranges_top[0] = final_laser_range

    def get_laser_bottom(self,msg):
        # self.disparity_img = msg
        # msg = LaserScan()
        laser_ranges = msg.ranges
        final_laser_range = []
        # for i in range (len(laser_ranges)):
        #     if i%10 == 0:
        #         final_laser_range.append(laser_ranges[i])

        final_laser_range = np.array(laser_ranges)
        final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 0.2
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        self.laser_ranges_bottom = np.roll(self.laser_ranges_bottom,1,axis=0)
        self.laser_ranges_bottom[0] = final_laser_range

    def get_laser_360(self,msg):
        # self.disparity_img = msg
        # msg = LaserScan()
        laser_ranges = msg.ranges
        # print(laser_ranges)
        final_laser_range = []
        # for i in range (len(laser_ranges)):
        #     if i%10 == 0:
        #         final_laser_range.append(laser_ranges[i])
        final_laser_range = np.array(laser_ranges)
        # final_laser_range[np.isnan(final_laser_range)] = 5.0
        final_laser_range[np.isinf(final_laser_range)] = 5.0
        final_laser_range = final_laser_range/5.0
        # print(final_laser_range)
        self.laser_ranges_360 = np.roll(self.laser_ranges_360,1,axis=0)
        self.laser_ranges_360[0] = final_laser_range
        # print(len(self.laser_ranges))

    # def get_model_states(self,msg):
    #     self.model_states = ModelStates()
    #     self.model_states.
    #     # print("got Models")

    def node_spin(self):
        while rclpy.ok:
            self.executor.spin()
            # if(self.close):
            #     self.close = False
            #     return

    def get_contact(self,msg):
        self.contact = msg

        if(len(self.contact.states) > 0):
            if(self.contact.states[0].collision2_name != "ground_plane::link::collision"):
                self.done = True

        # print("contact")
                
    def velocity_cb(self,msg):
        self.vel = msg

    def position_cb(self,msg):
        self.pose = msg
        # print(self.pose.pose.position)

        orientation_q = [0.0,0.0,0.0,0.0]
        orientation_q[0] = msg.pose.orientation.x
        orientation_q[1] = msg.pose.orientation.y
        orientation_q[2] = msg.pose.orientation.z
        orientation_q[3] = msg.pose.orientation.w

        #trueYaw is the drones current yaw value
        self.pitch, self.roll, self.trueYaw = euler_from_quaternion(orientation_q)

        
        self.distance = math.sqrt(math.pow((self.goal[0] - self.pose.pose.position.x),2) + math.pow((self.goal[1] - self.pose.pose.position.y),2))
        self.goal_heading = math.atan2((self.goal[1] - self.pose.pose.position.y),self.goal[0]-self.pose.pose.position.x)
        # print(self.pose.pose.position.x,self.pose.pose.position.y,self.goal_heading)
        if(abs(self.distance) < 0.5):
            self.done = True
            self.goal_reached = True

        if(self.pose.pose.position.x > 15):
            self.done = True
            self.goal_reached = False
            self.overshoot = False

        if ( self.pitch > 1.57 or self.pitch < -1.57):
            for i in range(100):
                self.reset_msg.state.pose.position.x = self.pose.pose.position.x
                self.reset_msg.state.pose.position.y = self.pose.pose.position.y
                self.reset_proxy.wait_for_service(timeout_sec=0.2)
                future = self.reset_proxy.call_async(self.reset_msg)
                time.sleep(0.1)
        elif(self.roll > 1.57 or self.roll < -1.57):
            for i in range(100):
                self.reset_msg.state.pose.position.x = self.pose.pose.position.x
                self.reset_msg.state.pose.position.y = self.pose.pose.position.y
                self.reset_proxy.wait_for_service(timeout_sec=0.2)
                future = self.reset_proxy.call_async(self.reset_msg)
                time.sleep(0.1)

        # if(self.pose.pose.position.y > 6.0 or self.pose.pose.position.y < -6.0):
        #     self.done = True
        #     self.goal_reached = False


    # def velocity_cb(self,msg):
    #     self.vel = msg
        
    def calculate_observation(self,data):
        # self.penalty = 0.0
        ranges = list(data.ranges)
        
        # min_laser = 7.0
        # for x in range(len(ranges)):
        #     if(float(ranges[x]) <= 1.5):
        #         if(float(ranges[x]) < min_laser):
        #             min_laser = float(ranges[x])
        #             self.penalty = -round(min(pow(4,1.5/float(ranges[x])),200),4)
        


        # if(abs(self.prev_distance) < 0.3):
        #     self.done = True
        #     self.goal_reached = True

        return ranges

    def reset(self,seed=None,options=None):
        # Resets the state of the environment and returns an initial observation.
        self.clear_trees()
        self.pid_distance.reset()
        self.pid_obstacle.reset()

        time.sleep(8.0)

        #randomize trees
        # t1= threading.Thread(target=self.randomize_trees)
        # t1.start()
        # self.prev_img = np.zeros([12,50,100])

        # self.image_counter = 0

        stop_msg = String()
        stop_msg.data = "half"
        self.stop_pub.publish(stop_msg)

            
        time.sleep(2.5)

        stop_msg = String()
        stop_msg.data = "stop"
        self.stop_pub.publish(stop_msg)
        # time.sleep(0.5)
    
        self.goal_reached = False
        self.overshoot = False

        model_request = GetEntityState.Request()
        model_request.name = "iris_rplidar"
        model_request.reference_frame = "world"

  
        time.sleep(8.0)

        # if(self.first_reset):
        #     time.sleep(40)
        #     self.first_reset = False

        # self.reset_robot()
        # for i in range(100):
        #     self.reset_proxy.wait_for_service(timeout_sec=0.2)
        #     future = self.reset_proxy.call_async(self.reset_msg)
        #     # self.close = True
        #     # self.et.join()
        #     # self.executor.spin_until_future_complete(future,timeout_sec=0.5)
        #     # self.et = threading.Thread(target=self.node_spin)
        #     # self.et.start()
        #     # rclpy.spin_until_future_complete(self.node, future, timeout_sec=2)
        #     time.sleep(0.1)

        # Unpause simulation to make observation
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")
        #read laser data
        # time.sleep(0.3)
        # stop_msg.data = "resume"
        # self.stop_pub.publish(stop_msg)
        if(self.model_state.service_is_ready()):
            future = self.model_state.call_async(model_request)
            while(not future.done()):
                if(future.cancelled()):
                    break
            print(future.result().state.pose.position.z)
            if(future.result().state.pose.position.z > 2.6 or future.result().state.pose.position.z < 0.3):
                for i in range(20):
                    self.reset_msg.state.pose.position.x = self.pose.pose.position.x
                    self.reset_msg.state.pose.position.y = self.pose.pose.position.y
                    self.reset_proxy.wait_for_service(timeout_sec=0.2)
                    future = self.reset_proxy.call_async(self.reset_msg)
                    time.sleep(0.1)

                time.sleep(2)
                
                if(self.model_state.service_is_ready()):
                    future = self.model_state.call_async(model_request)
                    while(not future.done()):
                        if(future.cancelled()):
                            break
                        if(future.result().state.pose.position.z > 2.6 or future.result().state.pose.position.z < 0.3):
                            print("drone immobile")
                            os.system('pkill -9 python')

        drone_pos = Pose()
        drone_pos.position.x = future.result().state.pose.position.x
        drone_pos.position.y = future.result().state.pose.position.y
        self.tree_locations[0] = drone_pos
        self.randomize_trees()
        # bridge = CvBridge()
        # dataimg = bridge.imgmsg_to_cv2(data,"32FC1")
        # dataimg = cv.normalize(dataimg, dataimg, 0, 255, cv.NORM_MINMAX)
        # dataimg = np.array(dataimg, dtype = np.uint8)
        # dataimg = cv.resize(dataimg, (100,100), interpolation = cv.INTER_CUBIC)
        # dataimg = dataimg[25:75,0:100]

        # final_img = np.concatenate((dataimg,self.prev_img[5],self.prev_img[11]),axis=0)


        # self.prev_img=np.roll(self.prev_img, 1,axis=0)
        # self.prev_img[0]  = np.copy(dataimg)
        # dataimg = dataimg[133:263,0:400]
        # cv.imshow("window",dataimg)
        # cv.waitKey(0)
        # dataimg = cv.resize(dataimg, (200,200), interpolation = cv.INTER_AREA)
        # np.expand_dims(dataimg, axis=0)
        # self.randomize_trees()
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        
        # orientation_list = [self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w]
        # roll,pitch,yaw = tf.transformations.euler_from_quaternion(orientation_list)
        laser_combination_level = np.concatenate([self.laser_ranges,self.laser_ranges_360],axis=1)
        laser_combination = np.concatenate([self.laser_ranges_top,self.laser_ranges, self.laser_ranges_bottom, self.laser_ranges_360],axis=1)
        self.closest_laser = np.min(laser_combination_level)
        # print(laser_combination)
        self.original_distance = math.sqrt(math.pow((self.goal[0] - self.pose.pose.position.x),2) + math.pow((self.goal[1] - self.pose.pose.position.y),2))

        self.goal_data = np.roll(self.goal_data,1,axis=0)
        self.goal_data[0] = np.array([(self.goal[0] - self.pose.pose.position.x) / self.goal[0] ,(self.goal[1] - self.pose.pose.position.y)/ self.goal[1], self.goal_heading/3.14 ,self.vel.twist.linear.x/0.4, self.vel.twist.linear.y/0.2, self.vel.twist.linear.z/0.05,self.distance/self.original_distance])

        state =  {"laser": laser_combination  , "goal":self.goal_data}
        # if(len(state) == 720):
        self.previous_error = 0.0

        self.ep_time = 0
        self.done = False
        self.goal_reached = False
        # print(state.shape)
        # else:
            # state = np.zeros((720,))
        # t1.join()

        return (state, {})

    def step(self, action):
        reward = 0.0
        truncated = False
        # Move the agent based on the selected action
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/unpause_physics service call failed")

        # if(not math.isnan(action[0]) and not math.isnan(action[1]) and not math.isnan(action[2])):
        vel_cmd = Twist()
        vel_cmd.linear.x = (action[0] + 1.01)* 0.025
        vel_cmd.linear.y = (action[1] * 0.07)
        # vel_cmd.linear.z = (action[2] * 0.025)
        vel_cmd.angular.z = self.goal_heading
        self.vel_pub.publish(vel_cmd)

        
        time.sleep(0.2)

        # self.randomize_trees()
        # data = self.disparity_img 
        # bridge = CvBridge()
        # dataimg = bridge.imgmsg_to_cv2(data,"32FC1")
        # dataimg = cv.normalize(dataimg, dataimg, 0, 255, cv.NORM_MINMAX)
        # dataimg = np.array(dataimg, dtype = np.uint8)
        # dataimg = cv.resize(dataimg, (100,100), interpolation = cv.INTER_CUBIC)
        # dataimg = dataimg[25:75,0:100]
        # final_img = np.concatenate((dataimg,self.prev_img[5],self.prev_img[11]),axis=0)
        
        # self.prev_img[0]  = np.copy(dataimg)
        # self.prev_img=np.roll(self.prev_img, 1, axis=0)
        # dataimg = dataimg[133:263,0:400]
        # cv.imshow("window",dataimg)
        # cv.waitKey(0)
        
        # dataimg = cv.resize(dataimg, (200,200), interpolation = cv.INTER_AREA)
        # np.expand_dims(final_img, axis=0)
        # print(data.ranges)
        # time.sleep(0.01)
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")
        
        # goal_heading = math.atan2((self.goal[1] - self.pose.pose.position.y),self.goal[0]-self.pose.pose.position.x)
        # orientation_list = [self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w]
        # roll,pitch,yaw = tf.transformations.euler_from_quaternion(orientation_list)
        # distance = math.sqrt(math.pow((self.goal[0] - self.pose.pose.position.x),2) + math.pow((self.goal[1] - self.pose.pose.position.y),2))
        laser_combination_level = np.concatenate([self.laser_ranges,self.laser_ranges_360],axis=1)
        laser_combination = np.concatenate([self.laser_ranges_top,self.laser_ranges, self.laser_ranges_bottom, self.laser_ranges_360],axis=1)
        self.closest_laser = np.min(laser_combination_level)
        # print(laser_combination)
        self.goal_data = np.roll(self.goal_data,1,axis=0)
        self.goal_data[0] = np.array([(self.goal[0] - self.pose.pose.position.x) / self.goal[0] ,(self.goal[1] - self.pose.pose.position.y)/ self.goal[1], self.goal_heading/3.14 ,self.vel.twist.linear.x/0.4, self.vel.twist.linear.y/0.4, self.vel.twist.linear.z/0.05,self.distance/self.original_distance])
        
        state =  {"laser": laser_combination  , "goal":self.goal_data}


        # self.executor.spin_once()

        if(self.ep_time > 400):
            self.done = True
            truncated = True

        if not self.done:
                # distance_pid = self.pid_distance.update(0.0,self.distance/self.original_distance)
                # pid_forward = self.pid_forward.update(0.0,abs(action[1]))
                # reward = (self.previous_error - distance_pid) + pid_forward
                # self.previous_error = distance_pid

                distance_pid = self.pid_distance.update(0.0,self.distance)
                pid_obstacle = self.pid_obstacle.update(0.0,self.closest_laser)
                reward = (distance_pid) - pid_obstacle
                print(reward)
                if(np.isnan(reward)):
                    reward = 0.0
                # reward = reward/100.0
                # print(distance_pid,pid_forward)
                # print(reward,self.distance,goal_heading-yaw)
        else:
            if(self.goal_reached):
                reward = 100.0
            elif(self.overshoot):
                reward = -100 + ((1.0/self.distance) * 24)
            else:
                reward = -100
            # reward = reward/100.0
            # print(reward)
            # print(reward)

        self.ep_time+=1
        return state, reward, self.done, truncated, {}
  

    def render(self):
        pass


    def check_pos(self,x,y):
        pos_ok = True
        
        # models = self.model_states
        for model in self.tree_locations:
            if( model.position.x + 1.8 > x > model.position.x - 1.8 and model.position.y + 1.8 > y > model.position.y - 1.8):
                pos_ok = False

        return pos_ok
    

    def clear_trees(self):
        for i in range(1,15):
            delete_req = DeleteEntity.Request()
            delete_req.name = "pine_tree_" + str(i)
            future = self.del_model_prox.call_async(delete_req)
            # self.close = True
            # self.et.join()
            # self.executor.spin_once_until_future_complete(future,timeout_sec=0.5)
            # self.et = threading.Thread(target=self.node_spin)
            # self.et.start()
            # rclpy.spin_until_future_complete(self.node, future, timeout_sec=2)
            if(len(self.tree_locations) > 1):
                self.tree_locations.pop()
            time.sleep(0.05)

        
        self.del_model_prox.wait_for_service(timeout_sec=5)
        delete_req = DeleteEntity.Request()
        delete_req.name = "goal"
        future = self.del_model_prox.call_async(delete_req)
    
    def randomize_trees(self,):
        
        print("randomizing")

        # for i in range(1,15):
        #     delete_req = DeleteEntity.Request()
        #     delete_req.name = "pine_tree_" + str(i)
        #     future = self.del_model_prox.call_async(delete_req)
        #     # self.close = True
        #     # self.et.join()
        #     # self.executor.spin_once_until_future_complete(future,timeout_sec=0.5)
        #     # self.et = threading.Thread(target=self.node_spin)
        #     # self.et.start()
        #     # rclpy.spin_until_future_complete(self.node, future, timeout_sec=2)
        #     if(len(self.tree_locations) > 1):
        #         self.tree_locations.pop()
        #     time.sleep(0.25)

        # self.del_model_prox.wait_for_service(timeout_sec=5)
        # delete_req = DeleteEntity.Request()
        # delete_req.name = "goal"
        # future = self.del_model_prox.call_async(delete_req)

        # time.sleep(3.0)

        for i in range(1,15):
            tree_ok = False
            tree_x = 0.0
            tree_y = 0.0
            while not tree_ok:
                tree_x = random.uniform(-5.5,5.5)
                tree_y = random.uniform(-5.5,5.5)
                tree_ok = self.check_pos(tree_x,tree_y)
            
            tree = random.randint(0,1)
            if(tree == 0):
                path = "/home/anas/ros2_ws/src/f4-project/urdf/Tree.urdf"
            else:
                path = "/home/anas/ros2_ws/src/f4-project/urdf/Tree_wider.urdf"

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
            # time.sleep(0.01)
            # self.close = True
            # self.et.join()
            # self.executor.spin_once_until_future_complete(future,timeout_sec=0.5)
            # self.et = threading.Thread(target=self.node_spin)
            # self.et.start()
            # rclpy.spin_until_future_complete(self.node, future, timeout_sec=2)
            
        goal_ok = False
        while not goal_ok:
            self.goal = [random.uniform(-5.0,5.0),random.uniform(-5.0,5.0)]
            goal_ok = self.check_pos(self.goal[0],self.goal[1])

        # self.close = True
        # self.et.join()
        # self.executor.spin_once_until_future_complete(future,timeout_sec=0.5)
        # self.et = threading.Thread(target=self.node_spin)
        # self.et.start()
        # rclpy.spin_until_future_complete(self.node, future, timeout_sec= 2)

        time.sleep(1.5)

        spawn_req = SpawnEntity.Request()
        spawn_req.name="goal"
        pose = Pose()
        pose.position.x = self.goal[0]
        pose.position.y = self.goal[1]
        pose.position.z = 0.05

        spawn_req.initial_pose = pose
        spawn_req.robot_namespace="/goal"
        spawn_req.reference_frame = 'world'
        spawn_req.xml = open("/home/anas/ros2_ws/src/f4-project/urdf/goal.urdf",'r').read()

        self.spawn_model_client.wait_for_service(timeout_sec=5)

        future = self.spawn_model_client.call_async(spawn_req)
        # self.close = True
        # self.et.join()
        # self.executor.spin_once_until_future_complete(future,timeout_sec=0.5)
        # self.et = threading.Thread(target=self.node_spin)
        # self.et.start()
        # rclpy.spin_until_future_complete(self.node, future, timeout_sec= 2)
        
        self.prev_distance = math.sqrt(math.pow(self.goal[0],2) + math.pow((self.goal[1]),2))

        time.sleep(5.0)




    # def get_quaternion_from_euler(self,roll, pitch, yaw):

    #     qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    #     qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    #     qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    #     qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
    #     return [qx, qy, qz, qw]
    
    # def reset_robot(self):

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     #resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")
        # self.counter +=1
        # delete_req = DeleteModelRequest()
        # delete_req.model_name="iris_rplidar"
        # rospy.wait_for_service('/gazebo/delete_model',5)
        # self.delete_proxy.call(delete_req)
        # time.sleep(0.5)

        # spawn_req = SpawnModelRequest()
        # spawn_req.model_name="iris_rplidar_" + str(self.counter)
        # model_xml = open('/home/anas/catkin_ws/src/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris_rplidar/iris_rplidar.sdf','r').read()
        # spawn_req.model_xml=model_xml
        # spawn_req.initial_pose = Pose()
        # spawn_req.initial_pose.position.x = 0.0
        # spawn_req.initial_pose.position.y = 0.0
        # spawn_req.initial_pose.position.z = 0.0
        # spawn_req.reference_frame = "world"
        # rospy.wait_for_service('/gazebo/spawn_sdf_model',5)
        # self.spawn_proxy.call(spawn_req)

        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        # except (rospy.ServiceException) as e:
            # print ("/gazebo/unpause_physics service call failed")



