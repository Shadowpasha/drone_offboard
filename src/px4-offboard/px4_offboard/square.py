#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleOdometry


class SquareOffboardControl(Node):

    def __init__(self):
        super().__init__('square_offboard_control')

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub)
        
        self.odometry_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile_sub)

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile_pub)

        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.dt = timer_period

        self.declare_parameter('altitude', 1.0)
        self.declare_parameter('speed', 0.1) # 0.1 m/s
        self.declare_parameter('hold_duration', 5.0)
        self.declare_parameter('square_side', 2.0)
        
        self.altitude = self.get_parameter('altitude').value
        self.speed = self.get_parameter('speed').value
        self.hold_duration = self.get_parameter('hold_duration').value
        self.square_side = self.get_parameter('square_side').value
        self.step_size = self.speed * self.dt
        
        self.declare_parameter('takeoff_speed', 0.1) # m/s (maximum)
        self.takeoff_speed = self.get_parameter('takeoff_speed').value
        self.takeoff_acceleration = 0.01 # m/s^2
        self.current_takeoff_speed = 0.01 # m/s (starting speed)
        self.active_setpoint_z = 0.0

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        
        # State machine variables
        self.flight_state = "PRE_FLIGHT"
        self.offboard_setpoint_counter = 0
        self.state_timer_start = None
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.data_valid = False
        
        # Position variables
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        self.current_pos_z = 0.0
        
        self.active_setpoint_x = 0.0
        self.active_setpoint_y = 0.0
        
        # Waypoints (WP0 is takeoff/home, WP1-4 are square corners)
        self.wp_home = np.array([0.0, 0.0, 0.0])
        self.wp_1 = [0.0, 0.0, 0.0] # Forward
        self.wp_2 = [0.0, 0.0, 0.0] # Left
        self.wp_3 = [0.0, 0.0, 0.0] # Backward
        self.wp_4 = [0.0, 0.0, 0.0] # Right (back to home)

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_odometry_callback(self, msg):
        # Update current position
        self.current_pos_x = msg.position[0]
        self.current_pos_y = msg.position[1]
        self.current_pos_z = msg.position[2]
        
        # Calculate yaw from quaternion (w, x, y, z)
        q = msg.q
        self.current_yaw = np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
        self.data_valid = True

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.publisher_vehicle_command.publish(msg)

    def calculate_square_waypoints(self):
        """Calculates the 4 corners of the square based on start position and yaw."""
        # WP0: Home (Takeoff position)
        self.wp_home = [self.current_pos_x, self.current_pos_y, self.current_pos_z]
        
        d = self.square_side
        yaw = self.target_yaw
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        # WP1: Forward (Body X+)
        # North shift = d * cos(yaw), East shift = d * sin(yaw)
        self.wp_1 = [
            self.wp_home[0] + d * cos_y,
            self.wp_home[1] + d * sin_y,
            self.wp_home[2]
        ]

        # WP2: Left of WP1 (Body Y-)
        # Left direction is Yaw - 90 deg
        # North shift = d * cos(yaw - pi/2) = d * sin(yaw)
        # East shift = d * sin(yaw - pi/2) = -d * cos(yaw)
        self.wp_2 = [
            self.wp_1[0] + d * np.sin(yaw),
            self.wp_1[1] - d * np.cos(yaw),
            self.wp_home[2]
        ]

        # WP3: Backward of WP2 (Body X-)
        # Backward direction is Yaw - 180 deg
        self.wp_3 = [
            self.wp_2[0] - d * cos_y,
            self.wp_2[1] - d * sin_y,
            self.wp_home[2]
        ]

        # WP4: Right of WP3 (Body Y+) -> Should return to Home
        # Right direction is Yaw + 90 deg
        self.wp_4 = [
            self.wp_3[0] - d * np.sin(yaw),
            self.wp_3[1] + d * np.cos(yaw),
            self.wp_home[2]
        ]

    def publish_setpoint(self, target_ned):
        # Smoothly interpolate active_setpoint towards target_ned
        dx = target_ned[0] - self.active_setpoint_x
        dy = target_ned[1] - self.active_setpoint_y
        dist = np.sqrt(dx**2 + dy**2)

        if dist > self.step_size:
            self.active_setpoint_x += (dx / dist) * self.step_size
            self.active_setpoint_y += (dy / dist) * self.step_size
        else:
            self.active_setpoint_x = target_ned[0]
            self.active_setpoint_y = target_ned[1]

        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        trajectory_msg.position[0] = self.active_setpoint_x
        trajectory_msg.position[1] = self.active_setpoint_y
        trajectory_msg.position[2] = self.active_setpoint_z
        trajectory_msg.yaw = self.target_yaw
        self.publisher_trajectory.publish(trajectory_msg)
        return dist

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.flight_state == "PRE_FLIGHT":
            if self.data_valid:
                self.target_yaw = self.current_yaw
                self.calculate_square_waypoints()

                # Send setpoints at current position (ground)
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                trajectory_msg.position[0] = self.wp_home[0]
                trajectory_msg.position[1] = self.wp_home[1]
                trajectory_msg.position[2] = self.wp_home[2]
                trajectory_msg.yaw = self.target_yaw
                self.publisher_trajectory.publish(trajectory_msg)

                if self.offboard_setpoint_counter >= 10:
                    self.flight_state = "TAKEOFF"
                    self.active_setpoint_x = self.wp_home[0]
                    self.active_setpoint_y = self.wp_home[1]
                    self.active_setpoint_z = self.wp_home[2]
                    self.get_logger().info("Waypoints calculated. Taking off.")

                self.offboard_setpoint_counter += 1

        elif self.flight_state == "TAKEOFF":
            # Accelerate takeoff
            if self.current_takeoff_speed < self.takeoff_speed:
                self.current_takeoff_speed += self.takeoff_acceleration * self.dt
            
            # Update Z setpoint
            target_z = self.wp_home[2] - self.altitude
            dz = target_z - self.active_setpoint_z
            step = self.current_takeoff_speed * self.dt
            if abs(dz) > step:
                self.active_setpoint_z += np.sign(dz) * step
            else:
                self.active_setpoint_z = target_z

            # Command takeoff (Ascend vertical)
            self.publish_setpoint(self.wp_home)

            # Recurrently spam Arm/Offboard commands
            if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if self.offboard_setpoint_counter % 10 == 0:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            
            self.offboard_setpoint_counter += 1

            # Monitor altitude
            dist_z = abs(self.current_pos_z - (self.wp_home[2] - self.altitude))
            if dist_z < 0.1:
                self.flight_state = "HOLD_AFTER_TAKEOFF"
                self.state_timer_start = current_time
                self.get_logger().info("Altitude reached. Holding for stability.")

        elif self.flight_state == "HOLD_AFTER_TAKEOFF":
            # Maintain altitude
            self.publish_setpoint(self.wp_home)
            
            # Wait for 2 seconds
            if (current_time - self.state_timer_start) > 2.0:
                self.flight_state = "WAYPOINT_1"
                self.state_timer_start = current_time
                self.get_logger().info("Stability hold complete. Moving Forward (WP1)")

        elif self.flight_state == "WAYPOINT_1":
            dist = self.publish_setpoint(self.wp_1)
            if dist < 0.05:
                if (current_time - self.state_timer_start) > self.hold_duration:
                    self.flight_state = "WAYPOINT_2"
                    self.state_timer_start = current_time
                    self.get_logger().info("Moving Left (WP2)")

        elif self.flight_state == "WAYPOINT_2":
            dist = self.publish_setpoint(self.wp_2)
            if dist < 0.05:
                if (current_time - self.state_timer_start) > self.hold_duration:
                    self.flight_state = "WAYPOINT_3"
                    self.state_timer_start = current_time
                    self.get_logger().info("Moving Backward (WP3)")

        elif self.flight_state == "WAYPOINT_3":
            dist = self.publish_setpoint(self.wp_3)
            if dist < 0.05:
                if (current_time - self.state_timer_start) > self.hold_duration:
                    self.flight_state = "WAYPOINT_4"
                    self.state_timer_start = current_time
                    self.get_logger().info("Moving Right/Home (WP4)")

        elif self.flight_state == "WAYPOINT_4":
            dist = self.publish_setpoint(self.wp_4)
            if dist < 0.05:
                if (current_time - self.state_timer_start) > self.hold_duration:
                    self.flight_state = "DESCENDING"
                    self.get_logger().info("Square complete. Descending slowly.")
                    self.current_takeoff_speed = 0.1 # Start slow for descent

        elif self.flight_state == "DESCENDING":
            # Accelerate descent
            if self.current_takeoff_speed < self.takeoff_speed:
                self.current_takeoff_speed += self.takeoff_acceleration * self.dt
                
            # Smoothly interpolate active_setpoint_z towards ground
            target_z = self.wp_home[2] - 0.1
            dz = target_z - self.active_setpoint_z
            step = self.current_takeoff_speed * self.dt
            
            if abs(dz) > step:
                self.active_setpoint_z += np.sign(dz) * step
            else:
                self.active_setpoint_z = target_z
                
            self.publish_setpoint(self.wp_home)

            # Once low enough, land
            if abs(self.current_pos_z - target_z) < 0.1:
                self.flight_state = "LANDING"
                self.get_logger().info("Final approach. Landing.")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        elif self.flight_state == "LANDING":
            if self.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.flight_state = "FINISHED"
                self.get_logger().info("Mission Complete.")


def main(args=None):
    rclpy.init(args=args)
    offboard_control = SquareOffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()