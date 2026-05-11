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


class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

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
        self.declare_parameter('hold_duration', 10.0)
        self.declare_parameter('speed', 0.1)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        
        self.altitude = self.get_parameter('altitude').value
        self.hold_duration = self.get_parameter('hold_duration').value
        self.speed = self.get_parameter('speed').value

        # State machine variables
        self.flight_state = "IDLE"
        self.offboard_setpoint_counter = 0
        
        # Position variables
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        self.current_pos_z = 0.0
        
        self.active_setpoint_x = 0.0
        self.active_setpoint_y = 0.0
        self.active_setpoint_z = 0.0

        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0
        self.origin_set = False

        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.data_valid = False
        self.hold_time_start = None
        self.target_z_goal = 0.0

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_odometry_callback(self, msg):
        self.current_pos_x = msg.position[0]
        self.current_pos_y = msg.position[1]
        self.current_pos_z = msg.position[2]
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

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

        if not self.data_valid:
            return

        if self.flight_state == "IDLE":
            self.target_yaw = self.current_yaw
            
            # Send few setpoints before switching to offboard
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.position[0] = self.current_pos_x
            trajectory_msg.position[1] = self.current_pos_y
            trajectory_msg.position[2] = self.current_pos_z
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            if self.offboard_setpoint_counter >= 10:
                self.active_setpoint_x = self.current_pos_x
                self.active_setpoint_y = self.current_pos_y
                self.active_setpoint_z = self.current_pos_z
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.flight_state = "TAKEOFF"
            self.offboard_setpoint_counter += 1

        elif self.flight_state == "TAKEOFF":
            if not self.origin_set:
                if self.arming_state == VehicleStatus.ARMING_STATE_ARMED and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self.origin_x = self.current_pos_x
                    self.origin_y = self.current_pos_y
                    self.origin_z = self.current_pos_z
                    self.origin_set = True
                    self.active_setpoint_x = self.current_pos_x
                    self.active_setpoint_y = self.current_pos_y
                    self.active_setpoint_z = self.current_pos_z
                    self.get_logger().info(f"Origin captured: [{self.origin_x:.2f}, {self.origin_y:.2f}, {self.origin_z:.2f}]")
                else:
                    # Maintain current position setpoint until armed/offboard
                    trajectory_msg = TrajectorySetpoint()
                    trajectory_msg.position[0] = self.current_pos_x
                    trajectory_msg.position[1] = self.current_pos_y
                    trajectory_msg.position[2] = self.current_pos_z
                    trajectory_msg.yaw = self.target_yaw
                    self.publisher_trajectory.publish(trajectory_msg)

                    if self.offboard_setpoint_counter % 10 == 0:
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self.offboard_setpoint_counter += 1
                    return

            # Re-send Arm/Offboard commands if not in correct state
            if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if self.offboard_setpoint_counter % 10 == 0:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.offboard_setpoint_counter += 1

            self.target_z_goal = self.origin_z - self.altitude
            
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.position[0] = self.active_setpoint_x
            trajectory_msg.position[1] = self.active_setpoint_y
            trajectory_msg.position[2] = self.target_z_goal
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # Monitor altitude
            if abs(self.current_pos_z - self.target_z_goal) < 0.5:
                if self.hold_time_start is None:
                    self.hold_time_start = self.get_clock().now().nanoseconds / 1e9
                    self.get_logger().info("Altitude reached. Holding.")
                
                current_time = self.get_clock().now().nanoseconds / 1e9
                if (current_time - self.hold_time_start) > self.hold_duration:
                    self.flight_state = "LANDING"
                    self.get_logger().info("Landing command sent.")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        elif self.flight_state == "LANDING":
            if self.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().info("Mission Complete.")
                self.flight_state = "FINISHED"



def main(args=None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()