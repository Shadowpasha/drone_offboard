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

        self.declare_parameter('altitude', 1.5)
        self.declare_parameter('hold_duration', 10.0)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        
        self.altitude = self.get_parameter('altitude').value
        self.hold_duration = self.get_parameter('hold_duration').value

        # State machine variables
        self.flight_state = "PRE_FLIGHT"
        self.offboard_setpoint_counter = 0
        # Position variables
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        self.current_pos_z = 0.0
        
        self.start_pos_x = 0.0
        self.start_pos_y = 0.0
        self.start_pos_z = 0.0

        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.data_valid = False
        
        self.declare_parameter('takeoff_speed', 0.05) # m/s (maximum)
        self.takeoff_speed = self.get_parameter('takeoff_speed').value
        self.takeoff_acceleration = 0.001 # m/s^2
        self.current_takeoff_speed = 0.001 # m/s (starting speed)
        self.active_setpoint_z = 0.0
        self.hold_time_start = None

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_odometry_callback(self, msg):
        # Update current position (NED)
        self.current_pos_x = msg.position[0]
        self.current_pos_y = msg.position[1]
        self.current_pos_z = msg.position[2]

        # Calculate yaw from quaternion (w, x, y, z)
        # PX4 msg.q is [w, x, y, z]
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

        # State Machine: PRE_FLIGHT -> TAKEOFF -> LANDING
        if self.flight_state == "PRE_FLIGHT":
            # Update target yaw and start position to match current state while on ground
            if self.data_valid:
                self.target_yaw = self.current_yaw
                self.start_pos_x = self.current_pos_x
                self.start_pos_y = self.current_pos_y
                self.start_pos_z = self.current_pos_z

                # 1. Send setpoints matching current position (ground)
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                trajectory_msg.position[0] = self.start_pos_x
                trajectory_msg.position[1] = self.start_pos_y
                trajectory_msg.position[2] = self.start_pos_z
                trajectory_msg.yaw = self.target_yaw
                self.publisher_trajectory.publish(trajectory_msg)

                # 2. Transition to TAKEOFF after a short buffer
                if self.offboard_setpoint_counter >= 10:
                    self.flight_state = "TAKEOFF"
                    self.active_setpoint_z = self.start_pos_z
                    self.current_takeoff_speed = 0.1 # Reset starting speed
                    self.get_logger().info(f"Taking off to {self.altitude}m relative to start.")

                self.offboard_setpoint_counter += 1

        elif self.flight_state == "TAKEOFF":
            # Accelerate the takeoff speed
            if self.current_takeoff_speed < self.takeoff_speed:
                self.current_takeoff_speed += self.takeoff_acceleration * self.dt

            # Smoothly interpolate active_setpoint_z towards target altitude
            target_z = self.start_pos_z - self.altitude
            dz = target_z - self.active_setpoint_z
            
            step = self.current_takeoff_speed * self.dt
            if abs(dz) > step:
                self.active_setpoint_z += np.sign(dz) * step
            else:
                self.active_setpoint_z = target_z

            # Command takeoff to altitude
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            trajectory_msg.position[0] = self.start_pos_x
            trajectory_msg.position[1] = self.start_pos_y
            trajectory_msg.position[2] = self.active_setpoint_z
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # Recurrently spam Arm/Offboard commands until we have actually taken off
            # This follows the training script approach to ensure initialization
            if self.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                if self.offboard_setpoint_counter % 10 == 0:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            
            self.offboard_setpoint_counter += 1

            # Monitor altitude to move to hold/landing
            dist_z = abs(self.current_pos_z - (self.start_pos_z - self.altitude))
            if dist_z < 0.2:
                # Start hold timer if not already started
                if self.hold_time_start is None:
                    self.hold_time_start = self.get_clock().now().nanoseconds / 1e9
                    self.get_logger().info("Altitude reached. Holding.")

            if self.hold_time_start is not None:
                current_time = self.get_clock().now().nanoseconds / 1e9
                if (current_time - self.hold_time_start) > self.hold_duration:
                    self.flight_state = "DESCENDING"
                    self.get_logger().info("Holding duration finished. Descending slowly.")
                    self.current_takeoff_speed = 0.1 # Re-use for descent speed

        elif self.flight_state == "DESCENDING":
            # Smoothly interpolate active_setpoint_z towards ground (start_pos_z)
            target_z = self.start_pos_z - 0.1 # Aim for 10cm above ground
            dz = target_z - self.active_setpoint_z
            
            # Accelerate the descent slightly
            if self.current_takeoff_speed < self.takeoff_speed:
                self.current_takeoff_speed += self.takeoff_acceleration * self.dt

            step = self.current_takeoff_speed * self.dt
            if abs(dz) > step:
                self.active_setpoint_z += np.sign(dz) * step
            else:
                self.active_setpoint_z = target_z

            # Publish the sliding active_setpoint
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            trajectory_msg.position[0] = self.start_pos_x
            trajectory_msg.position[1] = self.start_pos_y
            trajectory_msg.position[2] = self.active_setpoint_z
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # Once we are low enough, trigger final autonomous landing
            if abs(self.current_pos_z - target_z) < 0.1:
                self.flight_state = "LANDING"
                self.get_logger().info("Final approach. Landing.")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        elif self.flight_state == "LANDING":
            # Wait for disarm
            if self.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.flight_state = "FINISHED"
                self.get_logger().info("Mission Complete.")


def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()