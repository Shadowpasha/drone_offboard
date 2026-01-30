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
        self.declare_parameter('hold_duration', 10.0) # Duration for each flight phase
        self.declare_parameter('forward_distance', 1.0) # Distance to move forward

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        
        self.altitude = self.get_parameter('altitude').value
        self.hold_duration = self.get_parameter('hold_duration').value
        self.forward_distance = self.get_parameter('forward_distance').value

        # State machine variables
        self.flight_state = "PRE_FLIGHT"
        self.offboard_setpoint_counter = 0
        self.state_timer_start = None
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        
        # Position variables (Initialized to 0, updated by odometry)
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        self.current_pos_z = 0.0
        
        self.start_pos_x = 0.0
        self.start_pos_y = 0.0
        self.start_pos_z = 0.0
        
        self.target_pos_x = 0.0
        self.target_pos_y = 0.0

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_odometry_callback(self, msg):
        # Update current position
        self.current_pos_x = msg.position[0]
        self.current_pos_y = msg.position[1]
        self.current_pos_z = msg.position[2]
        
        # Calculate yaw from quaternion (w, x, y, z)
        # PX4 msg.q is [w, x, y, z]
        q = msg.q
        self.current_yaw = np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))

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

        # State Machine: Pre-Flight -> Armed -> Takeoff -> Move Forward -> Land
        if self.flight_state == "PRE_FLIGHT":
            # Continuously update start position to current position until we are ready to fly
            self.start_pos_x = self.current_pos_x
            self.start_pos_y = self.current_pos_y
            self.start_pos_z = self.current_pos_z
            self.target_yaw = self.current_yaw
            
            # Calculate forward position based on yaw
            self.target_pos_x = self.start_pos_x + (self.forward_distance * np.cos(self.target_yaw))
            self.target_pos_y = self.start_pos_y + (self.forward_distance * np.sin(self.target_yaw))

            # 1. Send setpoints matching current position (don't try to move yet)
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            trajectory_msg.position[0] = self.start_pos_x
            trajectory_msg.position[1] = self.start_pos_y
            trajectory_msg.position[2] = self.start_pos_z
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # 2. Wait for a few setpoints before trying to arm
            if self.offboard_setpoint_counter >= 10:
                self.flight_state = "ARMED_WAIT"
                self.get_logger().info(f"Setup complete. Start Pos: ({self.start_pos_x:.1f}, {self.start_pos_y:.1f}, {self.start_pos_z:.1f}). Target NED: ({self.target_pos_x:.1f}, {self.target_pos_y:.1f})")

            self.offboard_setpoint_counter += 1

        elif self.flight_state == "ARMED_WAIT":
            # Keep sending setpoints to ensure offboard mode stays active
            # Use the locked start_pos values from the moment we left PRE_FLIGHT
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            trajectory_msg.position[0] = self.start_pos_x
            trajectory_msg.position[1] = self.start_pos_y
            trajectory_msg.position[2] = self.start_pos_z
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # Check if we are in the correct state
            if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and 
                self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
                self.flight_state = "TAKEOFF"
                self.state_timer_start = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info("Takeoff started.")
            else:
                # RETRY LOGIC
                if self.offboard_setpoint_counter % 10 == 0:
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self.get_logger().info(f"Waiting... Nav: {self.nav_state}, Arm: {self.arming_state}")
                
                self.offboard_setpoint_counter += 1

        elif self.flight_state == "TAKEOFF":
            # Ascend to target altitude (Vertical Only)
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            trajectory_msg.position[0] = self.start_pos_x
            trajectory_msg.position[1] = self.start_pos_y
            trajectory_msg.position[2] = self.start_pos_z - self.altitude # NED Z is negative Up
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # Wait for hold_duration to ensure altitude is reached and stable
            current_time = self.get_clock().now().nanoseconds / 1e9
            if (current_time - self.state_timer_start) > self.hold_duration:
                self.flight_state = "MOVE_FORWARD"
                self.state_timer_start = current_time # Reset timer for next phase
                self.get_logger().info(f"Takeoff complete. Moving forward {self.forward_distance}m.")

        elif self.flight_state == "MOVE_FORWARD":
            # Move to the calculated forward coordinates at altitude
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            trajectory_msg.position[0] = self.target_pos_x
            trajectory_msg.position[1] = self.target_pos_y
            trajectory_msg.position[2] = self.start_pos_z - self.altitude
            trajectory_msg.yaw = self.target_yaw
            self.publisher_trajectory.publish(trajectory_msg)

            # Wait for hold_duration to ensure target is reached
            current_time = self.get_clock().now().nanoseconds / 1e9
            if (current_time - self.state_timer_start) > self.hold_duration:
                self.flight_state = "LANDING"
                self.get_logger().info("Target reached. Landing.")
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