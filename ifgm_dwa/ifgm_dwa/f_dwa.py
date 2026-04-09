#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class DWA_Velocity_Node(Node):

    def __init__(self):

        super().__init__('dwa_velocity_node')

        ###########################################
        # paramètres dynamiques (article)
        ###########################################

        self.dt = 0.1

        self.v_max = 0.25     # m/s
        self.v_min = 0.05     # m/s

        self.a_max = 0.8      # accélération max
        self.decel_max = 0.8  # freinage max

        ###########################################
        # poids fonction objectif
        ###########################################

        self.w_dist = 0.7
        self.w_vel  = 0.3

        ###########################################

        self.scan = None
        self.phi = 0.0

        self.v_current = self.v_min

        ###########################################
        # ROS2 interfaces
        ###########################################

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.create_subscription(
            Float32,
            '/phi',
            self.phi_callback,
            10)

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        ###########################################

        self.timer = self.create_timer(
            self.dt,
            self.control_loop)

        self.get_logger().info("DWA velocity node started")


    ###########################################
    # callbacks
    ###########################################

    def scan_callback(self, msg):

        ranges = np.array(msg.ranges)

        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max

        self.scan = ranges


    def phi_callback(self, msg):

        self.phi = msg.data


    ###########################################
    # distance minimale obstacle
    ###########################################

    def compute_dmin(self):

        if self.scan is None:

            return 10.0

        return float(np.min(self.scan))


    ###########################################
    # fenêtre dynamique
    ###########################################

    def dynamic_window(self):

        v_min = max(
            self.v_min,
            self.v_current - self.decel_max * self.dt)

        v_max = min(
            self.v_max,
            self.v_current + self.a_max * self.dt)

        return v_min, v_max


    ###########################################
    # contrainte sécurité
    ###########################################

    def admissible_velocity(self, dmin):

        v_stop = np.sqrt(
            2 * dmin * self.decel_max)

        return min(self.v_max, v_stop)


    ###########################################
    # fonction objectif DWA
    ###########################################

    def objective_function(self, v, dmin):

        score_dist = dmin
        score_vel  = v

        return (
            self.w_dist * score_dist
            + self.w_vel * score_vel
        )


    ###########################################
    # optimisation vitesse
    ###########################################

    def compute_velocity(self):

        dmin = self.compute_dmin()

        v_min, v_max = self.dynamic_window()

        v_safe = self.admissible_velocity(dmin)

        v_samples = np.linspace(v_min, v_max, 6)

        best_v = self.v_min
        best_score = -1e9

        for v in v_samples:

            v = min(v, v_safe)

            score = self.objective_function(v, dmin)

            if score > best_score:

                best_score = score
                best_v = v

        self.v_current = best_v

        return best_v


    ###########################################
    # boucle principale
    ###########################################

    def control_loop(self):

        if self.scan is None:

            return

        v = self.compute_velocity()

        cmd = Twist()

        cmd.linear.x = float(v)

        # l'angle sera appliqué par IFGM ou un autre node
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


###########################################
# main
###########################################

def main(args=None):

    rclpy.init(args=args)

    node = DWA_Velocity_Node()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()