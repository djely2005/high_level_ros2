#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

# =============================================================================
# PARAMETRES IDENTIQUES WEBOTS
# =============================================================================

IDX_AVANT = 180
CHAMP_VISION = 60

MASQUE_ROBOT_DEBUT = 330
MASQUE_ROBOT_FIN = 30

BUBBLE_RADIUS = 0.15
MAX_LIDAR_RANGE = 6.0
WINDOW_SIZE = 5
MIN_GAP_SIZE = 8

SPEED_MAX_KMH = 5.0
SPEED_MIN_KMH = 2.0

STEER_GAIN = 1.2
MAX_STEER_RAD = 0.28

NB_POINTS = 360
DEG_PAR_IDX = 360.0 / NB_POINTS

SEUIL_MUR_AVANT = 0.35
DUREE_RECUL_S = 1.5
SPEED_RECUL_KMH = -2.0

# normalisation pour voiture réelle
MAX_SPEED_KMH_REAL = 28.0


# =============================================================================
# FOLLOW THE GAP
# =============================================================================

class FollowGap:

    def preprocess_lidar(self, ranges):

        proc = np.array(ranges)

        proc = np.where(
            np.isinf(proc) | np.isnan(proc) | (proc == 0),
            MAX_LIDAR_RANGE,
            proc)

        proc = np.clip(proc, 0, MAX_LIDAR_RANGE)

        proc[MASQUE_ROBOT_DEBUT:] = MAX_LIDAR_RANGE
        proc[:MASQUE_ROBOT_FIN] = MAX_LIDAR_RANGE

        mask = np.zeros(NB_POINTS, dtype=bool)

        mask[
            IDX_AVANT-CHAMP_VISION:
            IDX_AVANT+CHAMP_VISION] = True

        proc[~mask] = 0

        kernel = np.ones(WINDOW_SIZE)/WINDOW_SIZE

        proc = np.convolve(proc,kernel,mode="same")

        return proc


    def apply_bubble(self,proc):

        champ = proc[
            IDX_AVANT-CHAMP_VISION:
            IDX_AVANT+CHAMP_VISION]

        local_min = np.argmin(champ)

        closest = (
            IDX_AVANT-CHAMP_VISION
            + local_min)

        dist = proc[closest]

        if dist>0 and BUBBLE_RADIUS<dist:

            half_angle = np.arcsin(
                min(BUBBLE_RADIUS/dist,1))

        else:

            half_angle = np.radians(20)

        bubble_half = int(
            half_angle/
            np.radians(DEG_PAR_IDX))

        i0 = max(0,closest-bubble_half)
        i1 = min(NB_POINTS-1,closest+bubble_half)

        proc[i0:i1+1] = 0

        return proc


    def find_max_gap(self,free):

        best_s = IDX_AVANT
        best_e = IDX_AVANT
        best_len = 0

        start = None

        for i in range(
            IDX_AVANT-CHAMP_VISION,
            IDX_AVANT+CHAMP_VISION):

            if free[i]>0:

                if start is None:
                    start = i

            else:

                if start is not None:

                    L = i-start

                    if L>best_len:

                        best_len = L
                        best_s = start
                        best_e = i-1

                    start = None


        if start is not None:

            L = IDX_AVANT+CHAMP_VISION-start

            if L>best_len:

                best_s = start
                best_e = IDX_AVANT+CHAMP_VISION-1


        return best_s,best_e


    def find_best_point(self,s,e,proc):

        idx = np.arange(s,e+1)

        w = proc[s:e+1]

        if w.sum()>0:

            best = int(
                np.round(
                    np.average(idx,weights=w)))

        else:

            best = (s+e)//2

        return best


    def compute_command(self,ranges):

        proc = self.preprocess_lidar(ranges)

        free = self.apply_bubble(proc.copy())

        s,e = self.find_max_gap(free)

        if e-s < MIN_GAP_SIZE:

            dist_g = np.mean(
                proc[
                IDX_AVANT-CHAMP_VISION:
                IDX_AVANT-CHAMP_VISION//2])

            dist_d = np.mean(
                proc[
                IDX_AVANT+CHAMP_VISION//2:
                IDX_AVANT+CHAMP_VISION])

            steer = (
                MAX_STEER_RAD
                if dist_g>dist_d
                else -MAX_STEER_RAD)

            return SPEED_MIN_KMH,steer,proc


        best = self.find_best_point(s,e,proc)

        offset = np.radians(
            (best-IDX_AVANT)
            * DEG_PAR_IDX)

        steer = np.clip(
            STEER_GAIN*offset,
            -MAX_STEER_RAD,
            MAX_STEER_RAD)

        ratio = abs(steer)/MAX_STEER_RAD

        speed = (
            SPEED_MAX_KMH
            - ratio*(SPEED_MAX_KMH-SPEED_MIN_KMH)
        )

        dist_avant = np.mean(
            proc[
            IDX_AVANT-10:
            IDX_AVANT+10])

        if dist_avant>3 and abs(steer)<0.05:

            speed = min(
                speed*1.2,
                SPEED_MAX_KMH)

        return speed,steer,proc


    def dist_avant(self,proc):

        return float(
            np.mean(
            proc[
            IDX_AVANT-10:
            IDX_AVANT+10]))


# =============================================================================
# NODE ROS2
# =============================================================================

class FollowGapNode(Node):

    def __init__(self):

        super().__init__("follow_gap_v4")

        self.ftg = FollowGap()

        self.scan = None

        self.mode_recul = False
        self.t_recul_fin = 0
        self.steer_recul = 0

        self.create_subscription(

            LaserScan,

            "/scan",

            self.scan_callback,

            10)

        self.pub_dir = self.create_publisher(

            Float32,

            "/cmd_dir",

            10)

        self.pub_vel = self.create_publisher(

            Float32,

            "/cmd_vel",

            10)

        self.timer = self.create_timer(

            0.05,

            self.control_loop)


        self.get_logger().info("FTG v4 node started")


    def scan_callback(self,msg):

        ranges = np.array(msg.ranges)

        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max

        self.scan = ranges


    def publish_cmd(self,speed_kmh,steer):

        dir_msg = Float32()
        vel_msg = Float32()

        dir_msg.data = float(
            np.clip(
                steer/MAX_STEER_RAD,
                -1,
                1))

        vel_msg.data = float(
            np.clip(
                speed_kmh/MAX_SPEED_KMH_REAL,
                -1,
                1))

        self.pub_dir.publish(dir_msg)
        self.pub_vel.publish(vel_msg)


    def control_loop(self):

        if self.scan is None:
            return

        t = self.get_clock().now().nanoseconds*1e-9


        speed,steer,proc = self.ftg.compute_command(self.scan)

        dist_av = self.ftg.dist_avant(proc)


        ########################################
        # marche arrière
        ########################################

        if self.mode_recul and t>=self.t_recul_fin:

            self.mode_recul = False


        if self.mode_recul:

            speed = SPEED_RECUL_KMH
            steer = self.steer_recul


        elif dist_av < SEUIL_MUR_AVANT:

            dist_g = np.mean(
                proc[
                IDX_AVANT-CHAMP_VISION:
                IDX_AVANT-10])

            dist_d = np.mean(
                proc[
                IDX_AVANT+10:
                IDX_AVANT+CHAMP_VISION])

            self.steer_recul = (
                -MAX_STEER_RAD
                if dist_g>dist_d
                else MAX_STEER_RAD)

            self.mode_recul = True

            self.t_recul_fin = t + DUREE_RECUL_S

            speed = SPEED_RECUL_KMH
            steer = self.steer_recul


        ########################################

        self.publish_cmd(speed,steer)


# =============================================================================

def main():

    rclpy.init()

    node = FollowGapNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__=="__main__":

    main()
