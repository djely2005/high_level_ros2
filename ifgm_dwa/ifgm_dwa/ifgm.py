import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class IFGMNode(Node):
    def __init__(self):
        super().__init__('ifgm_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.latest_scan = None
        self.latest_ranges = []
        self.latest_angles = []

        self.front_ranges = []
        self.front_angles = []
        self.free_mask = []

        self.safe_distance = 0.8  # Meter

        self.get_logger().info('IFGM node started, waiting for /scan ...')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

        clean_ranges = []
        angles = []

        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment

            if math.isinf(r):
                clean_r = msg.range_max
            elif math.isnan(r):
                clean_r = msg.range_min
            else:
                clean_r = max(msg.range_min, min(r, msg.range_max))

            clean_ranges.append(clean_r)
            angles.append(angle)

        self.latest_ranges = clean_ranges
        self.latest_angles = angles

        self.extract_front_view(math.pi/2)
        self.build_free_mask()

        free_count = sum(self.free_mask)
        blocked_count = len(self.free_mask) - free_count

        self.get_logger().info(
            f'Front scan points: {len(self.front_ranges)} | '
            f'free: {free_count} | blocked: {blocked_count}'
        )

    def extract_front_view(self, max_angle):
        self.front_ranges = []
        self.front_angles = []

        for angle, r in zip(self.latest_angles, self.latest_ranges):
            if -max_angle <= angle <= max_angle:
                self.front_angles.append(angle)
                self.front_ranges.append(r)

    def build_free_mask(self):
        self.free_mask = []

        for r in self.front_ranges:
            is_free = r > self.safe_distance
            self.free_mask.append(is_free)


def main(args=None):
    rclpy.init(args=args)

    node = IFGMNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()