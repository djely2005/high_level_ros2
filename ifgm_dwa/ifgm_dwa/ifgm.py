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
        self.gaps = []
        self.best_gap = None

        self.safe_distance = 0.8

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

        self.extract_front_view(math.pi / 2)
        self.build_free_mask()
        self.find_gaps()
        self.select_best_gap()

        free_count = sum(self.free_mask)
        blocked_count = len(self.free_mask) - free_count

        if self.best_gap is not None:
            self.get_logger().info(
                f'Front scan points: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: {len(self.gaps)} | '
                f'best gap center angle: {self.best_gap["center_angle"]:.2f} rad | '
                f'width: {self.best_gap["width"]}'
            )
        else:
            self.get_logger().info(
                f'Front scan points: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: 0'
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

    def find_gaps(self):
        self.gaps = []
        start_idx = None

        for i, is_free in enumerate(self.free_mask):
            if is_free and start_idx is None:
                start_idx = i
            elif not is_free and start_idx is not None:
                self.gaps.append((start_idx, i - 1))
                start_idx = None

        if start_idx is not None:
            self.gaps.append((start_idx, len(self.free_mask) - 1))

    def select_best_gap(self):
        self.best_gap = None

        if not self.gaps:
            return

        widest_width = -1

        for start_idx, end_idx in self.gaps:
            width = end_idx - start_idx + 1

            if width > widest_width:
                center_idx = (start_idx + end_idx) // 2
                center_angle = self.front_angles[center_idx]

                self.best_gap = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'center_idx': center_idx,
                    'center_angle': center_angle,
                    'width': width,
                }

                widest_width = width


def main(args=None):
    rclpy.init(args=args)

    node = IFGMNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()