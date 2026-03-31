import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class IFGMNode(Node):
    def __init__(self):
        super().__init__('ifgm_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.dir_pub = self.create_publisher(Float32, '/cmd_dir', 10)

        self.latest_scan = None
        self.latest_ranges = []
        self.latest_angles = []

        self.front_ranges = []
        self.front_angles = []
        self.free_mask = []
        self.gaps = []
        self.best_gap = None

        self.last_steering_angle = 0

        self.safe_distance = 0.3
        self.front_center_angle = math.pi       # 180° = front
        self.field_of_view = math.pi / 3        # ±60°
        self.max_steering_angle = math.radians(15.0)

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

        self.extract_front_view(self.field_of_view)

        for a, r in zip(self.front_angles[:5], self.front_ranges[:5]):
            print(f"angle: {a:.3f}")

        self.build_free_mask()
        self.find_gaps()
        self.select_best_gap()
        self.publish_dir()

        free_count = sum(self.free_mask)
        blocked_count = len(self.free_mask) - free_count

        if self.best_gap is not None:
            self.get_logger().info(
                f'Front scan points: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: {len(self.gaps)} | '
                f'best gap center angle: {self.best_gap["center_angle"]:.2f} rad | '
                f'steering angle: {self.best_gap["steering_angle"]:.2f} rad | '
                f'cmd_dir: {self.best_gap["cmd_dir"]:.2f} | '
                f'width: {self.best_gap["width"]}'
            )
        else:
            self.get_logger().info(
                f'Front scan points: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: 0'
            )

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def extract_front_view(self, max_angle):
        front_points = []

        for angle, r in zip(self.latest_angles, self.latest_ranges):
            relative_to_front = self.normalize_angle(angle - self.front_center_angle)

            if abs(relative_to_front) <= max_angle:
                front_points.append((relative_to_front, r))

        front_points.sort(key=lambda p: p[0])

        self.front_angles = [p[0] for p in front_points]
        self.front_ranges = [p[1] for p in front_points]

    def build_free_mask(self):
        raw_mask = []

        for r in self.front_ranges:
            raw_mask.append(r > self.safe_distance)

        self.free_mask = raw_mask[:]
        self.inflate_obstacles(index_radius=3)

    def inflate_obstacles(self, index_radius=3):
        inflated = self.free_mask[:]

        for i, is_free in enumerate(self.free_mask):
            if not is_free:
                start = max(0, i - index_radius)
                end = min(len(inflated) - 1, i + index_radius)

                for j in range(start, end + 1):
                    inflated[j] = False

        self.free_mask = inflated

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

        best_start = 0
        best_end = -1
        widest_width = -1

        # Erst nur das beste Gap finden
        for start_idx, end_idx in self.gaps:
            width = end_idx - start_idx + 1

            if width > widest_width:
                widest_width = width
                best_start = start_idx
                best_end = end_idx

        # Dann genau einmal alles Weitere berechnen
        center_idx = (best_start + best_end) // 2
        center_angle = self.front_angles[center_idx]

        steering_angle = -0.8 * center_angle + 0.2 * self.last_steering_angle
        self.last_steering_angle = steering_angle

        cmd_dir = steering_angle / self.max_steering_angle
        cmd_dir = max(-1.0, min(1.0, cmd_dir))

        self.best_gap = {
            'start_idx': best_start,
            'end_idx': best_end,
            'center_idx': center_idx,
            'center_angle': center_angle,
            'steering_angle': steering_angle,
            'cmd_dir': cmd_dir,
            'width': widest_width,
        }


    def publish_dir(self):
        msg = Float32()

        if self.best_gap is None:
            msg.data = 0.0
        else:
            msg.data = float(self.best_gap['cmd_dir'])

        self.dir_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = IFGMNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()