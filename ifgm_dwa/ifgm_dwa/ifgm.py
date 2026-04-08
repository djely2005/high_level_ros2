import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class HybridGapFollowerNode(Node):
    def __init__(self):
        super().__init__('hybrid_gap_follower_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.dir_pub = self.create_publisher(Float32, '/cmd_dir', 10)
        self.vel_pub = self.create_publisher(Float32, '/cmd_vel', 10)

        # =========================
        # State
        # =========================
        self.latest_ranges = []
        self.latest_angles = []

        self.front_indices = []
        self.front_ranges = []
        self.front_angles = []

        self.free_mask = []
        self.gaps = []
        self.best_gap = None

        self.last_steering_angle = 0.0
        self.current_angle = 0.0
        self.current_speed_kmh = 0.0

        self.reverse_counter = 0
        self.escape_mode = False

        # =========================
        # Limits
        # =========================
        self.max_speed_kmh = 28.0
        self.max_steering_angle = 0.28  # rad

        # =========================
        # Scan geometry
        # =========================
        self.front_center_angle = math.pi
        self.front_half_width = math.radians(90.0)

        self.left_corridor_angle = math.radians(240.0)
        self.right_corridor_angle = math.radians(120.0)
        self.rear_center_angle = 0.0

        # =========================
        # IFGM parameters
        # =========================
        self.safe_distance = 1.3
        self.inflate_radius = 10

        # =========================
        # Corridor assist
        # =========================
        self.corridor_mode_distance = 1.8
        self.very_tight_distance = 0.8
        self.corridor_sector_half_width = 8
        self.corridor_gain = 0.9

        # =========================
        # Steering / speed behavior
        # =========================
        self.steering_smoothing = 0.75
        self.max_steer_step = 0.035
        self.aggressive_steering = 2.5
        self.momentum = 0.5

        self.open_speed = 5.0
        self.tight_speed = 3.5
        self.very_tight_speed = 2.5

        # =========================
        # Emergency reverse
        # =========================
        self.front_stop_distance = 0.2
        self.reverse_speed = -2.0
        self.reverse_steps_total = 12

        self.get_logger().info('Hybrid gap follower node started, waiting for /scan ...')

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def mean(values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def sanitize_scan(self, msg: LaserScan):
        clean_ranges = []
        angles = []

        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment

            if math.isinf(r):
                clean_r = msg.range_max
            elif math.isnan(r):
                clean_r = msg.range_max
            else:
                clean_r = self.clamp(r, msg.range_min, msg.range_max)

            clean_ranges.append(clean_r)
            angles.append(angle)

        self.latest_ranges = clean_ranges
        self.latest_angles = angles

    def angle_to_index(self, target_angle):
        best_idx = 0
        best_err = float('inf')

        for i, a in enumerate(self.latest_angles):
            err = abs(self.normalize_angle(a - target_angle))
            if err < best_err:
                best_err = err
                best_idx = i

        return best_idx

    def get_sector_values_by_index(self, center_idx, half_width):
        if not self.latest_ranges:
            return []

        start = max(0, center_idx - half_width)
        end = min(len(self.latest_ranges) - 1, center_idx + half_width)
        return self.latest_ranges[start:end + 1]

    def get_sector_mean_by_index(self, center_idx, half_width):
        vals = self.get_sector_values_by_index(center_idx, half_width)
        return self.mean(vals)

    def get_sector_min_by_index(self, center_idx, half_width):
        vals = self.get_sector_values_by_index(center_idx, half_width)
        return min(vals) if vals else float('inf')

    def extract_front_view(self):
        front_points = []

        for i, (angle, r) in enumerate(zip(self.latest_angles, self.latest_ranges)):
            rel = self.normalize_angle(angle - self.front_center_angle)
            if abs(rel) <= self.front_half_width:
                front_points.append((rel, r, i))

        front_points.sort(key=lambda x: x[0])

        self.front_angles = [p[0] for p in front_points]
        self.front_ranges = [p[1] for p in front_points]
        self.front_indices = [p[2] for p in front_points]

    def build_free_mask(self):
        raw_mask = [r > self.safe_distance for r in self.front_ranges]

        inflated = raw_mask[:]
        for i, is_free in enumerate(raw_mask):
            if not is_free:
                start = max(0, i - self.inflate_radius)
                end = min(len(inflated) - 1, i + self.inflate_radius)
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

    def choose_gap_target(self, start_idx, end_idx):
        best_idx = start_idx
        best_score = -1e9
        gap_center = 0.5 * (start_idx + end_idx)

        for i in range(start_idx, end_idx + 1):
            distance_score = self.front_ranges[i]
            center_score = -abs(i - gap_center)
            edge_bonus = min(i - start_idx, end_idx - i)

            score = 2.0 * distance_score + 0.25 * center_score + 0.10 * edge_bonus

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def select_best_gap(self):
        self.best_gap = None

        if not self.gaps:
            return

        best_score = -1e9
        front_center_local = len(self.front_ranges) // 2

        for start_idx, end_idx in self.gaps:
            width = end_idx - start_idx + 1
            gap_ranges = self.front_ranges[start_idx:end_idx + 1]
            mean_depth = self.clamp(self.mean(gap_ranges), 0.5, 3.0)

            gap_mid_idx = (start_idx + end_idx) // 2
            target_idx = self.choose_gap_target(start_idx, end_idx)

            target_angle = self.front_angles[target_idx]

            # Keep the same sign convention as your working node:
            steering_angle = (
                -self.aggressive_steering * target_angle
                + self.momentum * self.last_steering_angle
            )
            steering_angle = self.clamp(
                steering_angle,
                -self.max_steering_angle,
                self.max_steering_angle
            )

            score = 1.0 * width + 1.8 * mean_depth

            if score > best_score:
                best_score = score
                self.best_gap = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'target_idx': target_idx,
                    'target_angle': target_angle,
                    'gap_mid_idx': gap_mid_idx,
                    'gap_mid_angle': self.front_angles[gap_mid_idx],
                    'width': width,
                    'mean_depth': mean_depth,
                    'score': score,
                    'steering_angle': steering_angle,
                }

        if self.best_gap is not None:
            self.last_steering_angle = self.best_gap['steering_angle']

    def compute_corridor_angle(self):
        left_idx = self.angle_to_index(self.left_corridor_angle)
        right_idx = self.angle_to_index(self.right_corridor_angle)

        left_mean = self.get_sector_mean_by_index(left_idx, self.corridor_sector_half_width)
        right_mean = self.get_sector_mean_by_index(right_idx, self.corridor_sector_half_width)

        diff = left_mean - right_mean
        corridor_angle = self.corridor_gain * diff

        corridor_angle = self.clamp(
            corridor_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        return corridor_angle, left_mean, right_mean

    def compute_front_clearance(self):
        front_idx = self.angle_to_index(self.front_center_angle)
        front_min = self.get_sector_min_by_index(front_idx, 10)
        front_mean = self.get_sector_mean_by_index(front_idx, 25)
        return front_min, front_mean

    def compute_rear_clearance(self):
        rear_idx = self.angle_to_index(self.rear_center_angle)
        rear_min = self.get_sector_min_by_index(rear_idx, 15)
        rear_mean = self.get_sector_mean_by_index(rear_idx, 30)
        return rear_min, rear_mean

    def blend_angles(self, gap_angle, corridor_angle, front_clearance):
        if front_clearance >= self.corridor_mode_distance:
            w_corridor = 0.0
        elif front_clearance <= self.very_tight_distance:
            w_corridor = 1.0
        else:
            span = self.corridor_mode_distance - self.very_tight_distance
            w_corridor = (self.corridor_mode_distance - front_clearance) / span

        w_gap = 1.0 - w_corridor
        blended = w_gap * gap_angle + w_corridor * corridor_angle

        return self.clamp(
            blended,
            -self.max_steering_angle,
            self.max_steering_angle
        ), w_corridor

    def smooth_steering(self, current_angle, desired_angle):
        blended = self.steering_smoothing * current_angle + (1.0 - self.steering_smoothing) * desired_angle
        delta = blended - current_angle
        delta = self.clamp(delta, -self.max_steer_step, self.max_steer_step)

        return self.clamp(
            current_angle + delta,
            -self.max_steering_angle,
            self.max_steering_angle
        )

    def compute_auto_speed(self, front_clearance, steering_angle):
        if front_clearance < self.very_tight_distance:
            target_speed = self.very_tight_speed
        elif front_clearance < self.corridor_mode_distance:
            target_speed = self.tight_speed
        else:
            target_speed = self.open_speed

        target_speed -= 1.0 * abs(steering_angle)
        return max(1.0, target_speed)

    def choose_escape_steering(self):
        return 0.0

    def emergency_escape_control(self, front_min):
        rear_min, _ = self.compute_rear_clearance()

        if self.reverse_counter > 0:
            self.reverse_counter -= 1

            if rear_min < 0.35:
                self.escape_mode = False
                self.reverse_counter = 0
                return True, 0.0, 0.0, f'EMERGENCY REAR BLOCKED | rear_min: {rear_min:.2f}'

            self.escape_mode = True
            return True, self.reverse_speed, self.choose_escape_steering(), (
                f'REVERSING STRAIGHT | steps_left: {self.reverse_counter} | '
                f'front_min: {front_min:.2f} | rear_min: {rear_min:.2f}'
            )

        if front_min < self.front_stop_distance:
            if rear_min < 0.35:
                self.escape_mode = False
                return True, 0.0, 0.0, (
                    f'EMERGENCY FRONT+REAR BLOCKED | front_min: {front_min:.2f} | rear_min: {rear_min:.2f}'
                )

            self.reverse_counter = self.reverse_steps_total - 1
            self.escape_mode = True
            return True, self.reverse_speed, 0.0, (
                f'START REVERSE STRAIGHT | front_min: {front_min:.2f} | rear_min: {rear_min:.2f}'
            )

        if self.escape_mode:
            self.escape_mode = False

        return False, 0.0, 0.0, ''

    def publish_commands(self, speed_kmh, angle):
        angle = self.clamp(angle, -self.max_steering_angle, self.max_steering_angle)
        speed_kmh = self.clamp(speed_kmh, -self.max_speed_kmh, self.max_speed_kmh)

        dir_msg = Float32()
        vel_msg = Float32()

        # Same steering sign convention as your working node
        dir_msg.data = float(angle / self.max_steering_angle)
        dir_msg.data = self.clamp(dir_msg.data, -1.0, 1.0)

        # Speed normalized to [-1, 1]
        vel_msg.data = float(speed_kmh / self.max_speed_kmh)
        vel_msg.data = self.clamp(vel_msg.data, -1.0, 1.0)

        self.dir_pub.publish(dir_msg)
        self.vel_pub.publish(vel_msg)

    def scan_callback(self, msg: LaserScan):
        self.sanitize_scan(msg)

        if not self.latest_ranges:
            return

        self.extract_front_view()
        self.build_free_mask()
        self.find_gaps()
        self.select_best_gap()

        gap_angle = 0.0 if self.best_gap is None else self.best_gap['steering_angle']

        corridor_angle, left_mean, right_mean = self.compute_corridor_angle()
        front_min, front_mean = self.compute_front_clearance()

        emergency_active, emergency_speed, emergency_angle, emergency_msg = self.emergency_escape_control(front_min)

        if emergency_active:
            self.current_speed_kmh = emergency_speed
            self.current_angle = emergency_angle
            self.publish_commands(self.current_speed_kmh, self.current_angle)
            self.get_logger().warn(emergency_msg)
            return

        desired_angle, corridor_weight = self.blend_angles(gap_angle, corridor_angle, front_mean)

        self.current_angle = self.smooth_steering(self.current_angle, desired_angle)
        self.current_speed_kmh = self.compute_auto_speed(front_mean, self.current_angle)

        self.publish_commands(self.current_speed_kmh, self.current_angle)

        free_count = sum(self.free_mask)
        blocked_count = len(self.free_mask) - free_count

        if self.best_gap is not None:
            self.get_logger().info(
                f'front_pts: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: {len(self.gaps)} | '
                f'gap_angle: {gap_angle:.3f} | '
                f'corridor_angle: {corridor_angle:.3f} | '
                f'blend_w: {corridor_weight:.2f} | '
                f'front_min: {front_min:.2f} | '
                f'front_mean: {front_mean:.2f} | '
                f'left_mean: {left_mean:.2f} | '
                f'right_mean: {right_mean:.2f} | '
                f'speed_kmh: {self.current_speed_kmh:.2f} | '
                f'cmd_vel: {self.current_speed_kmh / self.max_speed_kmh:.2f} | '
                f'angle: {self.current_angle:.3f} | '
                f'cmd_dir: {self.current_angle / self.max_steering_angle:.2f}'
            )
        else:
            self.get_logger().info(
                f'front_pts: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: 0 | '
                f'corridor_angle: {corridor_angle:.3f} | '
                f'front_min: {front_min:.2f} | '
                f'front_mean: {front_mean:.2f} | '
                f'speed_kmh: {self.current_speed_kmh:.2f} | '
                f'cmd_vel: {self.current_speed_kmh / self.max_speed_kmh:.2f} | '
                f'angle: {self.current_angle:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = HybridGapFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()