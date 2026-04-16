import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from bolide_interfaces.msg import ForkSpeed, MultipleRange


class HybridGapFollowerNode(Node):
    def __init__(self):
        super().__init__('hybrid_gap_follower_node')

        # =========================
        # ROS parameters
        # =========================

        self.declare_parameter('max_speed_kmh', 28.0)
        self.declare_parameter('max_steering_angle', 0.28)

        self.declare_parameter('front_center_angle', math.pi)
        self.declare_parameter('front_half_width_deg', 90.0)
        self.declare_parameter('left_corridor_angle_deg', 240.0)
        self.declare_parameter('right_corridor_angle_deg', 120.0)
        self.declare_parameter('rear_center_angle', 0.0)

        self.declare_parameter('safe_distance', 1.5)
        self.declare_parameter('inflate_radius', 15)

        self.declare_parameter('corridor_mode_distance', 1.4)
        self.declare_parameter('very_tight_distance', 0.4)
        self.declare_parameter('corridor_sector_half_width', 8)
        self.declare_parameter('corridor_gain', 0.9)

        self.declare_parameter('front_min_sector_half_width', 30)
        self.declare_parameter('front_mean_sector_half_width', 25)
        self.declare_parameter('rear_sector_half_width', 15)

        self.declare_parameter('steering_smoothing', 0.0)
        self.declare_parameter('max_steer_step', 0.07)
        self.declare_parameter('aggressive_steering', 2.0)
        self.declare_parameter('momentum', 0.0)

        self.declare_parameter('open_speed_kmh', 1.0)
        self.declare_parameter('tight_speed_kmh', 0.5)
        self.declare_parameter('very_tight_speed_kmh', 0.2)
        self.declare_parameter('steering_speed_penalty', 3.0)
        self.declare_parameter('min_forward_speed_kmh', 0.5)

        self.declare_parameter('front_stop_distance', 0.4)
        self.declare_parameter('rear_block_distance', 0.15)
        self.declare_parameter('reverse_speed_kmh', -8.0)
        self.declare_parameter('reverse_steps_total', 50)
        self.declare_parameter('reverse_neutral_steps', 4)
        self.declare_parameter('reverse_pulse_steps', 8)
        self.declare_parameter('reverse_final_steps', 24)

        self.declare_parameter('gap_depth_min_clip', 0.5)
        self.declare_parameter('gap_depth_max_clip', 5.0)
        self.declare_parameter('gap_distance_weight', 0.0)
        self.declare_parameter('gap_center_weight', 0.5)
        self.declare_parameter('gap_edge_weight', 1.5)
        self.declare_parameter('gap_width_score_weight', 1.5)
        self.declare_parameter('gap_depth_score_weight', 1.8)

        self.declare_parameter('log_every_n', 10)

        self.declare_parameter('stationary_speed_threshold', 0.2)
        self.declare_parameter('stationary_cycles_trigger', 12)
        self.declare_parameter('rear_ir_block_distance', 0.15)

        # =========================
        # NEW: lidar close-wall recovery parameters
        # =========================
        self.declare_parameter('lidar_invalid_reuse_distance', 0.25)
        self.declare_parameter('lidar_invalid_max_hold_cycles', 6)
        self.declare_parameter('lidar_fallback_distance', 3.0)

        self.load_parameters()

        # =========================
        # ROS interfaces
        # =========================
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.fork_sub = self.create_subscription(
            ForkSpeed,
            '/raw_fork_data',
            self.fork_callback,
            10
        )

        self.rear_range_sub = self.create_subscription(
            MultipleRange,
            '/raw_rear_range_data',
            self.rear_range_callback,
            10
        )

        self.dir_pub = self.create_publisher(Float32, '/cmd_dir', 10)
        self.vel_pub = self.create_publisher(Float32, '/cmd_vel', 10)

        # =========================
        # State
        # =========================
        self.latest_scan = None
        self.latest_ranges = []
        self.latest_angles = []

        self.front_ranges = []
        self.front_angles = []

        self.free_mask = []
        self.gaps = []
        self.best_gap = None

        self.last_steering_angle = 0.0
        self.current_angle = 0.0
        self.current_speed_kmh = 0.0

        self.escape_mode = False

        self.reverse_sequence_stage = 0
        # 0 = idle
        # 1 = neutral_1
        # 2 = reverse_1
        # 3 = neutral_2
        # 4 = reverse_2

        self.sequence_counter = 0
        self.log_counter = 0

        # =========================
        # NEW: external sensor state
        # =========================
        self.latest_fork_speed = 0.0
        self.latest_rear_ir_right = float('inf')
        self.latest_rear_ir_left = float('inf')
        self.have_fork_data = False
        self.have_rear_range_data = False
        self.stationary_counter = 0

        # =========================
        # NEW: lidar history for invalid close-range recovery
        # =========================
        self.prev_ranges = []
        self.invalid_hold_counters = []

        self.get_logger().info('Hybrid gap follower node started, waiting for scan ...')

    def load_parameters(self):
        self.max_speed_kmh = self.get_parameter(
            'max_speed_kmh'
        ).get_parameter_value().double_value

        self.max_steering_angle = self.get_parameter(
            'max_steering_angle'
        ).get_parameter_value().double_value

        self.front_center_angle = self.get_parameter(
            'front_center_angle'
        ).get_parameter_value().double_value

        self.front_half_width = math.radians(
            self.get_parameter('front_half_width_deg').get_parameter_value().double_value
        )

        self.left_corridor_angle = math.radians(
            self.get_parameter('left_corridor_angle_deg').get_parameter_value().double_value
        )

        self.right_corridor_angle = math.radians(
            self.get_parameter('right_corridor_angle_deg').get_parameter_value().double_value
        )

        self.rear_center_angle = self.get_parameter(
            'rear_center_angle'
        ).get_parameter_value().double_value

        self.safe_distance = self.get_parameter(
            'safe_distance'
        ).get_parameter_value().double_value

        self.inflate_radius = self.get_parameter(
            'inflate_radius'
        ).get_parameter_value().integer_value

        self.corridor_mode_distance = self.get_parameter(
            'corridor_mode_distance'
        ).get_parameter_value().double_value

        self.very_tight_distance = self.get_parameter(
            'very_tight_distance'
        ).get_parameter_value().double_value

        self.corridor_sector_half_width = self.get_parameter(
            'corridor_sector_half_width'
        ).get_parameter_value().integer_value

        self.corridor_gain = self.get_parameter(
            'corridor_gain'
        ).get_parameter_value().double_value

        self.front_min_sector_half_width = self.get_parameter(
            'front_min_sector_half_width'
        ).get_parameter_value().integer_value

        self.front_mean_sector_half_width = self.get_parameter(
            'front_mean_sector_half_width'
        ).get_parameter_value().integer_value

        self.rear_sector_half_width = self.get_parameter(
            'rear_sector_half_width'
        ).get_parameter_value().integer_value

        self.steering_smoothing = self.get_parameter(
            'steering_smoothing'
        ).get_parameter_value().double_value

        self.max_steer_step = self.get_parameter(
            'max_steer_step'
        ).get_parameter_value().double_value

        self.aggressive_steering = self.get_parameter(
            'aggressive_steering'
        ).get_parameter_value().double_value

        self.momentum = self.get_parameter(
            'momentum'
        ).get_parameter_value().double_value

        self.open_speed_kmh = self.get_parameter(
            'open_speed_kmh'
        ).get_parameter_value().double_value

        self.tight_speed_kmh = self.get_parameter(
            'tight_speed_kmh'
        ).get_parameter_value().double_value

        self.very_tight_speed_kmh = self.get_parameter(
            'very_tight_speed_kmh'
        ).get_parameter_value().double_value

        self.steering_speed_penalty = self.get_parameter(
            'steering_speed_penalty'
        ).get_parameter_value().double_value

        self.min_forward_speed_kmh = self.get_parameter(
            'min_forward_speed_kmh'
        ).get_parameter_value().double_value

        self.front_stop_distance = self.get_parameter(
            'front_stop_distance'
        ).get_parameter_value().double_value

        self.rear_block_distance = self.get_parameter(
            'rear_block_distance'
        ).get_parameter_value().double_value

        self.reverse_speed_kmh = self.get_parameter(
            'reverse_speed_kmh'
        ).get_parameter_value().double_value

        self.reverse_steps_total = self.get_parameter(
            'reverse_steps_total'
        ).get_parameter_value().integer_value

        self.reverse_neutral_steps = self.get_parameter(
            'reverse_neutral_steps'
        ).get_parameter_value().integer_value

        self.reverse_pulse_steps = self.get_parameter(
            'reverse_pulse_steps'
        ).get_parameter_value().integer_value

        self.reverse_final_steps = self.get_parameter(
            'reverse_final_steps'
        ).get_parameter_value().integer_value

        self.gap_depth_min_clip = self.get_parameter(
            'gap_depth_min_clip'
        ).get_parameter_value().double_value

        self.gap_depth_max_clip = self.get_parameter(
            'gap_depth_max_clip'
        ).get_parameter_value().double_value

        self.gap_distance_weight = self.get_parameter(
            'gap_distance_weight'
        ).get_parameter_value().double_value

        self.gap_center_weight = self.get_parameter(
            'gap_center_weight'
        ).get_parameter_value().double_value

        self.gap_edge_weight = self.get_parameter(
            'gap_edge_weight'
        ).get_parameter_value().double_value

        self.gap_width_score_weight = self.get_parameter(
            'gap_width_score_weight'
        ).get_parameter_value().double_value

        self.gap_depth_score_weight = self.get_parameter(
            'gap_depth_score_weight'
        ).get_parameter_value().double_value

        self.log_every_n = self.get_parameter(
            'log_every_n'
        ).get_parameter_value().integer_value

        self.stationary_speed_threshold = self.get_parameter(
            'stationary_speed_threshold'
        ).get_parameter_value().double_value

        self.stationary_cycles_trigger = self.get_parameter(
            'stationary_cycles_trigger'
        ).get_parameter_value().integer_value

        self.rear_ir_block_distance = self.get_parameter(
            'rear_ir_block_distance'
        ).get_parameter_value().double_value

        # NEW
        self.lidar_invalid_reuse_distance = self.get_parameter(
            'lidar_invalid_reuse_distance'
        ).get_parameter_value().double_value

        self.lidar_invalid_max_hold_cycles = self.get_parameter(
            'lidar_invalid_max_hold_cycles'
        ).get_parameter_value().integer_value

        self.lidar_fallback_distance = self.get_parameter(
            'lidar_fallback_distance'
        ).get_parameter_value().double_value

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
        self.latest_scan = msg

        n = len(msg.ranges)

        if len(self.invalid_hold_counters) != n:
            self.invalid_hold_counters = [0] * n

        clean_ranges = []
        angles = []

        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment

            is_invalid = math.isinf(r) or math.isnan(r)

            if is_invalid:
                reused_previous = False

                if i < len(self.prev_ranges):
                    prev_r = self.prev_ranges[i]

                    # If the beam was very close in the previous scan,
                    # keep that value for a few cycles instead of turning
                    # it into "free space".
                    if (
                        prev_r <= self.lidar_invalid_reuse_distance and
                        self.invalid_hold_counters[i] < self.lidar_invalid_max_hold_cycles
                    ):
                        clean_r = prev_r
                        self.invalid_hold_counters[i] += 1
                        reused_previous = True

                if not reused_previous:
                    clean_r = self.clamp(
                        self.lidar_fallback_distance,
                        msg.range_min,
                        self.lidar_fallback_distance
                    )
                    self.invalid_hold_counters[i] = 0

            else:
                clean_r = self.clamp(r, msg.range_min, self.lidar_fallback_distance)
                self.invalid_hold_counters[i] = 0

            clean_ranges.append(clean_r)
            angles.append(angle)

        self.latest_ranges = clean_ranges
        self.latest_angles = angles

        # Save for next cycle
        self.prev_ranges = clean_ranges[:]

    # =========================
    # NEW: sensor callbacks
    # =========================
    def fork_callback(self, msg: ForkSpeed):
        self.latest_fork_speed = msg.speed
        self.have_fork_data = True

    def rear_range_callback(self, msg: MultipleRange):
        rr = msg.ir_rear_right.range
        rl = msg.ir_rear_left.range

        if math.isnan(rr):
            self.latest_rear_ir_right = float('inf')
        elif math.isinf(rr):
            self.latest_rear_ir_right = float('inf')
        else:
            min_r = msg.ir_rear_right.min_range
            max_r = msg.ir_rear_right.max_range
            self.latest_rear_ir_right = self.clamp(rr, min_r, max_r)

        if math.isnan(rl):
            self.latest_rear_ir_left = float('inf')
        elif math.isinf(rl):
            self.latest_rear_ir_left = float('inf')
        else:
            min_r = msg.ir_rear_left.min_range
            max_r = msg.ir_rear_left.max_range
            self.latest_rear_ir_left = self.clamp(rl, min_r, max_r)

        self.have_rear_range_data = True

    def angle_to_index(self, target_angle):
        if self.latest_scan is None or not self.latest_ranges:
            return 0

        target_angle = self.normalize_angle(target_angle)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        idx = int(round((target_angle - angle_min) / angle_increment))
        return int(self.clamp(idx, 0, len(self.latest_ranges) - 1))

    def get_sector_values_by_index(self, center_idx, half_width):
        if not self.latest_ranges:
            return []

        start = max(0, center_idx - half_width)
        end = min(len(self.latest_ranges) - 1, center_idx + half_width)
        return self.latest_ranges[start:end + 1]

    def get_sector_mean_by_index(self, center_idx, half_width):
        return self.mean(self.get_sector_values_by_index(center_idx, half_width))

    def get_sector_min_by_index(self, center_idx, half_width):
        vals = self.get_sector_values_by_index(center_idx, half_width)
        return min(vals) if vals else float('inf')

    def extract_front_view(self):
        front_points = []

        for angle, r in zip(self.latest_angles, self.latest_ranges):
            rel = self.normalize_angle(angle - self.front_center_angle)
            if abs(rel) <= self.front_half_width:
                front_points.append((rel, r))

        front_points.sort(key=lambda x: x[0])

        self.front_angles = [p[0] for p in front_points]
        self.front_ranges = [p[1] for p in front_points]

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

            score = (
                self.gap_distance_weight * distance_score
                + self.gap_center_weight * center_score
                + self.gap_edge_weight * edge_bonus
            )

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def select_best_gap(self):
        self.best_gap = None

        if not self.gaps or not self.front_ranges:
            return

        best_score = -1e9

        for start_idx, end_idx in self.gaps:
            width = end_idx - start_idx + 1
            gap_ranges = self.front_ranges[start_idx:end_idx + 1]
            mean_depth = self.clamp(
                self.mean(gap_ranges),
                self.gap_depth_min_clip,
                self.gap_depth_max_clip
            )

            target_idx = self.choose_gap_target(start_idx, end_idx)
            target_angle = self.front_angles[target_idx]

            normalized_target = target_angle / self.front_half_width
            steering_angle = (
                -self.aggressive_steering * normalized_target * self.max_steering_angle
                + self.momentum * self.last_steering_angle
            )
            steering_angle = self.clamp(
                steering_angle,
                -self.max_steering_angle,
                self.max_steering_angle
            )

            score = (
                self.gap_width_score_weight * width
                + self.gap_depth_score_weight * mean_depth
            )

            if score > best_score:
                best_score = score
                self.best_gap = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'target_idx': target_idx,
                    'target_angle': target_angle,
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

        diff = right_mean - left_mean
        corridor_angle = self.corridor_gain * diff
        corridor_angle = self.clamp(
            corridor_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        return corridor_angle, left_mean, right_mean

    def compute_front_clearance(self):
        front_idx = self.angle_to_index(self.front_center_angle)
        front_min = self.get_sector_min_by_index(front_idx, self.front_min_sector_half_width)
        front_mean = self.get_sector_mean_by_index(front_idx, self.front_mean_sector_half_width)
        return front_min, front_mean

    def compute_rear_clearance(self):
        rear_idx = self.angle_to_index(self.rear_center_angle)
        return self.get_sector_min_by_index(rear_idx, self.rear_sector_half_width)

    def blend_angles(self, gap_angle, corridor_angle, left_mean, right_mean):
        side_sum = left_mean + right_mean

        if side_sum >= self.corridor_mode_distance:
            w_corridor = 0.0
        elif side_sum <= self.very_tight_distance:
            w_corridor = 1.0
        else:
            span = max(1e-6, self.corridor_mode_distance - self.very_tight_distance)
            w_corridor = (self.corridor_mode_distance - side_sum) / span

        w_gap = 1.0 - w_corridor
        blended = w_gap * gap_angle + w_corridor * corridor_angle

        return self.clamp(
            blended,
            -self.max_steering_angle,
            self.max_steering_angle
        ), w_corridor, side_sum

    def smooth_steering(self, current_angle, desired_angle):
        blended = self.steering_smoothing * current_angle + (1.0 - self.steering_smoothing) * desired_angle
        delta = blended - current_angle
        delta = self.clamp(delta, -self.max_steer_step, self.max_steer_step)

        return self.clamp(
            current_angle + delta,
            -self.max_steering_angle,
            self.max_steering_angle
        )

    def compute_target_speed_kmh(self, front_clearance, steering_angle):
        if front_clearance < self.very_tight_distance:
            target_speed = self.very_tight_speed_kmh
        elif front_clearance < self.corridor_mode_distance:
            target_speed = self.tight_speed_kmh
        else:
            target_speed = self.open_speed_kmh

        target_speed -= self.steering_speed_penalty * abs(steering_angle)
        return max(self.min_forward_speed_kmh, target_speed)

    def choose_escape_steering(self):
        return 0.0

    def update_stationary_state(self):
        wants_forward_motion = (
            not self.escape_mode and
            self.current_speed_kmh > 0.1
        )

        effectively_stationary = (
            self.have_fork_data and
            abs(self.latest_fork_speed) < self.stationary_speed_threshold
        )

        if wants_forward_motion and effectively_stationary:
            self.stationary_counter += 1
        else:
            self.stationary_counter = 0

    def rear_has_room_for_reverse(self):
        if not self.have_rear_range_data:
            return False
        return (self.latest_rear_ir_right > self.rear_ir_block_distance) and (self.latest_rear_ir_left > self.rear_ir_block_distance)

    def emergency_escape_control(self, front_min):
        rear_right = self.latest_rear_ir_right if self.have_rear_range_data else float('nan')
        rear_left = self.latest_rear_ir_left if self.have_rear_range_data else float('nan')
        rear_has_room = self.rear_has_room_for_reverse()

        if self.reverse_sequence_stage in (2, 4) and not rear_has_room:
            self.escape_mode = False
            self.reverse_sequence_stage = 0
            self.sequence_counter = 0
            return True, 0.0, 0.0, (
                f'EMERGENCY REAR BLOCKED | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
            )

        if self.reverse_sequence_stage == 1:
            self.escape_mode = True
            self.sequence_counter += 1

            if self.sequence_counter >= self.reverse_neutral_steps:
                self.reverse_sequence_stage = 2
                self.sequence_counter = 0

            return True, 0.0, 0.0, (
                f'REVERSE SEQ N1 | step: {self.sequence_counter}/{self.reverse_neutral_steps} | '
                f'front_min: {front_min:.2f} | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
            )

        if self.reverse_sequence_stage == 2:
            self.escape_mode = True
            self.sequence_counter += 1

            if self.sequence_counter >= self.reverse_pulse_steps:
                self.reverse_sequence_stage = 3
                self.sequence_counter = 0

            return True, self.reverse_speed_kmh, self.choose_escape_steering(), (
                f'REVERSE SEQ R1 | step: {self.sequence_counter}/{self.reverse_pulse_steps} | '
                f'front_min: {front_min:.2f} | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
            )

        if self.reverse_sequence_stage == 3:
            self.escape_mode = True
            self.sequence_counter += 1

            if self.sequence_counter >= self.reverse_neutral_steps:
                self.reverse_sequence_stage = 4
                self.sequence_counter = 0

            return True, 0.0, 0.0, (
                f'REVERSE SEQ N2 | step: {self.sequence_counter}/{self.reverse_neutral_steps} | '
                f'front_min: {front_min:.2f} | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
            )

        if self.reverse_sequence_stage == 4:
            self.escape_mode = True
            self.sequence_counter += 1

            if self.sequence_counter >= self.reverse_final_steps:
                self.reverse_sequence_stage = 0
                self.sequence_counter = 0
                self.escape_mode = False

            return True, self.reverse_speed_kmh, self.choose_escape_steering(), (
                f'REVERSE SEQ R2 | step: {self.sequence_counter}/{self.reverse_final_steps} | '
                f'front_min: {front_min:.2f} | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
            )

        front_blocked = front_min < self.front_stop_distance
        stuck_too_long = self.stationary_counter >= self.stationary_cycles_trigger

        if front_blocked or stuck_too_long:
            if not rear_has_room:
                self.escape_mode = False
                reason = 'FRONT BLOCKED' if front_blocked else 'STATIONARY TOO LONG'
                return True, 0.0, 0.0, (
                    f'NO REVERSE ROOM | reason: {reason} | '
                    f'front_min: {front_min:.2f} | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
                )

            self.reverse_sequence_stage = 1
            self.sequence_counter = 0
            self.stationary_counter = 0
            self.escape_mode = True

            reason = 'FRONT BLOCKED' if front_blocked else 'STATIONARY TOO LONG'
            return True, 0.0, 0.0, (
                f'START REVERSE SEQUENCE | reason: {reason} | '
                f'front_min: {front_min:.2f} | rear_ir_right: {rear_right:.2f} | rear_ir_left: {rear_left:.2f}'
            )

        self.escape_mode = False
        return False, 0.0, 0.0, ''

    def publish_commands(self, speed_kmh, angle):
        angle = self.clamp(angle, -self.max_steering_angle, self.max_steering_angle)
        speed_kmh = self.clamp(speed_kmh, -self.max_speed_kmh, self.max_speed_kmh)

        dir_msg = Float32()
        vel_msg = Float32()

        dir_msg.data = float(self.clamp(angle / self.max_steering_angle, -1.0, 1.0))
        vel_msg.data = float(self.clamp(speed_kmh / self.max_speed_kmh, -1.0, 1.0))

        self.dir_pub.publish(dir_msg)
        self.vel_pub.publish(vel_msg)

    def should_log(self):
        self.log_counter += 1
        return self.log_counter % self.log_every_n == 0

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

        self.update_stationary_state()

        emergency_active, emergency_speed, emergency_angle, emergency_msg = self.emergency_escape_control(front_min)

        if emergency_active:
            self.current_speed_kmh = emergency_speed
            self.current_angle = emergency_angle
            self.publish_commands(self.current_speed_kmh, self.current_angle)

            if self.should_log():
                self.get_logger().warn(emergency_msg)
            return

        desired_angle, corridor_weight, side_sum = self.blend_angles(
            gap_angle,
            corridor_angle,
            left_mean,
            right_mean
        )

        self.current_angle = self.smooth_steering(self.current_angle, desired_angle)
        self.current_speed_kmh = self.compute_target_speed_kmh(front_mean, self.current_angle)

        self.publish_commands(self.current_speed_kmh, self.current_angle)

        if not self.should_log():
            return

        free_count = sum(self.free_mask)
        blocked_count = len(self.free_mask) - free_count
        rear_ir_right = self.latest_rear_ir_right if self.have_rear_range_data else float('nan')
        rear_ir_left = self.latest_rear_ir_left if self.have_rear_range_data else float('nan')

        if self.best_gap is not None:
            self.get_logger().info(
                f'front_pts: {len(self.front_ranges)} | '
                f'free: {free_count} | blocked: {blocked_count} | '
                f'gaps: {len(self.gaps)} | '
                f'gap_angle: {gap_angle:.3f} | '
                f'corridor_angle: {corridor_angle:.3f} | '
                f'side_sum: {side_sum:.2f} | '
                f'blend_w: {corridor_weight:.2f} | '
                f'front_min: {front_min:.2f} | '
                f'front_mean: {front_mean:.2f} | '
                f'left_mean: {left_mean:.2f} | '
                f'right_mean: {right_mean:.2f} | '
                f'fork_speed: {self.latest_fork_speed:.3f} | '
                f'stationary_count: {self.stationary_counter} | '
                f'rear_ir_right: {rear_ir_right:.2f} | '
                f'rear_ir_left: {rear_ir_left:.2f} | '
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
                f'side_sum: {side_sum:.2f} | '
                f'blend_w: {corridor_weight:.2f} | '
                f'front_min: {front_min:.2f} | '
                f'front_mean: {front_mean:.2f} | '
                f'left_mean: {left_mean:.2f} | '
                f'right_mean: {right_mean:.2f} | '
                f'fork_speed: {self.latest_fork_speed:.3f} | '
                f'stationary_count: {self.stationary_counter} | '
                f'rear_ir_right: {rear_ir_right:.2f} | '
                f'rear_ir_left: {rear_ir_left:.2f} | '
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