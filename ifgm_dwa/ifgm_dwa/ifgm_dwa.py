#!/usr/bin/env python3
"""
Hybrid Gap (direction) + DWA (vitesse) — Nœud ROS2 fusionné pour TT-02
=======================================================================

Architecture :
  /scan  ──►  [ce nœud]  ──►  /cmd_dir  (Float32, normalisé -1..1)
                          ──►  /cmd_vel  (Float32, normalisé -1..1)

Direction : algorithme de ton collègue (Hybrid Gap Follower)
  - Détection et sélection des gaps
  - Mode corridor (pondération gauche/droite)
  - Lissage du steering

Vitesse : ton algorithme DWA
  - Fenêtre dynamique (contrainte accélération/freinage)
  - Contrainte admissibilité (distance obstacle)
  - Optimisation par fonction objectif
  - phi interne = angle de braquage courant (plus besoin de /phi externe)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class HybridDWANode(Node):

    def __init__(self):
        super().__init__('hybrid_dwa_node')

        # =====================================================================
        # Paramètres — Direction (collègue)
        # =====================================================================
        self.declare_parameter('max_steering_angle',         0.28)
        self.declare_parameter('front_center_angle',         math.pi)
        self.declare_parameter('front_half_width_deg',       90.0)
        self.declare_parameter('left_corridor_angle_deg',    240.0)
        self.declare_parameter('right_corridor_angle_deg',   120.0)
        self.declare_parameter('safe_distance',              1.3)
        self.declare_parameter('inflate_radius',             10)
        self.declare_parameter('corridor_mode_distance',     1.8)
        self.declare_parameter('very_tight_distance',        0.0)
        self.declare_parameter('corridor_sector_half_width', 8)
        self.declare_parameter('corridor_gain',              0.9)
        self.declare_parameter('front_min_sector_half_width',10)
        self.declare_parameter('front_mean_sector_half_width',25)
        self.declare_parameter('steering_smoothing',         0.55)  # réactivité virage (était 0.75)
        self.declare_parameter('max_steer_step',             0.035)
        self.declare_parameter('aggressive_steering',        1.2)
        self.declare_parameter('momentum',                   0.5)
        self.declare_parameter('gap_depth_min_clip',         0.5)
        self.declare_parameter('gap_depth_max_clip',         3.0)
        self.declare_parameter('gap_distance_weight',        2.0)
        self.declare_parameter('gap_center_weight',          0.25)
        self.declare_parameter('gap_edge_weight',            0.10)
        self.declare_parameter('gap_width_score_weight',     1.0)
        self.declare_parameter('gap_depth_score_weight',     1.8)
        self.declare_parameter('log_every_n',                10)

        # =====================================================================
        # Paramètres — Vitesse (DWA)
        # =====================================================================
        self.declare_parameter('max_speed_kmh',       28.0)  # pour normalisation Float32
        self.declare_parameter('dt',                  0.1)
        self.declare_parameter('v_max',               0.25)  # m/s
        self.declare_parameter('v_min',               0.05)  # m/s
        self.declare_parameter('a_max',               0.8)
        self.declare_parameter('decel_max',           0.8)
        self.declare_parameter('w_dist',              0.7)
        self.declare_parameter('w_vel',               0.3)
        self.declare_parameter('w_steer',             0.5)   # pénalité braquage (NOUVEAU)
        self.declare_parameter('dmin_half_width_deg', 50.0)  # ±50° — capte les murs de virage
        self.declare_parameter('dmin_carrosserie',         0.30)  # m — filtre carrosserie dmin global (NOUVEAU)
        self.declare_parameter('dmin_carrosserie_urgence', 0.12)  # m — filtre carrosserie front_urgence
        self.declare_parameter('seuil_stop_frontal',  0.25)  # m — seuil déclenchement urgence
        self.declare_parameter('seuil_arriere',       0.35)  # m — distance min arrière pour reculer
        self.declare_parameter('v_recul_kmh',        -1.5)   # km/h — vitesse de recul
        self.declare_parameter('nb_cycles_recul',     30)    # cycles de marche arrière

        self._load_parameters()

        # =====================================================================
        # État interne — Direction
        # =====================================================================
        self.latest_scan   = None
        self.latest_ranges = []
        self.latest_angles = []
        self.front_ranges  = []
        self.front_angles  = []
        self.free_mask     = []
        self.gaps          = []
        self.best_gap      = None
        self.last_steering_angle = 0.0
        self.current_angle       = 0.0
        self.log_counter         = 0

        # =====================================================================
        # État interne — Vitesse DWA
        # =====================================================================
        self.scan_array  = None
        self.v_current   = self.v_min
        self.cycles_recul = 0   # compteur de cycles de marche arrière en cours

        # =====================================================================
        # Interfaces ROS2
        # =====================================================================
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)

        self.dir_pub = self.create_publisher(Float32, '/cmd_dir', 10)
        self.vel_pub = self.create_publisher(Float32, '/cmd_vel', 10)

        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info('=' * 55)
        self.get_logger().info('  Hybrid Gap (dir) + DWA (vel) — TT-02 prêt')
        self.get_logger().info('=' * 55)

    # =========================================================================
    # Chargement des paramètres
    # =========================================================================

    def _load_parameters(self):
        g = self.get_parameter

        # Direction
        self.max_steering_angle          = g('max_steering_angle').value
        self.front_center_angle          = g('front_center_angle').value
        self.front_half_width            = math.radians(g('front_half_width_deg').value)
        self.left_corridor_angle         = math.radians(g('left_corridor_angle_deg').value)
        self.right_corridor_angle        = math.radians(g('right_corridor_angle_deg').value)
        self.safe_distance               = g('safe_distance').value
        self.inflate_radius              = g('inflate_radius').value
        self.corridor_mode_distance      = g('corridor_mode_distance').value
        self.very_tight_distance         = g('very_tight_distance').value
        self.corridor_sector_half_width  = g('corridor_sector_half_width').value
        self.corridor_gain               = g('corridor_gain').value
        self.front_min_sector_half_width = g('front_min_sector_half_width').value
        self.front_mean_sector_half_width= g('front_mean_sector_half_width').value
        self.steering_smoothing          = g('steering_smoothing').value
        self.max_steer_step              = g('max_steer_step').value
        self.aggressive_steering         = g('aggressive_steering').value
        self.momentum                    = g('momentum').value
        self.gap_depth_min_clip          = g('gap_depth_min_clip').value
        self.gap_depth_max_clip          = g('gap_depth_max_clip').value
        self.gap_distance_weight         = g('gap_distance_weight').value
        self.gap_center_weight           = g('gap_center_weight').value
        self.gap_edge_weight             = g('gap_edge_weight').value
        self.gap_width_score_weight      = g('gap_width_score_weight').value
        self.gap_depth_score_weight      = g('gap_depth_score_weight').value
        self.log_every_n                 = g('log_every_n').value

        # Vitesse
        self.max_speed_kmh      = g('max_speed_kmh').value
        self.dt                 = g('dt').value
        self.v_max              = g('v_max').value
        self.v_min              = g('v_min').value
        self.a_max              = g('a_max').value
        self.decel_max          = g('decel_max').value
        self.w_dist             = g('w_dist').value
        self.w_vel              = g('w_vel').value
        self.w_steer            = g('w_steer').value
        self.dmin_half_width_deg  = g('dmin_half_width_deg').value
        self.dmin_carrosserie          = g('dmin_carrosserie').value
        self.dmin_carrosserie_urgence  = g('dmin_carrosserie_urgence').value
        self.seuil_stop_frontal        = g('seuil_stop_frontal').value
        self.seuil_arriere             = g('seuil_arriere').value
        self.v_recul_kmh               = g('v_recul_kmh').value
        self.nb_cycles_recul           = g('nb_cycles_recul').value

    # =========================================================================
    # Utilitaires communs
    # =========================================================================

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _normalize_angle(angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    # =========================================================================
    # Callback LiDAR
    # =========================================================================

    def _scan_callback(self, msg: LaserScan):
        # -- Tableau numpy pour le DWA --
        arr = np.array(msg.ranges)
        arr[np.isinf(arr)] = msg.range_max
        arr[np.isnan(arr)] = msg.range_max
        self.scan_array = arr

        # -- Liste nettoyée pour la direction --
        self.latest_scan = msg
        clean_ranges, angles = [], []
        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if math.isinf(r) or math.isnan(r):
                clean_r = msg.range_max
            else:
                clean_r = self._clamp(r, msg.range_min, msg.range_max)
            clean_ranges.append(clean_r)
            angles.append(angle)
        self.latest_ranges = clean_ranges
        self.latest_angles = angles

    # =========================================================================
    # Direction — Hybrid Gap Follower (code collègue, sans vitesse)
    # =========================================================================

    def _angle_to_index(self, target_angle):
        if self.latest_scan is None:
            return 0
        target_angle = self._normalize_angle(target_angle)
        idx = int(round(
            (target_angle - self.latest_scan.angle_min)
            / self.latest_scan.angle_increment
        ))
        return int(self._clamp(idx, 0, len(self.latest_ranges) - 1))

    def _sector_values(self, center_idx, half_width):
        s = max(0, center_idx - half_width)
        e = min(len(self.latest_ranges) - 1, center_idx + half_width)
        return self.latest_ranges[s:e + 1]

    def _sector_mean(self, center_idx, half_width):
        return self._mean(self._sector_values(center_idx, half_width))

    def _sector_min(self, center_idx, half_width):
        vals = self._sector_values(center_idx, half_width)
        return min(vals) if vals else float('inf')

    def _extract_front_view(self):
        pts = []
        for angle, r in zip(self.latest_angles, self.latest_ranges):
            rel = self._normalize_angle(angle - self.front_center_angle)
            if abs(rel) <= self.front_half_width:
                pts.append((rel, r))
        pts.sort(key=lambda x: x[0])
        self.front_angles = [p[0] for p in pts]
        self.front_ranges = [p[1] for p in pts]

    def _build_free_mask(self):
        raw = [r > self.safe_distance for r in self.front_ranges]
        inflated = raw[:]
        for i, is_free in enumerate(raw):
            if not is_free:
                for j in range(
                    max(0, i - self.inflate_radius),
                    min(len(inflated) - 1, i + self.inflate_radius) + 1
                ):
                    inflated[j] = False
        self.free_mask = inflated

    def _find_gaps(self):
        self.gaps = []
        start = None
        for i, is_free in enumerate(self.free_mask):
            if is_free and start is None:
                start = i
            elif not is_free and start is not None:
                self.gaps.append((start, i - 1))
                start = None
        if start is not None:
            self.gaps.append((start, len(self.free_mask) - 1))

    def _choose_gap_target(self, start_idx, end_idx):
        best_idx, best_score = start_idx, -1e9
        gap_center = 0.5 * (start_idx + end_idx)
        for i in range(start_idx, end_idx + 1):
            score = (
                self.gap_distance_weight * self.front_ranges[i]
                + self.gap_center_weight * (-abs(i - gap_center))
                + self.gap_edge_weight   * min(i - start_idx, end_idx - i)
            )
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _select_best_gap(self):
        self.best_gap = None
        if not self.gaps or not self.front_ranges:
            return
        best_score = -1e9
        for start_idx, end_idx in self.gaps:
            width      = end_idx - start_idx + 1
            mean_depth = self._clamp(
                self._mean(self.front_ranges[start_idx:end_idx + 1]),
                self.gap_depth_min_clip,
                self.gap_depth_max_clip
            )
            target_idx   = self._choose_gap_target(start_idx, end_idx)
            target_angle = self.front_angles[target_idx]
            normalized   = target_angle / self.front_half_width
            steering     = self._clamp(
                -self.aggressive_steering * normalized * self.max_steering_angle
                + self.momentum * self.last_steering_angle,
                -self.max_steering_angle, self.max_steering_angle
            )
            score = (
                self.gap_width_score_weight * width
                + self.gap_depth_score_weight * mean_depth
            )
            if score > best_score:
                best_score = score
                self.best_gap = {
                    'target_angle':   target_angle,
                    'steering_angle': steering,
                    'width':          width,
                    'mean_depth':     mean_depth,
                }
        if self.best_gap is not None:
            self.last_steering_angle = self.best_gap['steering_angle']

    def _compute_corridor_angle(self):
        left_mean  = self._sector_mean(self._angle_to_index(self.left_corridor_angle),
                                       self.corridor_sector_half_width)
        right_mean = self._sector_mean(self._angle_to_index(self.right_corridor_angle),
                                       self.corridor_sector_half_width)
        corridor   = self._clamp(
            self.corridor_gain * (right_mean - left_mean),
            -self.max_steering_angle, self.max_steering_angle
        )
        return corridor, left_mean, right_mean

    def _compute_front_clearance(self):
        front_idx  = self._angle_to_index(self.front_center_angle)
        front_min  = self._sector_min(front_idx, self.front_min_sector_half_width)
        front_mean = self._sector_mean(front_idx, self.front_mean_sector_half_width)
        return front_min, front_mean

    def _blend_angles(self, gap_angle, corridor_angle, front_clearance):
        if front_clearance >= self.corridor_mode_distance:
            w_corridor = 0.0
        elif front_clearance <= self.very_tight_distance:
            w_corridor = 1.0
        else:
            span = self.corridor_mode_distance - self.very_tight_distance
            w_corridor = (self.corridor_mode_distance - front_clearance) / span
        blended = (1.0 - w_corridor) * gap_angle + w_corridor * corridor_angle
        return self._clamp(blended, -self.max_steering_angle, self.max_steering_angle), w_corridor

    def _smooth_steering(self, current, desired):
        blended = self.steering_smoothing * current + (1.0 - self.steering_smoothing) * desired
        delta   = self._clamp(blended - current, -self.max_steer_step, self.max_steer_step)
        return self._clamp(current + delta, -self.max_steering_angle, self.max_steering_angle)

    def _compute_direction(self):
        """Retourne l'angle de braquage (rad) issu du Hybrid Gap."""
        self._extract_front_view()
        self._build_free_mask()
        self._find_gaps()
        self._select_best_gap()

        gap_angle                    = 0.0 if self.best_gap is None else self.best_gap['steering_angle']
        corridor_angle, _, _         = self._compute_corridor_angle()
        _, front_mean                = self._compute_front_clearance()
        desired_angle, corridor_weight = self._blend_angles(gap_angle, corridor_angle, front_mean)
        self.current_angle           = self._smooth_steering(self.current_angle, desired_angle)

        return self.current_angle, front_mean, corridor_weight

    # =========================================================================
    # Vitesse — DWA
    # =========================================================================

    def _compute_dmin(self):
        """
        Distance minimale sur le secteur AVANT uniquement (±dmin_half_width_deg).
        Les valeurs < dmin_carrosserie sont ignorées (retours sur la carrosserie).
        Si tous les points sont filtrés, retourne range_max (pas d'obstacle devant).
        """
        if self.scan_array is None:
            return 10.0
        n        = len(self.scan_array)
        deg_step = 360.0 / n
        hw       = int(round(self.dmin_half_width_deg / deg_step))
        center   = n // 2
        i_start  = max(0, center - hw)
        i_end    = min(n - 1, center + hw)
        secteur  = self.scan_array[i_start:i_end + 1]
        valides  = secteur[secteur >= self.dmin_carrosserie]
        return float(np.min(valides)) if len(valides) > 0 else 10.0

    def _dynamic_window(self):
        v_lo = max(self.v_min, self.v_current - self.decel_max * self.dt)
        v_hi = min(self.v_max, self.v_current + self.a_max      * self.dt)
        return v_lo, v_hi

    def _admissible_velocity(self, dmin):
        return min(self.v_max, float(np.sqrt(2.0 * dmin * self.decel_max)))

    def _objective(self, v, dmin):
        """
        Fonction objectif corrigée :
          + w_dist  * dmin          → favorise l'espace libre devant
          + w_vel   * v             → favorise la vitesse (ligne droite)
          - w_steer * |steer|       → pénalise le braquage (ralentit en virage)
        Le braquage courant (self.current_angle) est connu car la direction
        est calculée avant la vitesse dans _control_loop.
        """
        steer_penalty = self.w_steer * abs(self.current_angle)
        return self.w_dist * dmin + self.w_vel * v - steer_penalty

    def _compute_velocity(self):
        """Retourne la vitesse optimale (m/s) via DWA."""
        dmin       = self._compute_dmin()
        v_lo, v_hi = self._dynamic_window()
        v_safe     = self._admissible_velocity(dmin)

        best_v, best_score = self.v_min, -1e9
        for v in np.linspace(v_lo, v_hi, 6):
            v     = min(v, v_safe)
            score = self._objective(v, dmin)
            if score > best_score:
                best_score = score
                best_v     = v

        self.v_current = best_v
        return best_v

    # =========================================================================
    # Publication
    # =========================================================================

    def _publish(self, speed_ms, angle_rad):
        """
        Publie en Float32 normalisé [-1, 1] comme attendu par le hardware.
          /cmd_dir : angle / max_steering_angle
          /cmd_vel : speed_ms → km/h / max_speed_kmh
        """
        speed_kmh = speed_ms * 3.6

        dir_msg      = Float32()
        vel_msg      = Float32()
        dir_msg.data = float(self._clamp(angle_rad / self.max_steering_angle, -1.0, 1.0))
        vel_msg.data = float(self._clamp(speed_kmh  / self.max_speed_kmh,     -1.0, 1.0))

        self.dir_pub.publish(dir_msg)
        self.vel_pub.publish(vel_msg)

    # =========================================================================
    # Boucle principale
    # =========================================================================

    def _control_loop(self):
        if self.scan_array is None or not self.latest_ranges:
            self.get_logger().warn('En attente du LiDAR…', throttle_duration_sec=2.0)
            return

        # -- Direction (Hybrid Gap) -- calculée en premier pour avoir current_angle
        angle_rad, front_mean, corridor_weight = self._compute_direction()

        # -- Stop d'urgence frontal + marche arrière --
        n             = len(self.scan_array)
        center        = n // 2
        hw_urg        = int(round(10.0 / (360.0 / n)))
        i_s           = max(0, center - hw_urg)
        i_e           = min(n - 1, center + hw_urg)
        sect_urg      = self.scan_array[i_s:i_e + 1]
        val_urg       = sect_urg[sect_urg >= self.dmin_carrosserie_urgence]
        front_urgence = float(np.min(val_urg)) if len(val_urg) > 0 else 10.0

        # Distance arrière (index 0 = arrière dans repère CW après conversion)
        sect_arr  = np.concatenate([self.scan_array[n-10:], self.scan_array[:11]])
        val_arr   = sect_arr[sect_arr >= self.dmin_carrosserie_urgence]
        dist_arr  = float(np.min(val_arr)) if len(val_arr) > 0 else 10.0

        # Côté le plus dégagé (pour choisir le sens du braquage en recul)
        left_idx   = self._angle_to_index(self.left_corridor_angle)
        right_idx  = self._angle_to_index(self.right_corridor_angle)
        left_mean  = self._sector_mean(left_idx,  self.corridor_sector_half_width)
        right_mean = self._sector_mean(right_idx, self.corridor_sector_half_width)

        speed_ms = 0.0

        if self.cycles_recul > 0:
            self.cycles_recul -= 1
            if dist_arr < self.seuil_arriere:
                self.cycles_recul = 0
                speed_ms  = 0.0
                angle_rad = 0.0
                self.get_logger().warn('Bloqué avant ET arrière — arrêt total',
                                       throttle_duration_sec=0.5)
            else:
                v_recul_ms = self.v_recul_kmh / 3.6
                speed_ms   = v_recul_ms
                angle_rad  = (-self.max_steering_angle if right_mean > left_mean
                              else self.max_steering_angle)
                self.get_logger().info(
                    f'[RECUL] {self.cycles_recul} cycles  '
                    f'arrière={dist_arr:.2f}m  steer={math.degrees(angle_rad):+.0f}°',
                    throttle_duration_sec=0.3)

        elif front_urgence < self.seuil_stop_frontal:
            if dist_arr > self.seuil_arriere:
                self.cycles_recul = int(self.nb_cycles_recul)
                speed_ms   = self.v_recul_kmh / 3.6
                angle_rad  = (-self.max_steering_angle if right_mean > left_mean
                              else self.max_steering_angle)
                self.v_current = self.v_min
                side = 'droite' if right_mean > left_mean else 'gauche'
                self.get_logger().warn(
                    f'[URGENCE→RECUL] mur à {front_urgence:.2f}m — recul {side} '
                    f'(L={left_mean:.2f}m R={right_mean:.2f}m)')
            else:
                speed_ms  = 0.0
                angle_rad = (self.max_steering_angle if right_mean > left_mean
                             else -self.max_steering_angle)
                self.get_logger().warn(
                    f'Bloqué partout (front={front_urgence:.2f}m arr={dist_arr:.2f}m)',
                    throttle_duration_sec=0.5)
        else:
            speed_ms = self._compute_velocity()

        # -- Publication --
        self._publish(speed_ms, angle_rad)

        # -- Logging --
        self.log_counter += 1
        if self.log_counter % self.log_every_n == 0:
            dmin = self._compute_dmin()
            self.get_logger().info(
                f'[DIR] angle={math.degrees(angle_rad):+.1f}°  '
                f'cmd_dir={angle_rad / self.max_steering_angle:+.2f}  '
                f'corridor_w={corridor_weight:.2f}  '
                f'front_mean={front_mean:.2f}m  '
                f'front_urgence={front_urgence:.2f}m  '
                f'gaps={len(self.gaps)} | '
                f'[VEL] v={speed_ms:.3f}m/s  '
                f'dmin_avant={dmin:.2f}m  '
                f'steer_penalty={self.w_steer * abs(angle_rad):.3f}  '
                f'cmd_vel={speed_ms * 3.6 / self.max_speed_kmh:.3f}'
            )


# =============================================================================
# Point d'entrée
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = HybridDWANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Arrêt — STOP envoyé')
        stop = Float32()
        stop.data = 0.0
        node.vel_pub.publish(stop)
        node.dir_pub.publish(stop)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
