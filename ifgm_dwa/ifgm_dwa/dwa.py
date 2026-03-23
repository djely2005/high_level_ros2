"""
DWA Node - Dynamic Window Approach
Véhicule  : Tamiya TT-02 (châssis Ackermann)
Capteur   : LiDAR (sensor_msgs/LaserScan)
Plateforme: ROS2, Raspberry Pi 3

Basé sur : Hossain et al. (2022) - "Local path planning for autonomous mobile
robots by integrating modified dynamic-window approach and improved follow
the gap method", Journal of Field Robotics, 39, 371-386.


MODÈLE CINÉMATIQUE ACKERMANN — Eq. (1)(2)(3) de l'article :

    ẋ = V · cos(θ)
    ẏ = V · sin(θ)
    θ̇ = (V / L) · tan(δ)

  V = vitesse linéaire [m/s]
  L = empattement TT-02 [m]
  δ = angle de braquage [rad]  — limité à ±15° (valeur physique mesurée)
  ω = (V / L) · tan(δ)        — taux de lacet déduit


INTÉGRATION IFGM-DWA — Eq. (8) de l'article :

  Le terme heading(v,ω) de la fonction objectif utilise l'angle guide Φ_final
  calculé par l'IFGM (Eq. 7) à la place de l'angle direct vers le goal.
  Cela permet au DWA de suivre le gap le plus sûr identifié par l'IFGM.

  Quand l'IFGM n'est pas disponible → heading direct vers le goal (mode dégradé).


INTERFACE AVEC LE NŒUD IFGM  :

  Le nœud IFGM publie sur /cmd_dir un Float32 normalisé ∈ [-1, 1] :
      cmd_dir = steering_angle / max_steering_angle
  avec max_steering_angle = math.radians(15°) = limite physique TT-02 mesurée.

  Convention : positif = gauche, négatif = droite 

  Ce nœud DWA reçoit cmd_dir et reconstruit l'angle absolu cible dans le
  repère global pour le calcul de heading :
      guide_angle_body [rad] = cmd_dir × math.radians(15.0)
      guide_angle_world [rad] = theta_robot + guide_angle_body


TOPICS EN ENTRÉE :
  /scan           (sensor_msgs/LaserScan)       → données LiDAR
  /odom           (nav_msgs/Odometry)            → position et vitesse actuelle
  /goal_position  (geometry_msgs/PointStamped)   → goal (planificateur global)
  /cmd_dir        (std_msgs/Float32)             → angle guide normalisé de l'IFGM

TOPICS EN SORTIE :
  /cmd_vel        (geometry_msgs/Twist)          → commandes robot
                      linear.x  = vitesse V [m/s]
                      angular.z = taux de lacet ω [rad/s]
  /dwa/d_min      (std_msgs/Float32)             → distance min obstacle → IFGM
  /dwa/status     (std_msgs/String)              → état courant (debug)

"""

import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32, String
from tf_transformations import euler_from_quaternion



# PARAMÈTRES

class DWAConfig:
    """
    Paramètres du DWA pour le TT-02.
    Les valeurs marquées [MESURER] doivent être calibrées sur le vrai robot.
    """

    #  Temps d'échantillonnage 
    dt: float = 0.1                        # [s]

    #  Géométrie TT-02 
    wheelbase: float   = 0.257             # Empattement L [m] (~257 mm)  
    robot_radius: float = 0.18             # Rayon d'encombrement [m]     

    #  Braquage — valeur physique mesurée sur le TT-02 
    max_steering_angle: float = math.radians(15.0)  # ±15° physique mesuré
    delta_max: float  =  math.radians(15.0)         # [rad]
    delta_min: float  = -math.radians(15.0)         # [rad]
    delta_dot_max: float = math.radians(60.0)       # [rad/s] réactivité servo 

    #  Vitesse linéaire 
    v_max: float     =  1.5                # [m/s]   
    v_min: float     =  0.0               # Pas de marche arrière en course
    v_dot_max: float =  1.0               # [m/s²]  
    v_dot_min: float = -2.0               # [m/s²]  

    #  Résolution d'échantillonnage de l'espace (V, δ) 
    v_resolution: float     = 0.05                 # [m/s]
    delta_resolution: float = math.radians(1.0)    # [rad] (1° par pas)

    #  Horizon de prédiction 
    predict_steps: int = 10                # Nombre de pas simulés (article p.378)

    #  Distances seuils 
    d_safe_threshold: float   = 0.40       # [m] seuil passage mode sécurité
    goal_limit_distance: float = 0.35      # [m] zone d'arrivée (article p.378)
    collision_threshold: float = 0.25      # [m] trajectoire inadmissible

    #  Coefficients fonction objectif G(V,δ) — Eq. (8) 
    # G = σ(α·heading + β·dist + γ·vel)
    #
    # heading : alignement du cap prédit avec l'angle guide Φ de l'IFGM
    #           → mesure si la trajectoire (V,δ) mène vers le gap choisi
    # dist    : distance aux obstacles (sécurité)
    # vel     : vitesse (performance en course)
    alpha: float = 0.20
    beta: float  = 0.45
    gamma: float = 0.35


# ÉTATS OPÉRATIONNELS — Figure 8 de l'article

class RobotState:
    FREE_SPACE    = "free_space"     # Espace libre → vitesse max     Eq. (17-19)
    HEADING_FIRST = "heading_first"  # Obstacle détecté → cap ajusté  Eq. (20-22)
    SAFETY_FIRST  = "safety_first"   # Obstacle très proche → stop    Eq. (23-25)


# NŒUD DWA

class DWANode(Node):

    def __init__(self):
        super().__init__('dwa_node')
        self.config = DWAConfig()

        #  État cinématique 
        self.x             = 0.0    # Position x [m]
        self.y             = 0.0    # Position y [m]
        self.theta         = 0.0    # Cap courant [rad] dans le repère global
        self.v_current     = 0.0    # Vitesse linéaire courante [m/s]
        self.delta_current = 0.0    # Angle de braquage courant [rad]

        #  Données mission 
        self.goal_x        = None
        self.goal_y        = None
        self.goal_reached  = False

        #  Données LiDAR 
        self.scan_ranges          = []
        self.scan_angle_min       = 0.0
        self.scan_angle_increment = 0.0

        #  Interface IFGM 
        # cmd_dir_raw  : valeur brute reçue ∈ [-1, 1]  (Float32, topic /cmd_dir)
        # guide_angle  : angle cible dans le repère du robot [rad]
        #                = cmd_dir_raw × max_steering_angle
        #                utilisé pour reconstruire l'angle global dans objective_function
        self.cmd_dir_raw = None     # None si IFGM pas encore connecté
        self.d_min       = float('inf')  # [m]

        #  État courant 
        self.current_state = RobotState.FREE_SPACE

        #  Subscribers 
        self.create_subscription(
            LaserScan,    '/scan',          self.scan_callback,        10)
        self.create_subscription(
            Odometry,     '/odom',          self.odom_callback,        10)
        self.create_subscription(
            PointStamped, '/goal_position', self.goal_callback,        10)
        self.create_subscription(
            Float32,      '/cmd_dir',       self.guide_angle_callback, 10)

        #  Publishers 
        self.cmd_vel_pub = self.create_publisher(Twist,   '/cmd_vel',    10)
        self.d_min_pub   = self.create_publisher(Float32, '/dwa/d_min',  10)
        self.status_pub  = self.create_publisher(String,  '/dwa/status', 10)

        #  Timer principal 
        self.create_timer(self.config.dt, self.compute_velocity_command)

        self.get_logger().info(
            f"DWA Node démarré | TT-02 Ackermann | "
            f"L={self.config.wheelbase}m | "
            f"v_max={self.config.v_max}m/s | "
            f"δ_max={math.degrees(self.config.delta_max):.0f}°"
        )

    # CALLBACKS

    def scan_callback(self, msg: LaserScan):
        """
        Réception LiDAR.
        Calcule d_min = distance libre minimale — Eq. (10) de l'article.
        Publie d_min sur /dwa/d_min pour que l'IFGM puisse l'utiliser.
        """
        self.scan_ranges          = list(msg.ranges)
        self.scan_angle_min       = msg.angle_min
        self.scan_angle_increment = msg.angle_increment

        valid = [
            r for r in self.scan_ranges
            if not math.isnan(r) and not math.isinf(r) and r > 0.02
        ]
        self.d_min = max(min(valid) - self.config.robot_radius, 0.0) if valid else float('inf')

        out = Float32()
        out.data = float(self.d_min)
        self.d_min_pub.publish(out)

    def odom_callback(self, msg: Odometry):
        """
        Réception odométrie.
        Met à jour (x, y, θ, v).
        Estime δ courant depuis ω mesuré : δ = atan(ω·L / V) — Eq. (3).
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.v_current = msg.twist.twist.linear.x

        omega_meas = msg.twist.twist.angular.z
        if abs(self.v_current) > 0.05:
            delta_est = math.atan(
                omega_meas * self.config.wheelbase / self.v_current
            )
            self.delta_current = max(
                self.config.delta_min,
                min(self.config.delta_max, delta_est)
            )

    def goal_callback(self, msg: PointStamped):
        """Réception du goal depuis le planificateur global."""
        self.goal_x      = msg.point.x
        self.goal_y      = msg.point.y
        self.goal_reached = False
        self.get_logger().info(
            f"Nouveau goal → ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    def guide_angle_callback(self, msg: Float32):
        """
        Réception du cmd_dir normalisé ∈ [-1, 1] depuis le nœud IFGM.

        Convention IFGM (collègue) :
            cmd_dir = steering_angle / max_steering_angle
            max_steering_angle = math.radians(15.0)  [limite physique TT-02]
            positif = gauche, négatif = droite

        On stocke la valeur brute. La conversion en angle global se fait
        dans objective_function() où self.theta est disponible et à jour.
        """
        self.cmd_dir_raw = msg.data

    # MODÈLE CINÉMATIQUE ACKERMANN — Eq. (1)(2)(3)


    def ackermann_step(
        self,
        px: float, py: float, ptheta: float,
        v: float, delta: float
    ) -> tuple:
        """
        Propage l'état (x, y, θ) d'un pas dt selon le modèle Ackermann.

            ẋ = V·cos(θ)              Eq. (1)
            ẏ = V·sin(θ)              Eq. (2)
            θ̇ = (V/L)·tan(δ)         Eq. (3)
        """
        dt = self.config.dt
        L  = self.config.wheelbase

        px_new     = px     + v * math.cos(ptheta) * dt
        py_new     = py     + v * math.sin(ptheta) * dt
        ptheta_new = ptheta + (v / L) * math.tan(delta) * dt

        return px_new, py_new, ptheta_new

    def delta_to_omega(self, v: float, delta: float) -> float:
        """
        Convertit (V, δ) → ω pour publication sur /cmd_vel.angular.z.
        ω = (V / L) · tan(δ)  — Eq. (3)
        """
        if abs(v) < 1e-6:
            return 0.0
        return (v / self.config.wheelbase) * math.tan(delta)

    # FENÊTRE DE VITESSES ATTEIGNABLES V_r — Eq. (15)

    def compute_reachable_window(self) -> tuple:
        """
        Calcule l'intervalle (V, δ) atteignable depuis l'état courant
        en un pas dt, sous contraintes d'accélération moteur et servo.

            V_r = {v | v ∈ [v_c + v̇_min·dt,  v_c + v̇_max·dt]}   Eq. (15)
            Δ_r = {δ | δ ∈ [δ_c - δ̇_max·dt,  δ_c + δ̇_max·dt]}

        Retourne (v_min_r, v_max_r, delta_min_r, delta_max_r).
        """
        cfg = self.config
        dt  = cfg.dt

        v_min_r = max(cfg.v_min,     self.v_current     + cfg.v_dot_min    * dt)
        v_max_r = min(cfg.v_max,     self.v_current     + cfg.v_dot_max    * dt)
        d_min_r = max(cfg.delta_min, self.delta_current - cfg.delta_dot_max * dt)
        d_max_r = min(cfg.delta_max, self.delta_current + cfg.delta_dot_max * dt)

        return v_min_r, v_max_r, d_min_r, d_max_r

    # ADMISSIBILITÉ V_a — Eq. (16)

    def is_admissible(self, v: float, delta: float) -> bool:
        """
        Vérifie que (V, δ) est admissible : le véhicule peut freiner
        avant toute collision sur l'horizon de prédiction — Eq. (16) :

            V ≤ √(2 · d_min_trajectory · |v̇_min|)

        Retourne True si la trajectoire est sûre.
        """
        dist = self._predict_min_distance(v, delta)
        if dist is None:
            return False
        v_brake = math.sqrt(max(0.0, 2.0 * dist * abs(self.config.v_dot_min)))
        return abs(v) <= v_brake

    def _predict_min_distance(self, v: float, delta: float) -> float | None:
        """
        Simule la trajectoire Ackermann (V, δ) sur predict_steps pas.
        Retourne la distance minimale aux obstacles, ou None si collision.

        Cinématique utilisée : ackermann_step() — Eq. (1-3).
        """
        if not self.scan_ranges:
            return self.d_min

        px, py, ptheta = self.x, self.y, self.theta
        min_dist = float('inf')

        for _ in range(self.config.predict_steps):
            px, py, ptheta = self.ackermann_step(px, py, ptheta, v, delta)
            dist = self._distance_to_obstacles_at(px, py)
            if dist < self.config.collision_threshold:
                return None
            min_dist = min(min_dist, dist)

        return min_dist

    def _distance_to_obstacles_at(self, px: float, py: float) -> float:
        """
        Distance minimale aux obstacles depuis le point prédit (px, py).
        Projette les mesures LiDAR en coordonnées globales — Eq. (9)(10).
        """
        if not self.scan_ranges:
            return float('inf')

        min_dist = float('inf')
        for i, r in enumerate(self.scan_ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0.02:
                continue
            angle = self.scan_angle_min + i * self.scan_angle_increment + self.theta
            ox = self.x + r * math.cos(angle)
            oy = self.y + r * math.sin(angle)
            # Eq. (9) : d = √((px-ox)² + (py-oy)²) - r_robot
            dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2) - self.config.robot_radius
            min_dist = min(min_dist, dist)

        return max(min_dist, 0.0)

    # FONCTION OBJECTIF G(V, δ) — Eq. (8)

    def _compute_heading_angle(self) -> float:
        """
        Calcule l'angle cible dans le repère GLOBAL pour le terme heading.

        Intégration IFGM-DWA (cœur de l'article) :

        L'IFGM publie cmd_dir ∈ [-1,1] = steering_angle / max_steering_angle.
        steering_angle est un angle relatif au cap courant du robot (repère robot).

        Pour le terme heading(v,ω) de l'Eq. (8) — qui compare le cap prédit
        du robot avec l'angle vers la destination — on doit travailler dans
        le repère GLOBAL :

            angle_guide_global = theta_robot + cmd_dir × max_steering_angle

        Quand l'IFGM n'est pas disponible → angle direct vers le goal (mode dégradé).

        Retourne l'angle cible en radians dans le repère global.
        """
        cfg = self.config

        if self.cmd_dir_raw is not None:
            # Reconstruction de l'angle global depuis la valeur normalisée IFGM
            # steering_angle [rad] = cmd_dir × max_steering_angle  (repère robot)
            # angle_global   [rad] = theta_robot + steering_angle
            steering_angle = self.cmd_dir_raw * cfg.max_steering_angle
            return self.theta + steering_angle
        else:
            # Mode dégradé : angle direct vers le goal dans le repère global
            if self.goal_x is not None and self.goal_y is not None:
                return math.atan2(self.goal_y - self.y, self.goal_x - self.x)
            return self.theta   # Fallback : garder le cap actuel

    def objective_function(
        self, v: float, delta: float, target_angle_global: float
    ) -> float:
        """
        G(V, δ) = σ(α·heading + β·dist + γ·vel)   — Eq. (8)

        heading(v, δ) :
            Mesure l'alignement entre le cap PRÉDIT du robot après un pas
            Ackermann et l'angle cible global (fourni par l'IFGM ou le goal).

            C'est exactement le terme heading(v,ω) de l'article :
            "represents the angle between the robot heading and the goal
            coordinates" — où "goal coordinates" est remplacé par l'angle
            guide Φ de l'IFGM quand disponible.

            Cap prédit : θ_prédit = θ + (V/L)·tan(δ)·dt   — Eq. (3)
            Erreur     : |target_angle_global - θ_prédit|  normalisée dans [0,1]

        dist(v, δ) :
            Distance minimale normalisée aux obstacles sur la trajectoire prédite.

        vel(v, δ) :
            Vitesse linéaire normalisée — favorise la rapidité en course.

        Retourne -inf si la trajectoire est inadmissible.
        """
        cfg = self.config

        #  Cap prédit après un pas Ackermann — Eq. (3) 
        omega           = self.delta_to_omega(v, delta)
        predicted_theta = self.theta + omega * cfg.dt

        #  heading score — terme heading(v,ω) de l'Eq. (8) 
        # Erreur angulaire entre cap prédit et angle guide global
        # Ramenée dans [0, π] puis normalisée dans [0, 1]
        heading_error = abs(target_angle_global - predicted_theta)
        heading_error = heading_error % (2 * math.pi)
        if heading_error > math.pi:
            heading_error = 2 * math.pi - heading_error
        # heading_score = 1 quand parfaitement aligné, 0 quand opposé
        heading_score = (math.pi - heading_error) / math.pi

        #  dist score — terme dist(v,ω) de l'Eq. (8) 
        dist = self._predict_min_distance(v, delta)
        if dist is None:
            return -float('inf')
        dist_score = min(dist / 5.0, 1.0)   # 5 m = saturation

        #  vel score — terme vel(v,ω) de l'Eq. (8) 
        vel_score = v / cfg.v_max if cfg.v_max > 0 else 0.0

        #  Fonction objectif finale — Eq. (8) 
        return cfg.alpha * heading_score + cfg.beta * dist_score + cfg.gamma * vel_score

    # RÉDUCTION VITESSE À L'APPROCHE DU GOAL — Eq. (29)


    def apply_goal_speed_reduction(self, v: float) -> float:
        """
        Réduit progressivement la vitesse dans la zone d'arrivée — Eq. (29).
        Évite de dépasser le goal (problème de convergence globale de l'article).

            v_opt = v                               si d_goal > goal_limit
            v_opt = v · (d_goal / goal_limit)       si d_goal ≤ goal_limit
        """
        if self.goal_x is None:
            return v
        d_goal = math.sqrt(
            (self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2
        )
        if d_goal < self.config.goal_limit_distance:
            return v * (d_goal / self.config.goal_limit_distance)
        return v

    # ÉTAT OPÉRATIONNEL — Figure 8 de l'article

    def update_robot_state(self):
        """
        Transitions d'état selon la proximité des obstacles (Figure 8).

        FREE_SPACE    : d_min > 2·seuil → priorité vitesse    Eq. (17-19)
        HEADING_FIRST : seuil < d_min ≤ 2·seuil → cap ajusté  Eq. (20-22)
        SAFETY_FIRST  : d_min ≤ seuil  → quasi-arrêt           Eq. (23-25)
        """
        cfg = self.config
        if self.d_min > cfg.d_safe_threshold * 2.0:
            self.current_state = RobotState.FREE_SPACE
        elif self.d_min > cfg.d_safe_threshold:
            self.current_state = RobotState.HEADING_FIRST
        else:
            self.current_state = RobotState.SAFETY_FIRST

    # BOUCLE PRINCIPALE DWA

    def compute_velocity_command(self):
        """
        Boucle principale cadencée à 1/dt Hz.

        Étapes :
        1. Vérification données disponibles
        2. Condition d'arrivée au goal — Eq. (26)(28)
        3. Calcul de l'angle cible global (IFGM ou mode dégradé)
        4. Fenêtre atteignable (V, δ) — Eq. (15)
        5. Maximisation G(V, δ) sur V_r ∩ V_a — Eq. (8)(16)
        6. Réduction vitesse au goal — Eq. (29)
        7. Conversion (V, δ) → Twist et publication sur /cmd_vel
        """
        if self.goal_reached:
            return

        if not self.scan_ranges or self.goal_x is None:
            return

        #  Condition d'arrivée — Eq. (26)(28) 
        d_goal = math.sqrt(
            (self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2
        )
        if d_goal < self.config.goal_limit_distance:
            self._stop_vehicle()
            self.goal_reached = True
            self.get_logger().info("✓ Goal atteint — véhicule arrêté.")
            return

        #  Mise à jour état opérationnel 
        self.update_robot_state()

        #  Angle cible global pour le terme heading — Eq. (8) 
        # Reconstruit depuis cmd_dir IFGM (repère robot → repère global)
        # ou cap direct vers le goal si IFGM non disponible
        target_angle_global = self._compute_heading_angle()

        if self.cmd_dir_raw is None:
            self.get_logger().warn(
                "⚠  /cmd_dir non reçu — mode dégradé (cap direct vers goal)",
                throttle_duration_sec=5.0
            )

        #  Fenêtre de vitesses atteignables — Eq. (15) 
        v_min_r, v_max_r, d_min_r, d_max_r = self.compute_reachable_window()

        #  Recherche du meilleur (V, δ) dans V_r ∩ V_a 
        cfg        = self.config
        best_v     = 0.0
        best_delta = 0.0
        best_score = -float('inf')

        v_samples     = np.arange(v_min_r, v_max_r + cfg.v_resolution,     cfg.v_resolution)
        delta_samples = np.arange(d_min_r, d_max_r + cfg.delta_resolution, cfg.delta_resolution)

        for v in v_samples:
            for delta in delta_samples:

                #  Filtrage par état opérationnel (Figure 8) 
                if self.current_state == RobotState.FREE_SPACE:
                    # Eq. (17-19) : priorité vitesse, pas de marche AR en course
                    if v <= 0.0:
                        continue

                elif self.current_state == RobotState.SAFETY_FIRST:
                    # Eq. (23-25) : vitesse très limitée
                    if v > cfg.v_max * 0.25:
                        continue

                #  Admissibilité — Eq. (16) 
                if not self.is_admissible(v, delta):
                    continue

                #  Fonction objectif — Eq. (8) 
                score = self.objective_function(v, delta, target_angle_global)
                if score > best_score:
                    best_score = score
                    best_v     = v
                    best_delta = delta

        #  Aucune trajectoire admissible → arrêt d'urgence 
        if best_score == -float('inf'):
            self.get_logger().warn("⚠  Aucune trajectoire admissible — arrêt d'urgence")
            self._stop_vehicle()
            return

        #  Réduction vitesse à l'approche du goal — Eq. (29) 
        best_v = self.apply_goal_speed_reduction(best_v)

        #  Conversion (V, δ) → Twist 
        # linear.x  = V [m/s]
        # angular.z = ω = (V/L)·tan(δ) [rad/s]
        omega = self.delta_to_omega(best_v, best_delta)

        cmd = Twist()
        cmd.linear.x  = float(best_v)
        cmd.angular.z = float(omega)
        self.cmd_vel_pub.publish(cmd)

        #  Debug 
        ifgm_status = f"{self.cmd_dir_raw:.2f}" if self.cmd_dir_raw is not None else "N/A"
        status = String()
        status.data = (
            f"[{self.current_state}] "
            f"V={best_v:.2f}m/s | "
            f"δ={math.degrees(best_delta):.1f}° | "
            f"ω={omega:.3f}rad/s | "
            f"d_min={self.d_min:.2f}m | "
            f"cmd_dir={ifgm_status} | "
            f"target={math.degrees(target_angle_global):.1f}° | "
            f"d_goal={d_goal:.2f}m"
        )
        self.status_pub.publish(status)
        self.get_logger().info(status.data, throttle_duration_sec=0.5)

    def _stop_vehicle(self):
        """Arrêt complet du véhicule."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


# ENTRY POINT

def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DWA Node arrêté.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()