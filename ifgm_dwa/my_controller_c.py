# Contrôleur Webots TT-02 — Follow the Gap v4
# Correctif blocage : détection cul-de-sac + marche arrière

from vehicle import Driver
from controller import Lidar
import numpy as np

# =============================================================================
# PARAMÈTRES
# =============================================================================

IDX_AVANT          = 180
CHAMP_VISION       = 60        # ±60° autour de l'avant

MASQUE_ROBOT_DEBUT = 330
MASQUE_ROBOT_FIN   = 30

BUBBLE_RADIUS      = 0.15
MAX_LIDAR_RANGE    = 6.0
WINDOW_SIZE        = 5
MIN_GAP_SIZE       = 8

SPEED_MAX_KMH      = 5.0
SPEED_MIN_KMH      = 2.0
STEER_GAIN         = 1.2
MAX_STEER_RAD      = 0.28

NB_POINTS          = 360
DEG_PAR_IDX        = 360.0 / NB_POINTS

# Seuils de détection de blocage
SEUIL_MUR_AVANT    = 0.35      # m — en dessous : on est trop près du mur
DUREE_RECUL_S      = 1.5       # secondes de marche arrière
SPEED_RECUL_KMH    = -2.0      # km/h négatif = marche arrière

# =============================================================================
# CLASSE FOLLOW THE GAP
# =============================================================================

class FollowGap:

    def preprocess_lidar(self, ranges):
        proc = np.array(ranges, dtype=float)
        proc = np.where(np.isinf(proc) | np.isnan(proc) | (proc == 0.0),
                        MAX_LIDAR_RANGE, proc)
        proc = np.clip(proc, 0.0, MAX_LIDAR_RANGE)
        proc[MASQUE_ROBOT_DEBUT:] = MAX_LIDAR_RANGE
        proc[:MASQUE_ROBOT_FIN]   = MAX_LIDAR_RANGE
        masque = np.zeros(NB_POINTS, dtype=bool)
        masque[IDX_AVANT - CHAMP_VISION : IDX_AVANT + CHAMP_VISION] = True
        proc[~masque] = 0.0
        kernel = np.ones(WINDOW_SIZE) / WINDOW_SIZE
        proc   = np.convolve(proc, kernel, mode='same')
        return proc

    def apply_bubble(self, proc_ranges):
        champ        = proc_ranges[IDX_AVANT - CHAMP_VISION : IDX_AVANT + CHAMP_VISION]
        local_min    = np.argmin(champ)
        closest_idx  = (IDX_AVANT - CHAMP_VISION) + local_min
        closest_dist = proc_ranges[closest_idx]

        if closest_dist > 0 and BUBBLE_RADIUS < closest_dist:
            half_angle = np.arcsin(min(BUBBLE_RADIUS / closest_dist, 1.0))
        else:
            half_angle = np.radians(20)

        bubble_half = int(half_angle / np.radians(DEG_PAR_IDX))
        i_start = max(0, closest_idx - bubble_half)
        i_end   = min(NB_POINTS - 1, closest_idx + bubble_half)
        proc_ranges[i_start:i_end+1] = 0.0

        print(f"[BUBBLE] idx={closest_idx} dist={closest_dist:.2f}m "
              f"bulle={i_end-i_start+1}pts")
        return proc_ranges

    def find_max_gap(self, free_ranges):
        best_s, best_e, best_len = IDX_AVANT, IDX_AVANT, 0
        cur_start = None

        for i in range(IDX_AVANT - CHAMP_VISION, IDX_AVANT + CHAMP_VISION):
            if free_ranges[i] > 0:
                if cur_start is None:
                    cur_start = i
            else:
                if cur_start is not None:
                    length = i - cur_start
                    if length > best_len:
                        best_len = length
                        best_s   = cur_start
                        best_e   = i - 1
                    cur_start = None

        if cur_start is not None:
            length = (IDX_AVANT + CHAMP_VISION) - cur_start
            if length > best_len:
                best_s = cur_start
                best_e = IDX_AVANT + CHAMP_VISION - 1

        print(f"[GAP] [{best_s}:{best_e}] ({best_e-best_s+1}pts) "
              f"[{(best_s-IDX_AVANT)*DEG_PAR_IDX:+.0f}° "
              f"→ {(best_e-IDX_AVANT)*DEG_PAR_IDX:+.0f}°]")
        return best_s, best_e

    def find_best_point(self, start_i, end_i, proc_ranges):
        indices = np.arange(start_i, end_i + 1)
        poids   = proc_ranges[start_i:end_i+1]
        best_idx = int(np.round(np.average(indices, weights=poids))) \
                   if poids.sum() > 0 else (start_i + end_i) // 2
        print(f"[BEST] idx={best_idx} "
              f"offset={best_idx-IDX_AVANT:+d}pts "
              f"({(best_idx-IDX_AVANT)*DEG_PAR_IDX:+.1f}°) "
              f"dist={proc_ranges[best_idx]:.2f}m")
        return best_idx

    def compute_commands(self, best_idx, proc_ranges):
        offset_rad = np.radians((best_idx - IDX_AVANT) * DEG_PAR_IDX)
        steer = np.clip(STEER_GAIN * offset_rad, -MAX_STEER_RAD, MAX_STEER_RAD)
        ratio = abs(steer) / MAX_STEER_RAD
        speed = SPEED_MAX_KMH - ratio * (SPEED_MAX_KMH - SPEED_MIN_KMH)
        dist_avant = float(np.mean(proc_ranges[IDX_AVANT - 10 : IDX_AVANT + 10]))
        if dist_avant > 3.0 and abs(steer) < 0.05:
            speed = min(speed * 1.2, SPEED_MAX_KMH)
        print(f"[CMD] steer={np.degrees(steer):+.1f}°  "
              f"speed={speed:.2f}km/h  dist_avant={dist_avant:.2f}m")
        return speed, steer

    def dist_avant(self, proc_ranges):
        return float(np.mean(proc_ranges[IDX_AVANT - 10 : IDX_AVANT + 10]))

    def process(self, ranges):
        proc = self.preprocess_lidar(ranges)
        free = self.apply_bubble(proc.copy())
        s, e = self.find_max_gap(free)

        if e - s < MIN_GAP_SIZE:
            print("[PROCESS] ⚠ Pas de gap — rotation vers le côté le plus libre")
            dist_g = float(np.mean(proc[IDX_AVANT - CHAMP_VISION : IDX_AVANT - CHAMP_VISION//2]))
            dist_d = float(np.mean(proc[IDX_AVANT + CHAMP_VISION//2 : IDX_AVANT + CHAMP_VISION]))
            steer  = MAX_STEER_RAD if dist_g > dist_d else -MAX_STEER_RAD
            return SPEED_MIN_KMH, steer

        best = self.find_best_point(s, e, proc)
        return self.compute_commands(best, proc)


# =============================================================================
# CONTRÔLEUR WEBOTS
# =============================================================================

driver = Driver()
basicTimeStep  = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("RpLidarA2")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

ftg = FollowGap()

modeManuel  = False
modeAuto    = False
modeDebug   = False
modeRecul   = False          # NEW : marche arrière active
t_recul_fin = 0.0            # NEW : timestamp de fin du recul
steer_recul = 0.0            # NEW : direction du recul

speed_kmh  = 0.0
steer_rad  = 0.0
step_count = 0

driver.setCruisingSpeed(0)
driver.setSteeringAngle(0)

print("=" * 60)
print("  Follow the Gap v4 — anti-blocage marche arrière")
print("  a:Auto  m:Manuel  n:Stop  d:Debug  l:LiDAR")
print("=" * 60)

while driver.step() != -1:
    step_count += 1
    t_sim  = driver.getTime()
    ranges = lidar.getRangeImage()

    while True:
        key = keyboard.getKey()
        if key == -1:
            break
        if key in (ord('a'), ord('A')):
            if not modeAuto:
                modeAuto = True; modeManuel = False; modeRecul = False
                print("\n[MODE] Auto Follow the Gap")
        elif key in (ord('m'), ord('M')):
            if not modeManuel:
                modeManuel = True; modeAuto = False; modeRecul = False
                print("\n[MODE] Manuel")
        elif key in (ord('n'), ord('N')):
            modeAuto = False; modeManuel = False; modeRecul = False
            speed_kmh = 0.0;  steer_rad  = 0.0
            print("\n[MODE] Stop")
        elif key in (ord('d'), ord('D')):
            modeDebug = not modeDebug
            print(f"\n[DEBUG] {'ON' if modeDebug else 'OFF'}")
        elif key in (ord('l'), ord('L')):
            arr = np.array(ranges)
            print(f"\n[LIDAR] avant(180)={arr[180]:.2f}m  "
                  f"gauche(120)={arr[120]:.2f}m  "
                  f"droite(240)={arr[240]:.2f}m")
        if modeManuel:
            if key == keyboard.UP:
                speed_kmh = min(speed_kmh + 0.5, SPEED_MAX_KMH)
            elif key == keyboard.DOWN:
                speed_kmh = max(speed_kmh - 0.5, 0.0)
            elif key == keyboard.LEFT:
                steer_rad = max(steer_rad - 0.04, -MAX_STEER_RAD)
            elif key == keyboard.RIGHT:
                steer_rad = min(steer_rad + 0.04,  MAX_STEER_RAD)

    if not modeManuel and not modeAuto:
        speed_kmh = 0.0
        steer_rad = 0.0

    if modeAuto:
        proc = ftg.preprocess_lidar(ranges)
        dist_av = ftg.dist_avant(proc)

        # ── Fin du recul ──────────────────────────────────────────────────────
        if modeRecul and t_sim >= t_recul_fin:
            modeRecul = False
            print(f"[RECUL] Terminé à t={t_sim:.2f}s — reprise Follow the Gap")

        # ── Mode recul actif ──────────────────────────────────────────────────
        if modeRecul:
            speed_kmh = SPEED_RECUL_KMH
            steer_rad = steer_recul
            print(f"[RECUL] t_restant={t_recul_fin - t_sim:.2f}s  "
                  f"steer={np.degrees(steer_recul):+.1f}°")

        # ── Détection blocage → déclencher recul ─────────────────────────────
        elif dist_av < SEUIL_MUR_AVANT:
            # Choisir le côté de recul = côté le plus dégagé latéralement
            dist_g = float(np.mean(proc[IDX_AVANT - CHAMP_VISION : IDX_AVANT - 10]))
            dist_d = float(np.mean(proc[IDX_AVANT + 10 : IDX_AVANT + CHAMP_VISION]))
            # En reculant, gauche/droite sont inversés
            steer_recul = -MAX_STEER_RAD if dist_g > dist_d else MAX_STEER_RAD
            modeRecul   = True
            t_recul_fin = t_sim + DUREE_RECUL_S
            speed_kmh   = SPEED_RECUL_KMH
            steer_rad   = steer_recul
            print(f"[RECUL] ⚠ Mur à {dist_av:.2f}m < {SEUIL_MUR_AVANT}m — "
                  f"recul {'gauche' if steer_recul < 0 else 'droite'} "
                  f"pendant {DUREE_RECUL_S}s")

        # ── Navigation normale ────────────────────────────────────────────────
        else:
            speed_kmh, steer_rad = ftg.process(ranges)

        if modeDebug or step_count % 50 == 0:
            print(f"[step={step_count} t={t_sim:.1f}s] "
                  f"{'RECUL' if modeRecul else 'FTG  '}  "
                  f"speed={speed_kmh:.2f}km/h  "
                  f"steer={np.degrees(steer_rad):.1f}°  "
                  f"dist_avant={dist_av:.2f}m")

    driver.setCruisingSpeed(np.clip(speed_kmh, SPEED_RECUL_KMH, SPEED_MAX_KMH))
    driver.setSteeringAngle(np.clip(steer_rad, -MAX_STEER_RAD, MAX_STEER_RAD))