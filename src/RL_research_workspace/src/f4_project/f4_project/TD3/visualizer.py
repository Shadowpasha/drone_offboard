import cv2
import numpy as np
import math

def start_visualizer(state_queue):
    """
    Premium drone navigation visualizer.

    Queue data format (10 elements):
        (lidar, goal_dist, goal_heading_rad, dev_x, dev_y, action, vx_body, vy_body, pos_x, pos_y)

    - lidar: 60 normalized rays [0,1], where 1.0 = max_range (12m), matching MuJoCo deployment env.
    - goal_dist: raw distance in meters to goal.
    - goal_heading: heading error in radians [-pi, pi].
    - dev_x: forward offset from drone to goal (body frame), in meters.
    - dev_y: lateral (left) offset from drone to goal (body frame), in meters.
    - action: [fwd_cmd, lat_cmd] policy output in range [-1, 1].
    - vx_body: body-frame forward velocity (m/s).
    - vy_body: body-frame lateral velocity (m/s).
    - pos_x: world East position (m).
    - pos_y: world North position (m).
    """
    print("DEBUG: OpenCV Drone Visualizer started (60-ray LiDAR / 70-dim obs mode).")

    # --- Layout ---
    canvas_w = 900
    canvas_h = 700
    cx = 305           # LiDAR display center X
    cy = 350           # LiDAR display center Y
    lidar_scale = 22.0  # pixels per meter (12m max → 264px)
    lidar_max_px = 270  # clip radius

    # Color palette (BGR)
    C_BG          = (20,  20,  28)
    C_GRID        = (50,  55,  70)
    C_GRID_LABEL  = (90,  95, 110)
    C_RAY_HIT     = (40,  40, 220)
    C_RAY_CLEAR   = (50,  70,  40)
    C_HIT_DOT     = (30,  30, 255)
    C_DRONE       = (0,  160, 255)
    C_HEADING     = (255, 255, 255)
    C_GOAL        = (0,  220,  80)
    C_ACTION      = (255, 220,   0)
    C_VEL         = (255, 140,   0)
    C_ACCENT      = (0,  200, 160)
    C_TEXT_MAIN   = (230, 235, 245)
    C_TEXT_DIM    = (120, 128, 148)
    C_WARN        = (0,  110, 255)
    C_SAFE        = (0,  200, 100)
    C_PANEL_SEP   = (45,  50,  65)

    max_range_m = 12.0

    last_data = None
    while True:
        if not state_queue.empty():
            data = None
            while not state_queue.empty():
                data = state_queue.get()
            if data is not None:
                last_data = data

        if last_data is not None:
            # Unpack with backward-compatible fallback
            if len(last_data) >= 10:
                lidar, goal_dist, goal_heading, dev_x, dev_y, action, vx_body, vy_body, pos_x, pos_y = last_data[:10]
            elif len(last_data) == 6:
                lidar, goal_dist, goal_heading, dev_x, dev_y, action = last_data
                vx_body = vy_body = pos_x = pos_y = 0.0
            else:
                last_data = None
                continue

            lidar = np.asarray(lidar, dtype=np.float32)
            action_arr = np.asarray(action, dtype=np.float32).flatten()
            if action_arr.size < 2:
                action_arr = np.pad(action_arr, (0, 2 - action_arr.size))

            speed = math.sqrt(float(vx_body)**2 + float(vy_body)**2)
            min_lidar_m = float(np.min(lidar)) * max_range_m

            # ── Canvas ───────────────────────────────────────────────────────────
            img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            img[:] = C_BG

            # ── Title bar ────────────────────────────────────────────────────────
            cv2.putText(img, "DRONE RL NAVIGATOR", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_ACCENT, 2)
            cv2.putText(img, "60-RAY LIDAR  |  OBS DIM: 70  |  MUJOCO-ALIGNED", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT_DIM, 1)
            cv2.line(img, (0, 58), (640, 58), C_PANEL_SEP, 1)

            # ── Panel divider ────────────────────────────────────────────────────
            cv2.line(img, (640, 0), (640, canvas_h), C_PANEL_SEP, 1)

            # ── Grid rings ───────────────────────────────────────────────────────
            for r_m in [2, 4, 6, 8, 10, 12]:
                r_px = int(r_m * lidar_scale)
                if r_px > lidar_max_px:
                    break
                cv2.circle(img, (cx, cy), r_px, C_GRID, 1)
                lx = cx + r_px + 3
                if lx < 620:
                    cv2.putText(img, f"{r_m}m", (lx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_GRID_LABEL, 1)

            # Cardinal ticks
            for deg, lbl in [(0, "FWD"), (90, "L"), (180, "BWD"), (270, "R")]:
                a = math.radians(deg - 90)
                tx = int(cx + (lidar_max_px + 14) * math.cos(a))
                ty = int(cy + (lidar_max_px + 14) * math.sin(a))
                cv2.putText(img, lbl, (tx - 8, ty + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_TEXT_DIM, 1)

            # ── LiDAR rays ───────────────────────────────────────────────────────
            num_rays = len(lidar)
            for i, val in enumerate(lidar):
                dist_m = float(val) * max_range_m
                # index 0 = 180° (behind), index num_rays//2 = 0° (forward)
                angle_deg = (i / num_rays) * 360.0 - 180.0
                angle_rad = math.radians(-angle_deg - 90.0)

                px = int(cx + dist_m * lidar_scale * math.cos(angle_rad))
                py = int(cy + dist_m * lidar_scale * math.sin(angle_rad))
                px = max(5, min(px, 630))
                py = max(62, min(py, canvas_h - 5))

                hit = val < 0.99
                cv2.line(img, (cx, cy), (px, py), C_RAY_HIT if hit else C_RAY_CLEAR, 1)
                if hit:
                    cv2.circle(img, (px, py), 2, C_HIT_DOT, -1)

            # ── Safe-distance ring ───────────────────────────────────────────────
            safe_px = int(0.6 * lidar_scale)
            ring_col = C_WARN if min_lidar_m < 0.6 else C_SAFE
            cv2.circle(img, (cx, cy), safe_px, ring_col, 1)

            # ── Goal ─────────────────────────────────────────────────────────────
            gx = int(cx - dev_y * lidar_scale)
            gy = int(cy - dev_x * lidar_scale)
            gx = max(10, min(gx, 628))
            gy = max(62, min(gy, canvas_h - 10))
            cv2.circle(img, (gx, gy), 13, (0, 70, 30), -1)
            cv2.circle(img, (gx, gy), 9, C_GOAL, -1)
            cv2.circle(img, (gx, gy), 9, (180, 255, 180), 1)
            cv2.line(img, (cx, cy), (gx, gy), (0, 90, 35), 1)
            cv2.putText(img, "GOAL", (gx + 12, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GOAL, 1)

            # ── Action vector ────────────────────────────────────────────────────
            apx = int(cx - float(action_arr[1]) * 60)
            apy = int(cy - float(action_arr[0]) * 60)
            cv2.arrowedLine(img, (cx, cy), (apx, apy), C_ACTION, 2, tipLength=0.3)
            cv2.putText(img, "CMD", (apx + 4, apy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_ACTION, 1)

            # ── Velocity vector ──────────────────────────────────────────────────
            vscale = 50.0
            vpx = int(cx - float(vy_body) * vscale)
            vpy = int(cy - float(vx_body) * vscale)
            cv2.arrowedLine(img, (cx, cy), (vpx, vpy), C_VEL, 2, tipLength=0.3)
            cv2.putText(img, "VEL", (vpx + 4, vpy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_VEL, 1)

            # ── Drone marker ─────────────────────────────────────────────────────
            cv2.circle(img, (cx, cy), 15, (0, 70, 130), -1)
            cv2.circle(img, (cx, cy), 11, C_DRONE, -1)
            cv2.circle(img, (cx, cy), 11, (200, 220, 255), 1)
            cv2.arrowedLine(img, (cx, cy), (cx, cy - 22), C_HEADING, 2, tipLength=0.4)

            # ═══════════════════════════════════════════════════════════════════
            # Right panel: telemetry
            # ═══════════════════════════════════════════════════════════════════
            rpx = 652

            def row(y, label, value, color=C_TEXT_MAIN, unit=""):
                cv2.putText(img, label, (rpx, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT_DIM, 1)
                cv2.putText(img, f"{value}{unit}", (rpx + 145, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            def sec(y, title):
                cv2.putText(img, title, (rpx, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.41, C_ACCENT, 1)
                cv2.line(img, (rpx, y + 4), (rpx + 240, y + 4), (45, 55, 75), 1)

            sec(80,  "NAVIGATION")
            dist_col = C_SAFE if goal_dist < 1.0 else C_TEXT_MAIN
            row(103, "Distance",   f"{float(goal_dist):.2f}", dist_col, " m")
            row(122, "Heading Err", f"{math.degrees(float(goal_heading)):+.1f}", C_TEXT_MAIN, " deg")
            row(141, "Dev Fwd",    f"{float(dev_x):+.2f}", C_TEXT_MAIN, " m")
            row(160, "Dev Left",   f"{float(dev_y):+.2f}", C_TEXT_MAIN, " m")

            sec(188, "POSITION  (ENU)")
            row(211, "East  X",   f"{float(pos_x):+.2f}", C_TEXT_MAIN, " m")
            row(230, "North Y",   f"{float(pos_y):+.2f}", C_TEXT_MAIN, " m")

            sec(258, "BODY VELOCITY")
            spd_col = C_WARN if speed > 0.25 else C_TEXT_MAIN
            row(281, "Vx (fwd)",  f"{float(vx_body):+.3f}", spd_col, " m/s")
            row(300, "Vy (left)", f"{float(vy_body):+.3f}", spd_col, " m/s")
            row(319, "Speed",     f"{speed:.3f}", spd_col, " m/s")

            sec(347, "POLICY OUTPUT")
            row(370, "Act[0] Fwd", f"{float(action_arr[0]):+.3f}", C_ACTION)
            row(389, "Act[1] Lat", f"{float(action_arr[1]):+.3f}", C_ACTION)

            sec(417, "LIDAR  (60 rays / 12m)")
            min_col = C_WARN if min_lidar_m < 0.6 else (C_TEXT_MAIN if min_lidar_m < 2.0 else C_SAFE)
            row(440, "Min range",  f"{min_lidar_m:.2f}", min_col, " m")
            row(459, "Mean range", f"{float(np.mean(lidar)) * max_range_m:.2f}", C_TEXT_MAIN, " m")
            row(478, "Blind rays", f"{int(np.sum(lidar >= 0.999))}", C_TEXT_DIM, " / 60")

            sec(506, "SAFETY STATUS")
            bar_x, bar_y = rpx, 526
            bar_w, bar_h = 232, 13
            fill = int(min(1.0, min_lidar_m / max_range_m) * bar_w)
            fc = C_WARN if min_lidar_m < 0.6 else (C_ACTION if min_lidar_m < 1.5 else C_SAFE)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (38, 42, 52), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), fc, -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 80), 1)
            thresh_px = int(0.6 / max_range_m * bar_w)
            cv2.line(img, (bar_x + thresh_px, bar_y - 2), (bar_x + thresh_px, bar_y + bar_h + 2), C_WARN, 1)
            cv2.putText(img, "0.6m", (bar_x + thresh_px - 10, bar_y + bar_h + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_WARN, 1)

            # Obs strip (70-dim)
            sec(552, "OBS VECTOR  (70-dim)")
            obs_full = np.concatenate([lidar,
                                       [float(action_arr[0]), float(action_arr[1]),
                                        float(goal_dist), float(goal_heading),
                                        float(vx_body), float(vy_body),
                                        float(pos_x), float(pos_y), 0.0, 0.0]])[:70]
            obs_n = np.clip((obs_full + 1.0) / 2.0, 0.0, 1.0)
            strip_x, strip_y = rpx, 572
            strip_w_total, strip_h = 232, 15
            sw = max(1, strip_w_total // 70)
            for ii, vv in enumerate(obs_n):
                bx = strip_x + ii * sw
                cv_val = int(vv * 255)
                col = (0, cv_val // 2, cv_val) if ii < 60 else (0, cv_val, 0)
                cv2.rectangle(img, (bx, strip_y), (bx + sw - 1, strip_y + strip_h), col, -1)
            cv2.rectangle(img, (strip_x, strip_y), (strip_x + strip_w_total, strip_y + strip_h), (60, 65, 80), 1)
            cv2.putText(img, "laser[0:60]", (strip_x, strip_y + strip_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_TEXT_DIM, 1)
            cv2.putText(img, "state[60:70]", (strip_x + 160, strip_y + strip_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 200, 100), 1)

            # Bottom status
            cv2.line(img, (0, canvas_h - 30), (640, canvas_h - 30), C_PANEL_SEP, 1)
            status = (f"Dist:{float(goal_dist):.2f}m  HeadErr:{math.degrees(float(goal_heading)):+.1f}deg  "
                      f"MinLiDAR:{min_lidar_m:.2f}m  Speed:{speed:.3f}m/s  "
                      f"Pos:({float(pos_x):.1f},{float(pos_y):.1f})")
            cv2.putText(img, status, (10, canvas_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1)
            cv2.putText(img, "q: quit", (rpx, canvas_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT_DIM, 1)

            cv2.imshow("Drone RL Visualizer", img)

        else:
            # Idle screen
            img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            img[:] = (20, 20, 28)
            cv2.putText(img, "DRONE RL NAVIGATOR", (canvas_w // 2 - 130, canvas_h // 2 - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 160), 2)
            cv2.putText(img, "Waiting for environment data...", (canvas_w // 2 - 130, canvas_h // 2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 128, 148), 1)
            cv2.imshow("Drone RL Visualizer", img)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import queue as q_mod
    q = q_mod.Queue()
    lidar = np.ones(60, dtype=np.float32) * 0.95
    lidar[10:15] = 0.12   # obstacles ahead-left
    lidar[45:50] = 0.25   # obstacles right
    q.put((lidar, 4.5, 0.35, 3.8, -1.2, np.array([0.6, -0.4]), 0.12, -0.05, 2.5, 1.3))
    start_visualizer(q)
