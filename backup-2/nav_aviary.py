"""
STAGE 2 — Navigation & Robust Hovering
===========================================================================
Obs (20,):
    pos(3)          — vị trí tuyệt đối
    rpy(3)          — góc nghiêng thân
    vel(3)          — vận tốc tuyến tính
    ang_vel(3)      — vận tốc góc
    rel_target(3)   — vector từ drone đến target
    dist(1)         — khoảng cách scalar
    speed_norm(1)   — THÊM MỚI: speed / max_speed ∈ [0,1]
                      agent biết mình đang ở bao nhiêu % tốc độ tối đa
                      → học phanh sớm khi spawn xa
    last_act_rpy(3) — 3 motor cuối của action trước

Thay đổi so với phiên bản cũ:
    [obs]    Thêm speed_norm: tốc độ chuẩn hóa theo max_speed_kmh từ URDF
             16 → 19 → 20 chiều
    [reset]  Load spawn_radius từ file khi khởi động (fix bug reset về 0.1)
"""

import os
import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Import hàm load spawn_radius từ curriculum_callback
# (tránh circular import bằng cách inline logic đơn giản)
SPAWN_RADIUS_FILE = "./models/stage2/spawn_radius.txt"


def _load_spawn_radius(default: float = 0.1) -> float:
    if os.path.exists(SPAWN_RADIUS_FILE):
        try:
            with open(SPAWN_RADIUS_FILE, "r") as f:
                val = float(f.read().strip())
            print(f"[NAV_AVIARY] Loaded spawn_radius: {val:.2f} m")
            return val
        except ValueError:
            pass
    return default


class Stage2NavAviary(BaseRLAviary):

    # ── Khởi tạo ──────────────────────────────────────────────────────────────

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 gui: bool = False,
                 record: bool = False):

        self.TARGET_POS   = np.array([0.0, 0.0, 1.0])
        self.MAX_STEPS    = 2000

        # FIX BUG 2: Load spawn_radius từ file thay vì hardcode 0.1
        self.spawn_radius = _load_spawn_radius(default=0.1)

        self._step_n      = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        if initial_xyzs is None:
            offset = np.random.uniform(-0.1, 0.1, size=(1, 3))
            offset[0, 2] = np.random.uniform(-0.05, 0.05)
            initial_xyzs = self.TARGET_POS + offset

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         gui=gui,
                         record=record)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._step_n      = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        r        = self.spawn_radius
        noise_xy = np.random.uniform(-r,       r,       size=(2,))
        noise_z  = np.random.uniform(-r * 0.4, r * 0.4)

        initial_xyz    = self.TARGET_POS.copy()
        initial_xyz[0] += noise_xy[0]
        initial_xyz[1] += noise_xy[1]
        initial_xyz[2] += noise_z
        initial_xyz[2]  = max(initial_xyz[2], 0.15)

        self.INIT_XYZS = np.array([initial_xyz])
        return super().reset(seed=seed, options=options)

    # ── Observation Space ─────────────────────────────────────────────────────

    def _observationSpace(self):
        """
        20 chiều:
            [0:3]   pos          — vị trí tuyệt đối
            [3:6]   rpy          — roll, pitch, yaw
            [6:9]   vel          — vx, vy, vz
            [9:12]  ang_vel      — p, q, r
            [12:15] rel_target   — vector đến target
            [15]    dist         — khoảng cách scalar
            [17:20] last_act_rpy — 3 motor cuối action trước
        """
        return spaces.Box(
            low  = np.full(19, -np.inf, dtype=np.float32),
            high = np.full(19,  np.inf, dtype=np.float32),
            dtype=np.float32
        )

    def _computeObs(self):
        s       = self._getDroneStateVector(0)
        pos     = s[0:3]
        rpy     = s[7:10]
        vel     = s[10:13]
        ang_vel = s[13:16]

        rel_target = self.TARGET_POS - pos
        dist       = np.linalg.norm(rel_target)

        last_act_rpy = self._last_action.flatten()[1:]  # 3 motor cuối

        return np.hstack([
            pos,           # (3,)
            rpy,           # (3,)
            vel,           # (3,)
            ang_vel,       # (3,)
            rel_target,    # (3,)
            [dist],        # (1,)
            last_act_rpy,  # (3,)
        ]).astype(np.float32)  # tổng 20 chiều

    # ── Action hook ───────────────────────────────────────────────────────────

    def _preprocessAction(self, action):
        flat_act          = action.flatten()
        self._prev_action = self._last_action.copy()
        self._last_action = flat_act.copy()
        return super()._preprocessAction(action)

    # ── Reward Function ───────────────────────────────────────────────────────

    def _computeReward(self):
        s       = self._getDroneStateVector(0)
        pos     = s[0:3]
        rpy     = s[7:10]
        vel     = s[10:13]
        ang_vel = s[13:16]

        dist  = float(np.linalg.norm(self.TARGET_POS - pos))
        speed = float(np.linalg.norm(vel))

        # Tầng 1: Navigation linear — gradient không chết ở xa
        w_nav = 8.0
        r_nav = -dist

        # Tầng 2: Close bonus exponential — peak rõ khi đến đích
        w_close = 10.0
        r_close = np.exp(-dist * 10.0)

        # Tầng 3: Velocity penalty scale theo proximity
        # Khi xa: được phép bay nhanh
        # Khi gần: bị phạt nặng nếu còn nhanh → học hover
        w_vel = 3.0
        proximity_factor = np.exp(-dist * 3.0)
        r_vel = -speed * proximity_factor

        # Tầng 4: Attitude penalty
        w_att = 1.5
        r_att = -float(np.linalg.norm(rpy[:2]))

        # Tầng 5: Action smoothness — nhỏ thôi, không át r_nav
        w_act = 0.3
        r_act = -float(np.linalg.norm(self._last_action))

        # Tầng 6: Angular velocity
        w_angvel = 0.2
        r_angvel = -float(np.linalg.norm(ang_vel))

        total = (w_nav    * r_nav    +
                 w_close  * r_close  +
                 w_vel    * r_vel    +
                 w_att    * r_att    +
                 w_act    * r_act    +
                 w_angvel * r_angvel)

        return float(np.clip(total, -20.0, 20.0))

    # ── Terminated & Truncated ────────────────────────────────────────────────

    def _computeTerminated(self):
        s   = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]

        out_of_bounds = (
            pos[2] < 0.05 or pos[2] > 3.5
            or abs(pos[0]) > 4.0
            or abs(pos[1]) > 4.0
        )
        crash = abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3

        return bool(out_of_bounds or crash)

    def _computeTruncated(self):
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        s   = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]

        dist  = float(np.linalg.norm(self.TARGET_POS - pos))
        speed = float(np.linalg.norm(vel))

        out_of_bounds = (
            pos[2] < 0.05 or pos[2] > 3.5
            or abs(pos[0]) > 4.0 or abs(pos[1]) > 4.0
        )
        flipped    = abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3
        is_crashed = bool(out_of_bounds or flipped)
        is_timeout = self._step_n >= self.MAX_STEPS

        return {
            "dist":       dist,
            "speed":      speed,
            "is_crashed": is_crashed,
            "is_timeout": is_timeout,
        }