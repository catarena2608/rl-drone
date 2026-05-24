"""
STAGE 1 — Hover tại chỗ (Stabilization & Position Control theo chuẩn Bài báo)
===========================================================================
Bài toán: Giữ yên drone tại vị trí mục tiêu cố định [0, 0, 1.0] sử dụng không 
gian trạng thái đầy đủ 12 chiều theo đặc tả của gym-pybullet-drones.

Obs (12,): pos(3) + rpy(3) + vel(3) + ang_vel(3)

Fixes:
    - reset() override: _step_n, _last_action, _prev_action reset đúng mỗi episode
    - _computeReward: _last_action được set từ action qua _preprocessAction override
    - _computeTerminated: nới rộng ngưỡng crash/out-of-bounds để tránh die ngay step 1
    - __init__: initial_xyzs mặc định random xung quanh target thay vì đúng tâm
"""

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class Stage1HoverAviary(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 gui: bool = False,
                 record: bool = False):

        # Mục tiêu cố định: tâm hệ tọa độ ở độ cao 1m
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        self.MAX_STEPS  = 2000

        # Các biến nội bộ — sẽ được reset đúng trong reset()
        self._step_n      = 0
        self._last_action = None
        self._prev_action = None

        # FIX: spawn ngẫu nhiên trong bán kính nhỏ xung quanh target
        # thay vì đúng tâm [0,0,1] — giúp agent học tổng quát hơn
        if initial_xyzs is None:
            offset = np.random.uniform(-0.1, 0.1, size=(1, 3))
            offset[0, 2] = np.random.uniform(-0.05, 0.05)  # ít offset hơn theo Z
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
        """
        FIX CHÍNH: reset toàn bộ biến nội bộ mỗi episode để tránh
        _step_n tích lũy gây truncated ngay step 1 ở episode tiếp theo.
        """
        self._step_n      = 0
        self._last_action = None
        self._prev_action = None

        # FIX: random spawn mới mỗi lần reset
        offset = np.random.uniform(-0.1, 0.1, size=(1, 3))
        offset[0, 2] = np.random.uniform(-0.05, 0.05)
        self.INIT_XYZS = self.TARGET_POS + offset

        return super().reset(seed=seed, options=options)

    # ── Observation Space ─────────────────────────────────────────────────────

    def _observationSpace(self):
        """
        Không gian trạng thái 12 chiều theo chuẩn bài báo:
        [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
        """
        lo = -np.inf
        hi =  np.inf
        return spaces.Box(low=np.array([lo]*12,  dtype=np.float32),
                          high=np.array([hi]*12, dtype=np.float32),
                          dtype=np.float32)

    def _computeObs(self):
        """
        Build vector trạng thái 12 chiều từ PyBullet state vector.
        s layout: pos(0:3), quat(3:7), rpy(7:10), vel(10:13), ang_vel(13:16), last_action(16:20)
        """
        s       = self._getDroneStateVector(0)
        pos     = s[0:3]
        rpy     = s[7:10]
        vel     = s[10:13]
        ang_vel = s[13:16]
        return np.hstack([pos, rpy, vel, ang_vel]).astype(np.float32)

    # ── Action hook để track _last_action ─────────────────────────────────────

    def step(self, action):
        """
        FIX: ghi lại action trước khi gọi super().step() để _computeReward
        có thể tính phạt thay đổi hành động đột ngột.
        """
        # Cập nhật lịch sử action
        self._prev_action = self._last_action.copy() if self._last_action is not None else None
        self._last_action = np.array(action, dtype=np.float32).flatten()

        return super().step(action)

    # ── Reward Function ───────────────────────────────────────────────────────

    def _computeReward(self):
        """
        Hàm phần thưởng theo chuẩn bài báo:
          r = -dist² - 0.1*att² - 0.1*ang_vel² - 0.01*Δaction²
        """
        s       = self._getDroneStateVector(0)
        pos     = s[0:3]
        rpy     = s[7:10]
        ang_vel = s[13:16]

        # 1. Phạt khoảng cách Euclidean tới target
        dist_error = np.linalg.norm(pos - self.TARGET_POS)
        r_dist = -np.sqrt(dist_error) * 1.5

        # 2. Phạt góc nghiêng Roll/Pitch
        r_att = -0.1 * float(np.linalg.norm(rpy[0:2]) ** 2)

        # 3. Phạt vận tốc góc (chống xoay lắc)
        r_stability = -0.1 * float(np.linalg.norm(ang_vel) ** 2)

        r_progress = 0.0
        if self.prev_dist is not None:
            # Nếu khoảng cách giảm đi => Drone đi đúng hướng => Thưởng lớn
            r_progress = (self.prev_dist - dist_error) * 20.0
        self.prev_dist = dist_error

        # 4. Phạt thay đổi action đột ngột
        # FIX: _last_action và _prev_action được set đúng trong step() override
        r_action_leash = 0.0
        if self._last_action is not None and self._prev_action is not None:
            r_action_leash = -0.01 * float(np.linalg.norm(self._last_action - self._prev_action) ** 2)

        return float(r_dist + r_att + r_stability + r_action_leash + r_progress)

    # ── Terminated & Truncated ────────────────────────────────────────────────

    def _computeTerminated(self):
        """
        FIX: Nới rộng ngưỡng crash và out-of-bounds để tránh die ngay step 1
        do PyBullet jitter khi spawn.
          - Z thấp: 0.05 thay vì 0.1
          - Z cao:  3.0 thay vì 2.0
          - XY:     3.0 thay vì 2.0
          - Góc:    1.2 rad (~69°) thay vì 1.0 rad (~57°)
        """
        s   = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]

        out_of_bounds = (
            pos[2] < 0.05
            or pos[2] > 3.0
            or abs(pos[0]) > 3.0
            or abs(pos[1]) > 3.0
        )
        crash = abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2

        return bool(out_of_bounds or crash)

    def _computeTruncated(self):
        """
        FIX: _step_n được reset trong reset() nên không tích lũy qua các episode.
        """
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        s   = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]

        dist       = np.linalg.norm(pos - self.TARGET_POS)
        is_hovering = (
            dist < 0.15
            and np.linalg.norm(vel)    < 0.2
            and np.linalg.norm(rpy[:2]) < 0.2
        )

        return {
            "step_n":         self._step_n,
            "dist_to_target": float(dist),
            "is_hovering":    bool(is_hovering),
            "pos_z":          float(pos[2]),
            "roll":           float(rpy[0]),
            "pitch":          float(rpy[1]),
        }
