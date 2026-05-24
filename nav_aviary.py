"""
STAGE 2 — Navigation & Robust Hovering
===========================================================================
Bài toán: Di chuyển drone từ vị trí bất kỳ đến TARGET_POS [0, 0, 1.0]
và hover ổn định tại đó.

Obs (19,):
    pos(3)          — vị trí tuyệt đối, cần thiết để generalize xa
    rpy(3)          — góc nghiêng thân
    vel(3)          — vận tốc tuyến tính
    ang_vel(3)      — vận tốc góc
    rel_target(3)   — vector từ drone đến target (hướng đi)
    dist(1)         — khoảng cách scalar đến target
    last_act_rpy(3) — 3 motor cuối của action trước (bỏ motor 0 vì redundant)

Thay đổi so với phiên bản cũ:
    [obs]     Thêm pos(3) tuyệt đối → 16 chiều lên 19 chiều
              Agent cần biết mình đang ở đâu trong không gian để generalize
    [reward]  Thêm trọng số rõ ràng theo thứ bậc ưu tiên
              Thêm velocity penalty có scale theo khoảng cách (proximity_factor)
              Xóa r_alive — tránh incentive sai khiến agent ì lại
              r_nav dùng linear thay vì exponential để gradient không chết ở xa
    [reset]   Noise theo Z nhỏ hơn noise theo XY (Z khó điều khiển hơn)
    [obs_space] Cập nhật đúng kích thước 19
"""

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


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
        self.spawn_radius = 0.1

        # Biến nội bộ — reset đúng trong reset()
        self._step_n      = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        # Spawn ban đầu gần target để super().__init__ không bị lỗi
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
        """
        Spawn ngẫu nhiên trong spawn_radius.
        
        FIX: Noise Z nhỏ hơn noise XY vì:
        - Trục Z liên quan trực tiếp đến thrust — sai lệch lớn theo Z
          đòi hỏi agent phải học bù thrust ngay lập tức, rất khó
        - Trục XY chỉ cần nghiêng thân là di chuyển được, dễ hơn
        - Tăng dần Z scale theo spawn_radius để curriculum mượt
        """
        self._step_n      = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        r    = self.spawn_radius
        # XY lấy full radius, Z chỉ lấy 40% radius
        noise_xy = np.random.uniform(-r,        r,        size=(2,))
        noise_z  = np.random.uniform(-r * 0.4,  r * 0.4)

        initial_xyz    = self.TARGET_POS.copy()
        initial_xyz[0] += noise_xy[0]
        initial_xyz[1] += noise_xy[1]
        initial_xyz[2] += noise_z
        initial_xyz[2]  = max(initial_xyz[2], 0.15)  # không cho spawn dưới đất

        self.INIT_XYZS = np.array([initial_xyz])
        return super().reset(seed=seed, options=options)

    # ── Observation Space ─────────────────────────────────────────────────────

    def _observationSpace(self):
        """
        19 chiều:
            [0:3]   pos          — vị trí tuyệt đối (x, y, z)
            [3:6]   rpy          — roll, pitch, yaw
            [6:9]   vel          — vx, vy, vz
            [9:12]  ang_vel      — p, q, r
            [12:15] rel_target   — vector đến target
            [15]    dist         — khoảng cách scalar
            [16:19] last_act_rpy — 3 motor cuối của action trước
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
        dist       = np.array([np.linalg.norm(rel_target)], dtype=np.float32)

        # Lấy 3 motor cuối (bỏ motor 0 vì 4 motor có redundancy)
        last_act_rpy = self._last_action.flatten()[1:]  # shape (3,)

        return np.hstack([
            pos,           # (3,) — THÊM MỚI: vị trí tuyệt đối
            rpy,           # (3,)
            vel,           # (3,)
            ang_vel,       # (3,)
            rel_target,    # (3,)
            dist,          # (1,)
            last_act_rpy,  # (3,)
        ]).astype(np.float32)  # tổng 19 chiều

    # ── Action hook ───────────────────────────────────────────────────────────

    def _preprocessAction(self, action):
        flat_act          = action.flatten()
        self._prev_action = self._last_action.copy()
        self._last_action = flat_act.copy()
        return super()._preprocessAction(action)

    # ── Reward Function ───────────────────────────────────────────────────────

    def _computeReward(self):
        """
        Reward có trọng số theo thứ bậc ưu tiên:

        TẦNG 1 (w=8.0)  r_nav    — tiếp cận mục tiêu, LINEAR gradient
                                    không dùng exponential vì gradient chết ở xa
        TẦNG 2 (w=10.0) r_close  — bonus khi đến rất gần, exponential
                                    tạo "peak" rõ ràng để agent biết đây là đích
        TẦNG 3 (w=3.0)  r_vel    — phạt tốc độ, scale theo proximity_factor
                                    khi xa được bay nhanh, khi gần phải chậm lại
                                    — đây là điều kiện BẮT BUỘC để hover
        TẦNG 4 (w=1.5)  r_att    — phạt góc nghiêng, trung bình
                                    không quá lớn vì agent cần nghiêng để di chuyển
        TẦNG 5 (w=0.3)  r_act    — phạt action lớn, nhỏ thôi
                                    nếu lớn hơn r_nav agent sẽ học cách đứng yên
        TẦNG 6 (w=0.2)  r_angvel — phạt xoay thân, rất nhỏ

        KHÔNG có r_alive:
            r_alive tạo incentive sai — agent được thưởng chỉ vì tồn tại
            dù không tiếp cận target, dẫn đến behavior "lơ lửng tại chỗ"

        Kiểm tra magnitude tại các điểm quan trọng:
            dist=1.5m → r_nav đóng góp -12.0 (dominant) → agent tập trung bay
            dist=0.05m → r_close đóng góp +6.07 (dominant) → agent tập trung hover
        """
        s       = self._getDroneStateVector(0)
        pos     = s[0:3]
        rpy     = s[7:10]
        vel     = s[10:13]
        ang_vel = s[13:16]

        dist  = float(np.linalg.norm(self.TARGET_POS - pos))
        speed = float(np.linalg.norm(vel))

        # ── Tầng 1: Navigation (linear, luôn có gradient dù ở xa) ────────────
        w_nav = 8.0
        r_nav = -dist                           # ∈ (-∞, 0]

        # ── Tầng 2: Close bonus (exponential, mạnh khi dist < 0.3m) ──────────
        w_close = 10.0
        r_close = np.exp(-dist * 10.0)          # ∈ [0, 1]

        # ── Tầng 3: Velocity penalty (scale theo proximity) ───────────────────
        # proximity_factor ≈ 1 khi gần, ≈ 0 khi xa
        # Khi xa: agent được phép bay nhanh để tiếp cận
        # Khi gần: agent bị phạt nặng nếu còn đang bay nhanh
        w_vel = 3.0
        proximity_factor = np.exp(-dist * 3.0)  # ∈ [0, 1]
        r_vel = -speed * proximity_factor        # ∈ (-∞, 0]

        # ── Tầng 4: Attitude penalty ──────────────────────────────────────────
        w_att = 1.5
        r_att = -float(np.linalg.norm(rpy[:2]))  # roll + pitch, ∈ (-π, 0]

        # ── Tầng 5: Action smoothness ─────────────────────────────────────────
        w_act = 0.3
        r_act = -float(np.linalg.norm(self._last_action))  # ∈ (-∞, 0]

        # ── Tầng 6: Angular velocity ──────────────────────────────────────────
        w_angvel = 0.2
        r_angvel = -float(np.linalg.norm(ang_vel))  # ∈ (-∞, 0]

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
        crash = abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3  # ~75 độ

        return bool(out_of_bounds or crash)

    def _computeTruncated(self):
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        s    = self._getDroneStateVector(0)
        pos  = s[0:3]
        rpy  = s[7:10]
        vel  = s[10:13]
        dist = float(np.linalg.norm(self.TARGET_POS - pos))
 
        # is_crashed: kết thúc do out-of-bounds hoặc lật (terminated=True)
        out_of_bounds = (
            pos[2] < 0.05 or pos[2] > 3.5
            or abs(pos[0]) > 4.0
            or abs(pos[1]) > 4.0
        )
        flipped    = abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3
        is_crashed = bool(out_of_bounds or flipped)
 
        # is_timeout: episode đã đủ MAX_STEPS (truncated=True)
        is_timeout = self._step_n >= self.MAX_STEPS
 
        return {
            "dist":       dist,
            "speed":      float(np.linalg.norm(vel)),
            "is_crashed": is_crashed,   # callback dùng để tính crash_rate
            "is_timeout": is_timeout,   # callback dùng để tính timeout_rate
        }
