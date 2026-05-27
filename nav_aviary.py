"""
STAGE 2 — Navigation & Robust Hovering với Moving Waypoint Planner
===========================================================================
Kiến trúc hybrid 2 tầng:
    Tầng 1 (WaypointPlanner): toán học thuần túy
        - Nhận goal_pos (random mỗi episode)
        - Tạo waypoint di chuyển liên tục về phía goal
        - Waypoint luôn cách drone tối đa LOOKAHEAD = 0.3m

    Tầng 2 (RL Controller): policy học được
        - Chỉ thấy rel_waypoint (luôn <= 0.3m) — không bao giờ OOD
        - Không quan tâm goal ở đâu, chỉ bay đến waypoint

Tại sao giải quyết được generalize và dynamic goal:
    Generalize:    RL chỉ thấy vector nhỏ <= 0.3m dù goal cách bao xa
    Dynamic goal:  set_new_goal() bất kỳ lúc nào, waypoint tự update

Obs (19,) — tương thích model cũ:
    pos(3)          — vị trí tuyệt đối
    rpy(3)          — góc nghiêng thân
    vel(3)          — vận tốc tuyến tính
    ang_vel(3)      — vận tốc góc
    rel_waypoint(3) — vector từ drone đến WAYPOINT (luôn <= 0.3m)
    dist_wp(1)      — khoảng cách đến waypoint
    last_act(3)     — 3 motor cuối action trước

Thay đổi so với phiên bản cũ:
    [planner]  Thêm WaypointPlanner
    [goal]     TARGET_POS random mỗi episode thay vì cố định [0,0,1]
    [obs]      rel_target -> rel_waypoint
    [reward]   dist dùng dist_wp (luôn <= 0.3m, gradient không chết)
    [info]     Thêm dist_goal, at_goal
    [bounds]   Nới rộng out_of_bounds vì goal có thể ở xa hơn
"""

import os
import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from waypoint_planner import WaypointPlanner

SPAWN_RADIUS_FILE = "./models/stage2/spawn_radius.txt"

# Không gian bay hợp lệ để random goal mỗi episode
GOAL_X_RANGE = (-2.0,  2.0)
GOAL_Y_RANGE = (-2.0,  2.0)
GOAL_Z_RANGE = ( 0.5,  2.0)


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
        self.spawn_radius = _load_spawn_radius(default=0.1)

        self._step_n      = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        # Tầng 1: WaypointPlanner
        # LOOKAHEAD = 0.3m — bán kính RL controller đã học tốt
        self.planner = WaypointPlanner(lookahead=0.3, goal_threshold=0.05)
        self.planner.set_goal(self.TARGET_POS)

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

        # Random goal mới mỗi episode
        self.TARGET_POS = np.array([
            np.random.uniform(*GOAL_X_RANGE),
            np.random.uniform(*GOAL_Y_RANGE),
            np.random.uniform(*GOAL_Z_RANGE),
        ])
        self.planner.set_goal(self.TARGET_POS)

        # Spawn quanh goal trong spawn_radius (curriculum vẫn hoạt động)
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
        19 chiều — tương thích model cũ:
            [0:3]   pos          — vị trí tuyệt đối
            [3:6]   rpy          — roll, pitch, yaw
            [6:9]   vel          — vx, vy, vz
            [9:12]  ang_vel      — p, q, r
            [12:15] rel_waypoint — vector đến waypoint (luôn <= 0.3m)
            [15]    dist_wp      — khoảng cách đến waypoint
            [16:19] last_act     — 3 motor cuối action trước
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

        # Waypoint từ planner — luôn cách drone tối đa LOOKAHEAD
        waypoint     = self.planner.update(pos)
        rel_waypoint = waypoint - pos
        dist_wp      = float(np.linalg.norm(rel_waypoint))

        last_act = self._last_action.flatten()[1:]  # 3 motor cuối

        return np.hstack([
            pos,           # (3,)
            rpy,           # (3,)
            vel,           # (3,)
            ang_vel,       # (3,)
            rel_waypoint,  # (3,) — vector đến waypoint, không phải goal
            [dist_wp],     # (1,)
            last_act,      # (3,)
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
        Reward dùng dist_wp (khoảng cách đến waypoint) thay vì dist đến goal.

        dist_wp luôn <= 0.3m nên:
        - Gradient r_nav không bao giờ chết dù goal xa bao nhiêu
        - r_close luôn có tác dụng rõ ràng
        - Agent học hover tại waypoint, tự nhiên hover tại goal
          khi waypoint trùng goal (lúc dist_to_goal < LOOKAHEAD)
        """
        s       = self._getDroneStateVector(0)
        pos     = s[0:3]
        rpy     = s[7:10]
        vel     = s[10:13]
        ang_vel = s[13:16]

        waypoint = self.planner.waypoint
        if waypoint is None:
            waypoint = self.TARGET_POS

        dist_wp = float(np.linalg.norm(waypoint - pos))
        speed   = float(np.linalg.norm(vel))

        # Tầng 1: Navigation đến waypoint
        w_nav = 4.5
        r_nav = -dist_wp

        # Tầng 2: Close bonus khi gần waypoint
        w_close = 8.0
        r_close = np.exp(-dist_wp * 10.0)

        # Tầng 3: Velocity penalty scale theo proximity đến waypoint
        w_vel            = 4.2
        r_vel            = -speed

        # Tầng 4: Attitude penalty
        w_att = 2.0
        r_att = -float(np.linalg.norm(rpy[:2]))

        # Tầng 5: Action smoothness
        w_act = 0.3
        r_act = -float(np.linalg.norm(self._last_action))

        # Tầng 6: Angular velocity
        w_angvel = 0.5
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

        # Nới rộng bounds vì goal có thể ở bất kỳ đâu trong GOAL_RANGE
        out_of_bounds = (
            pos[2] < 0.05 or pos[2] > 4.0
            or abs(pos[0]) > 5.0
            or abs(pos[1]) > 5.0
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

        waypoint = self.planner.waypoint
        if waypoint is None:
            waypoint = self.TARGET_POS

        dist_wp   = float(np.linalg.norm(waypoint - pos))
        dist_goal = float(np.linalg.norm(self.TARGET_POS - pos))
        speed     = float(np.linalg.norm(vel))

        out_of_bounds = (
            pos[2] < 0.05 or pos[2] > 4.0
            or abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0
        )
        flipped    = abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3
        is_crashed = bool(out_of_bounds or flipped)
        is_timeout = self._step_n >= self.MAX_STEPS

        return {
            "dist":       dist_wp,               # curriculum callback dùng cái này
            "dist_goal":  dist_goal,             # khoảng cách thực đến goal
            "speed":      speed,
            "is_crashed": is_crashed,
            "is_timeout": is_timeout,
            "at_goal":    self.planner.is_at_goal(pos),
        }

    # ── Dynamic Goal API ──────────────────────────────────────────────────────

    def set_new_goal(self, new_goal: np.ndarray):
        """
        Đặt goal mới giữa episode — dynamic goal.
        Planner tự điều chỉnh waypoint từ bước tiếp theo.

        Ví dụ:
            env.set_new_goal(np.array([1.0, 2.0, 1.5]))
        """
        self.TARGET_POS = np.array(new_goal, dtype=np.float64)
        self.planner.set_goal(self.TARGET_POS)
