"""
STAGE 2 — Navigation & Robust Hovering với Moving Waypoint Planner
===========================================================================

Bản này tách rõ TRAIN MODE và DEMO MODE:
- Train: vẫn giữ max_steps=2000 và kết thúc sớm khi crash/out-of-bounds để học ổn định.
- Demo: có thể nới max_steps, nới bounds, đặt start/goal cố định, đổi goal giữa episode.

Ý tưởng chính vẫn giữ nguyên:
    goal xa -> WaypointPlanner tạo waypoint gần -> policy PPO bay đến waypoint.
    Khi goal đổi giữa chừng, planner đổi waypoint ngay bước kế tiếp.
"""

import os
import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from waypoint_planner import WaypointPlanner

SPAWN_RADIUS_FILE = "./models/stage2/spawn_radius.txt"

GOAL_X_RANGE = (-2.0,  2.0)
GOAL_Y_RANGE = (-2.0,  2.0)
GOAL_Z_RANGE = ( 0.5,  2.0)


def _load_spawn_radius(default: float = 0.1) -> float:
    if os.path.exists(SPAWN_RADIUS_FILE):
        try:
            with open(SPAWN_RADIUS_FILE, "r", encoding="utf-8") as f:
                val = float(f.read().strip())
            print(f"[NAV_AVIARY] Loaded spawn_radius: {val:.2f} m")
            return val
        except ValueError:
            pass
    return default


class Stage2NavAviary(BaseRLAviary):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 gui: bool = False,
                 record: bool = False,
                 max_steps: int = 2000,
                 lookahead: float = 0.3,
                 goal_threshold: float = 0.05,
                 demo_mode: bool = False,
                 bounds_xy: float = 5.0,
                 bounds_z_max: float = 4.0):

        self.TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.MAX_STEPS = int(max_steps)
        self.demo_mode = bool(demo_mode)
        self.terminate_on_bounds = not self.demo_mode
        self.bounds_xy = float(bounds_xy)
        self.bounds_z_max = float(bounds_z_max)
        self.spawn_radius = _load_spawn_radius(default=0.1)

        self._step_n = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        self.planner = WaypointPlanner(lookahead=lookahead, goal_threshold=goal_threshold)
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

    def save_spawn_radius(self):
        os.makedirs(os.path.dirname(SPAWN_RADIUS_FILE), exist_ok=True)
        try:
            with open(SPAWN_RADIUS_FILE, "w", encoding="utf-8") as f:
                f.write(str(self.spawn_radius))
        except Exception as e:
            print(f"Lỗi khi lưu {SPAWN_RADIUS_FILE}: {e}")

    def configure_demo(self,
                       max_steps: int = 6000,
                       lookahead: float = 0.45,
                       bounds_xy: float = 10.0,
                       bounds_z_max: float = 6.0,
                       terminate_on_bounds: bool = False):
        """
        Gọi trước env.reset() trong test_agent.py.
        Train không dùng hàm này, nên curriculum và episode 2000 step vẫn giữ nguyên.
        """
        self.demo_mode = True
        self.MAX_STEPS = int(max_steps)
        self.bounds_xy = float(bounds_xy)
        self.bounds_z_max = float(bounds_z_max)
        self.terminate_on_bounds = bool(terminate_on_bounds)
        self.planner.lookahead = float(lookahead)

    def reset(self, seed=None, options=None):
        options = options or {}
        self._step_n = 0
        self._last_action = np.zeros(4)
        self._prev_action = np.zeros(4)

        forced_goal = options.get("target_pos", None)
        forced_start = options.get("initial_xyz", None)
        spawn_near_goal = bool(options.get("spawn_near_goal", True))

        if forced_goal is not None:
            self.TARGET_POS = np.array(forced_goal, dtype=np.float64)
        else:
            self.TARGET_POS = np.array([
                np.random.uniform(*GOAL_X_RANGE),
                np.random.uniform(*GOAL_Y_RANGE),
                np.random.uniform(*GOAL_Z_RANGE),
            ], dtype=np.float64)
        self.planner.set_goal(self.TARGET_POS)

        if forced_start is not None:
            initial_xyz = np.array(forced_start, dtype=np.float64)
        elif spawn_near_goal:
            r = self.spawn_radius
            noise_xy = np.random.uniform(-r, r, size=(2,))
            noise_z = np.random.uniform(-r * 0.4, r * 0.4)
            initial_xyz = self.TARGET_POS.copy()
            initial_xyz[0] += noise_xy[0]
            initial_xyz[1] += noise_xy[1]
            initial_xyz[2] += noise_z
        else:
            initial_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        initial_xyz[2] = max(initial_xyz[2], 0.15)
        self.INIT_XYZS = np.array([initial_xyz])
        return super().reset(seed=seed, options=options)

    def _observationSpace(self):
        return spaces.Box(
            low=np.full(19, -np.inf, dtype=np.float32),
            high=np.full(19, np.inf, dtype=np.float32),
            dtype=np.float32
        )

    def _computeObs(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]
        ang_vel = s[13:16]

        waypoint = self.planner.update(pos)
        rel_waypoint = waypoint - pos
        dist_wp = float(np.linalg.norm(rel_waypoint))
        last_act = self._last_action.flatten()[1:]

        return np.hstack([
            pos,
            rpy,
            vel,
            ang_vel,
            rel_waypoint,
            [dist_wp],
            last_act,
        ]).astype(np.float32)

    def _preprocessAction(self, action):
        flat_act = action.flatten()
        self._prev_action = self._last_action.copy()
        self._last_action = flat_act.copy()
        return super()._preprocessAction(action)

    def _computeReward(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]
        vel = s[10:13]
        ang_vel = s[13:16]

        waypoint = self.planner.waypoint
        if waypoint is None:
            waypoint = self.TARGET_POS

        dist_wp = float(np.linalg.norm(waypoint - pos))
        speed = float(np.linalg.norm(vel))

        w_nav = 4.5
        r_nav = -dist_wp
        w_close = 8.0
        r_close = np.exp(-dist_wp * 10.0)
        w_vel = 4.2
        r_vel = -speed
        w_att = 2.0
        r_att = -float(np.linalg.norm(rpy[:2]))
        w_act = 0.3
        r_act = -float(np.linalg.norm(self._last_action))
        w_angvel = 0.5
        r_angvel = -float(np.linalg.norm(ang_vel))

        total = (w_nav * r_nav +
                 w_close * r_close +
                 w_vel * r_vel +
                 w_att * r_att +
                 w_act * r_act +
                 w_angvel * r_angvel)

        return float(np.clip(total, -20.0, 20.0))

    def _is_out_of_bounds_or_crashed(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        rpy = s[7:10]
        out_of_bounds = (
            pos[2] < 0.05 or pos[2] > self.bounds_z_max
            or abs(pos[0]) > self.bounds_xy
            or abs(pos[1]) > self.bounds_xy
        )
        flipped = abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3
        return bool(out_of_bounds or flipped), bool(out_of_bounds), bool(flipped)

    def _computeTerminated(self):
        is_bad, out_of_bounds, flipped = self._is_out_of_bounds_or_crashed()
        if self.demo_mode and not self.terminate_on_bounds:
            return bool(flipped)
        return bool(is_bad)

    def _computeTruncated(self):
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        vel = s[10:13]

        waypoint = self.planner.waypoint
        if waypoint is None:
            waypoint = self.TARGET_POS

        dist_wp = float(np.linalg.norm(waypoint - pos))
        dist_goal = float(np.linalg.norm(self.TARGET_POS - pos))
        speed = float(np.linalg.norm(vel))
        is_bad, out_of_bounds, flipped = self._is_out_of_bounds_or_crashed()
        is_timeout = self._step_n >= self.MAX_STEPS

        return {
            "dist": dist_wp,
            "dist_wp": dist_wp,
            "dist_goal": dist_goal,
            "speed": speed,
            "is_crashed": bool(is_bad),
            "out_of_bounds": bool(out_of_bounds),
            "flipped": bool(flipped),
            "is_timeout": bool(is_timeout),
            "at_goal": self.planner.is_at_goal(pos),
            "target_pos": self.TARGET_POS.copy(),
            "waypoint": waypoint.copy(),
            "lookahead": float(self.planner.lookahead),
            "demo_mode": bool(self.demo_mode),
        }

    def set_new_goal(self, new_goal: np.ndarray):
        self.TARGET_POS = np.array(new_goal, dtype=np.float64)
        self.planner.set_goal(self.TARGET_POS)
