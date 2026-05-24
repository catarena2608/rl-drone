"""
CurriculumCallback — sửa 2 bug chính:

BUG 1 — Điều kiện upgrade dùng state tức thời:
    Cũ: current_dist và current_speed lấy tại 1 thời điểm ngẫu nhiên
        → nếu check lúc agent đang bay về target thì dist > 0.15 → không upgrade
    Fix: Tự tích lũy mean_dist và mean_speed theo từng episode trong callback

BUG 2 — spawn_radius reset về 0.1 mỗi lần chạy lại:
    Cũ: spawn_radius chỉ tồn tại trong RAM
    Fix: Lưu vào file spawn_radius.txt, load lại khi resume
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

SPAWN_RADIUS_FILE = "./models/stage2/spawn_radius.txt"


def save_spawn_radius(radius: float):
    os.makedirs(os.path.dirname(SPAWN_RADIUS_FILE), exist_ok=True)
    with open(SPAWN_RADIUS_FILE, "w") as f:
        f.write(str(radius))


def load_spawn_radius(default: float = 0.1) -> float:
    if os.path.exists(SPAWN_RADIUS_FILE):
        try:
            with open(SPAWN_RADIUS_FILE, "r") as f:
                val = float(f.read().strip())
            print(f"[CURRICULUM] Loaded spawn_radius: {val:.2f} m")
            return val
        except ValueError:
            pass
    print(f"[CURRICULUM] spawn_radius không tìm thấy, dùng default: {default:.2f} m")
    return default


class CurriculumCallback(BaseCallback):

    def __init__(self, check_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq

        # Buffer tích lũy mean dist/speed theo từng episode
        # (SB3 ep_info_buffer chỉ có 'r' và 'l', không có custom metrics)
        self._ep_dists   = []   # mean dist của các episode đã xong
        self._ep_speeds  = []   # mean speed của các episode đã xong
        self._cur_dists  = []   # tích lũy bước hiện tại
        self._cur_speeds = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        dist  = info.get("dist",  None)
        speed = info.get("speed", None)
        if dist  is not None: self._cur_dists.append(float(dist))
        if speed is not None: self._cur_speeds.append(float(speed))

        if done:
            if self._cur_dists:
                self._ep_dists.append(float(np.mean(self._cur_dists)))
            if self._cur_speeds:
                self._ep_speeds.append(float(np.mean(self._cur_speeds)))
            self._ep_dists  = self._ep_dists[-50:]
            self._ep_speeds = self._ep_speeds[-50:]
            self._cur_dists  = []
            self._cur_speeds = []

        if self.n_calls % self.check_freq != 0:
            return True

        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0].unwrapped
        else:
            env = self.training_env.unwrapped

        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])

        # FIX BUG 1: dùng mean của cả episode, không phải state tức thời
        mean_dist  = float(np.mean(self._ep_dists[-20:]))  if len(self._ep_dists)  >= 3 else 1.0
        mean_speed = float(np.mean(self._ep_speeds[-20:])) if len(self._ep_speeds) >= 3 else 1.0

        upgrade_conditions = (
            mean_reward > -500
            and mean_length > 1500
            and mean_dist  < 0.05    # mean dist cả episode < 5cm
            and mean_speed < 0.05    # mean speed cả episode thấp
            and env.spawn_radius < 2.0
        )

        if upgrade_conditions:
            old_radius = env.spawn_radius
            env.spawn_radius = min(env.spawn_radius + 0.1, 2.0)
            save_spawn_radius(env.spawn_radius)  # FIX BUG 2: persist
            print(f"\n{'='*60}")
            print(f"[CURRICULUM UPGRADE] ✅")
            print(f"  Mean Reward  : {mean_reward:.1f}")
            print(f"  Mean Length  : {mean_length:.1f} steps")
            print(f"  Mean Dist    : {mean_dist:.4f} m")
            print(f"  Mean Speed   : {mean_speed:.4f} m/s")
            print(f"  Spawn radius : {old_radius:.2f} → {env.spawn_radius:.2f} m")
            print(f"{'='*60}\n")

        elif mean_length < 500 and env.spawn_radius > 0.15:
            old_radius = env.spawn_radius
            env.spawn_radius = max(env.spawn_radius - 0.1, 0.1)
            save_spawn_radius(env.spawn_radius)
            print(f"\n{'='*60}")
            print(f"[CURRICULUM DOWNGRADE] ⚠️")
            print(f"  Mean Length  : {mean_length:.1f} steps")
            print(f"  Spawn radius : {old_radius:.2f} → {env.spawn_radius:.2f} m")
            print(f"{'='*60}\n")

        return True
