"""
CurriculumCallback cho Stage 2 — Navigation & Robust Hovering
===========================================================================
Tự động nới rộng hoặc thu hẹp spawn_radius dựa trên chất lượng policy hiện tại.

Thay đổi so với phiên bản cũ:
    [ngưỡng]   Reward scale thay đổi hoàn toàn do reward function mới
               Ngưỡng upgrade/downgrade được hiệu chỉnh lại cho phù hợp
    [metric]   Thêm speed metric vào điều kiện upgrade
               Agent phải vừa đến gần target VÀ đang di chuyển chậm
               mới được tăng radius — tránh trường hợp agent bay qua
               target rồi mới dừng (overshoot)
    [bước]     Giảm bước tăng từ 0.15 xuống 0.1 để curriculum mượt hơn
               ở vùng > 0.8m vốn là điểm khó
    [log]      Thêm log speed và dist để dễ debug
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):

    def __init__(self, check_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # Lấy môi trường gốc an toàn
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0].unwrapped
        else:
            env = self.training_env.unwrapped

        if len(self.model.ep_info_buffer) == 0:
            return True

        # ── Lấy metrics từ buffer ──────────────────────────────────────────────
        mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])

        # Lấy state hiện tại để đánh giá chất lượng hover
        state = env._getDroneStateVector(0)
        pos   = state[0:3]
        vel   = state[10:13]

        current_dist  = float(np.linalg.norm(pos - env.TARGET_POS))
        current_speed = float(np.linalg.norm(vel))

        # ── UPGRADE: Tăng radius khi policy đủ tốt ────────────────────────────
        #
        # Điều kiện upgrade được thiết kế chặt hơn phiên bản cũ:
        #   1. mean_reward > ngưỡng reward mới (reward function đã thay đổi scale)
        #   2. mean_length > 1500: agent sống đủ lâu, không crash sớm
        #   3. current_dist < 0.15: agent đang ở gần target
        #   4. current_speed < 0.1: agent đang hover yên, không chỉ bay qua
        #
        # Lý do thêm điều kiện speed:
        #   Phiên bản cũ chỉ check dist — agent có thể đạt dist < 0.1
        #   nhưng vẫn đang bay nhanh qua target (overshoot).
        #   Nếu upgrade lúc này, agent sẽ fail ở radius lớn hơn.
        #
        # Ngưỡng reward mới (~-2.0 đến 0.0 khi hover tốt):
        #   dist=0.05, speed=0.02 → reward ≈ -0.4 + 6.07 - 0.06 - ... ≈ 4.5/step
        #   × 2000 steps ≈ 9000 tổng (nhưng ep_info_buffer lưu cumulative)
        #   Dùng mean_reward > -500 là ngưỡng thực tế hợp lý
        #
        upgrade_conditions = (
            mean_reward > -500         # reward tích lũy đủ tốt
            and mean_length > 1500     # agent sống đủ lâu
            and current_dist  < 0.15  # đang ở gần target
            and current_speed < 0.15  # đang hover yên (THÊM MỚI)
            and env.spawn_radius < 2.0
        )

        if upgrade_conditions:
            old_radius = env.spawn_radius
            # Bước tăng nhỏ hơn (0.1 thay vì 0.15) để curriculum mượt ở vùng > 0.8m
            env.spawn_radius = min(env.spawn_radius + 0.1, 2.0)
            print(f"\n{'='*60}")
            print(f"[CURRICULUM UPGRADE] ✅")
            print(f"  Mean Reward : {mean_reward:.1f}")
            print(f"  Mean Length : {mean_length:.1f} steps")
            print(f"  Dist to target  : {current_dist:.3f} m")
            print(f"  Current speed   : {current_speed:.3f} m/s")
            print(f"  Spawn radius    : {old_radius:.2f} → {env.spawn_radius:.2f} m")
            print(f"{'='*60}\n")

        # ── DOWNGRADE: Giảm radius khi agent fail ─────────────────────────────
        #
        # Ngưỡng downgrade cũng thay đổi theo reward scale mới.
        # Dùng mean_length < 500 thay vì chỉ dùng reward:
        #   Nếu agent crash thường xuyên → mean_length sẽ thấp
        #   Đây là signal rõ ràng hơn reward vì reward có thể âm lớn
        #   chỉ vì agent ở xa target, không nhất thiết vì nó kém.
        #
        elif (mean_length < 500 and env.spawn_radius > 0.15):
            old_radius = env.spawn_radius
            env.spawn_radius = max(env.spawn_radius - 0.1, 0.1)
            print(f"\n{'='*60}")
            print(f"[CURRICULUM DOWNGRADE] ⚠️")
            print(f"  Mean Reward : {mean_reward:.1f}")
            print(f"  Mean Length : {mean_length:.1f} steps  ← crash thường xuyên")
            print(f"  Spawn radius    : {old_radius:.2f} → {env.spawn_radius:.2f} m")
            print(f"{'='*60}\n")

        return True
