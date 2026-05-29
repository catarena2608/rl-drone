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

        # Phải có ít nhất một vài episodes hoàn thành trong buffer để tính toán toán học
        if len(self.model.ep_info_buffer) == 0:
            return True

        # 1. Thu thập dữ liệu thống kê trung bình thực tế từ lịch sử các tập đã qua
        # Stable-baselines3 tự động lưu 'r' (reward) và 'l' (length) trong ep_info_buffer
        mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])

        # Trích xuất 'dist' và 'speed' trung bình (được đẩy từ hàm _computeInfo của nav_aviary vào)
        # Nếu tập đầu chưa kịp ghi nhận thì fallback về giá trị an toàn
        mean_dist = np.mean([ep_info.get("dist", 0.5) for ep_info in self.model.ep_info_buffer])
        mean_speed = np.mean([ep_info.get("speed", 0.5) for ep_info in self.model.ep_info_buffer])

        old_radius = env.spawn_radius

        # 2. Định nghĩa điều kiện nâng cấp (UPGRADE) chuẩn xác bằng giá trị TRUNG BÌNH
        # Thay thế current_dist/current_speed bằng mean_dist/mean_speed
        upgrade_conditions = (
            mean_reward > 3000          # Ngưỡng reward tùy bạn cấu hình
            and mean_length > 1500      # Sống sót lâu (chứng tỏ hover tốt không crash)
            and mean_dist < 0.15        # Khoảng cách trung bình tới target sát sạt
            and mean_speed < 0.15       # Tốc độ trung bình khi tiếp cận phải cực kỳ chậm (phanh chuẩn)
        )

        if upgrade_conditions and env.spawn_radius < 2.0:
            env.spawn_radius = min(env.spawn_radius + 0.1, 2.0)
            
            # ĐỒNG BỘ: Ghi ngay lập tức bán kính mới vào file text để lần sau resume không bị mất
            env.save_spawn_radius() 
            
            print(f"\n{'='*60}")
            print(f"[CURRICULUM UPGRADE] ✅")
            print(f"  Mean Reward    : {mean_reward:.1f}")
            print(f"  Mean Length    : {mean_length:.1f} steps")
            print(f"  Mean Target Dist: {mean_dist:.3f} m")
            print(f"  Mean Speed     : {mean_speed:.3f} m/s")
            print(f"  Spawn radius   : {old_radius:.2f} → {env.spawn_radius:.2f} m")
            print(f"{'='*60}\n")

        # 3. Định nghĩa điều kiện giảm cấp (DOWNGRADE)
        elif mean_length < 500 and env.spawn_radius > 0.15:
            env.spawn_radius = max(env.spawn_radius - 0.1, 0.1)
            
            # ĐỒNG BỘ: Ghi lại file khi hạ độ khó
            env.save_spawn_radius()
            
            print(f"\n{'='*60}")
            print(f"[CURRICULUM DOWNGRADE] ⚠️")
            print(f"  Mean Length    : {mean_length:.1f} steps (Agent crash quá nhiều!)")
            print(f"  Spawn radius   : {old_radius:.2f} → {env.spawn_radius:.2f} m")
            print(f"{'='*60}\n")

        return True
