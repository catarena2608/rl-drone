"""
test_stage2.py
=====================
Chương trình kiểm thử và đánh giá độc lập cho STAGE 2 — Navigation & Robust Hovering.
Chạy mô hình đã huấn luyện trong môi trường đồ họa, hỗ trợ tùy chỉnh bán kính thử thách.

Cách dùng:
    python test_agent2.py --gui                   # Chạy test mặc định với giao diện đồ họa
    python test_agent2.py --gui --radius 1.2      # Thử thách Drone xuất phát từ khoảng cách xa 1.2m
    python test_agent2.py --episodes 20           # Chạy đánh giá nhanh 20 episodes không GUI
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import numpy as np
from stable_baselines3 import PPO

# Import môi trường Stage 2
from nav_aviary import Stage2NavAviary

MODEL_DIR = "./models/stage2"

def test_stage2(gui=True, n_episodes=10, custom_radius=None):
    print(f"\n{'='*60}\n  EVALUATING: STAGE 2 NAVIGATION & ROBUST HOVERING\n{'='*60}\n")
    
    # Khởi tạo môi trường đơn để test độc lập
    env = Stage2NavAviary(gui=gui)
    
    # Định cấu hình bán kính spawn để kiểm thử
    if custom_radius is not None:
        env.spawn_radius = custom_radius
        print(f"[TEST CONFIG] Sử dụng bán kính thử thách tùy chỉnh: {env.spawn_radius:.2f} m")
    else:
        print(f"[TEST CONFIG] Sử dụng bán kính mặc định của môi trường: {env.spawn_radius:.2f} m")

    # Tự động tìm và load trọng số tốt nhất (hoặc final nếu không có best)
    model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
        if not os.path.exists(model_path):
            print(f"❌ Error: Không tìm thấy mô hình đã huấn luyện tại {MODEL_DIR}.")
            print("Vui lòng chạy huấn luyện trước bằng câu lệnh: python train_stage2.py")
            env.close()
            return

    print(f" -> Đang tải trọng số mô hình từ: {model_path}\n")
    model = PPO.load(model_path, env=env)

    episode_results = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0
        dist_history = []

        while not done:
            # Predict với chính sách deterministic=True để đạt độ ổn định cao nhất khi test
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            step_count += 1
            
            # Lấy khoảng cách tới đích từ hàm info để theo dõi hành trình
            dist = info.get("dist", None)
            if dist is not None:
                dist_history.append(dist)
                
            done = terminated or truncated

        # Tính toán các chỉ số chất lượng cho tập (Episode) này
        mean_dist = np.mean(dist_history) if dist_history else 0.0
        final_dist = dist_history[-1] if dist_history else 0.0
        reward_density = ep_reward / step_count if step_count > 0 else 0.0
        
        # Đánh giá trạng thái kết thúc
        status = "CRASHED 💥" if step_count < env.MAX_STEPS else "COMPLETED 🟢"

        episode_results.append({
            "reward": ep_reward,
            "length": step_count,
            "mean_dist": mean_dist,
            "final_dist": final_dist,
            "density": reward_density,
            "success": step_count == env.MAX_STEPS
        })
        
        print(f" Episode {ep+1:02d} | {status} | Steps: {step_count:>4d} | "
              f"Reward: {ep_reward:>7.1f} | Density: {reward_density:.2f} | "
              f"Dist (Mean/Final): {mean_dist:.2f}m / {final_dist:.2f}m")

    # --- TỔNG HỢP VÀ THỐNG KÊ TOÀN CỤC ---
    rews = [r["reward"] for r in episode_results]
    lens = [r["length"] for r in episode_results]
    dists = [r["mean_dist"] for r in episode_results]
    densities = [r["density"] for r in episode_results]
    success_rate = np.mean([1 if r["success"] else 0 for r in episode_results]) * 100

    print(f"\n{'='*60}\n  BÁO CÁO THỐNG KÊ THỬ NGHIỆM (Sau {n_episodes} Episodes)\n{'='*60}")
    print(f" 🎯 Tỷ lệ bay hết thời gian (Success Rate): {success_rate:.1f}%")
    print(f" 📊 Chiều dài tập (Mean Ep Length):       {np.mean(lens):.1f} ± {np.std(lens):.1f} steps")
    print(f" 💰 Tổng điểm thưởng (Mean Reward):      {np.mean(rews):.1f} ± {np.std(rews):.1f}")
    print(f" 📈 Mật độ điểm (Mean Reward Density):    {np.mean(densities):.2f} điểm/step")
    print(f" 🗺️ Khoảng cách tới đích trung bình:     {np.mean(dists):.3f} m")
    print(f"{'='*60}\n")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chương trình kiểm thử Stage 2 độc lập.")
    parser.add_argument("--gui", action="store_true", help="Bật giao diện mô phỏng 3D PyBullet")
    parser.add_argument("--episodes", type=int, default=10, help="Số episodes chạy kiểm thử (Mặc định: 10)")
    parser.add_argument("--radius", type=float, default=None, help="Bán kính xuất phát tùy chỉnh để test độ bền (Ví dụ: 0.5, 1.0, 1.5)")
    args = parser.parse_args()

    test_stage2(gui=args.gui, n_episodes=args.episodes, custom_radius=args.radius)