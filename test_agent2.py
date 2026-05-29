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
import matplotlib.pyplot as plt

# Import môi trường Stage 2
from nav_aviary import Stage2NavAviary

MODEL_DIR = "./models/stage2"

def test_stage2(gui=True, n_episodes=5, custom_radius=None):
    print(f"\n{'='*60}\n  EVALUATING: STAGE 2 NAVIGATION & ROBUST HOVERING\n{'='*60}\n")
    
    env = Stage2NavAviary(gui=gui)
    
    if custom_radius is not None:
        env.spawn_radius = custom_radius
        print(f"🎮 Thử thách tùy chỉnh: Spawn Radius = {custom_radius}m")
    else:
        print(f"📦 Test với Spawn Radius hiện tại của môi trường: {env.spawn_radius}m")

    # Tìm checkpoint mới nhất
    model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "latest_model.zip")
        
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình tại {MODEL_DIR}! Vui lòng kiểm tra lại.")
        env.close()
        return

    print(f"🤖 Đang nạp mô hình: {model_path}")
    model = PPO.load(model_path, env=env)

    episode_results = []
    
    # Khởi tạo mảng lưu dữ liệu để vẽ đồ thị cho Episode đầu tiên (để phân tích sâu)
    first_ep_history = {"steps": [], "dist_goal": [], "speed": []}

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        
        # Các biến đếm để tính toán độ chuẩn xác Hover
        hover_steps = 0
        total_dist_to_goal = []
        total_speed = []

        # TỰ ĐỘNG ĐỔI GOAL GIỮA CHỪNG (Test tính năng Dynamic Goal của Stage 2)
        # Ví dụ: Ở step thứ 300, ép drone phải bay sang một vị trí hoàn toàn mới
        dynamic_goal_changed = False

        while not done:
            action, _ = model.predict(obs, deterministic=True) # deterministic=True để test chuẩn chính xác
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            step_count += 1
            
            dist_goal = info.get("dist_goal", 1.0)
            speed = info.get("speed", 0.0)
            
            total_dist_to_goal.append(dist_goal)
            total_speed.append(speed)

            # --- KIỂM TRA ĐIỀU KIỆN HOVER CHUẨN XÁC ---
            # Thỏa mãn định nghĩa bài toán: sát đích (< 0.15m) VÀ đứng yên (< 0.15m/s)
            if dist_goal < 0.15 and speed < 0.15:
                hover_steps += 1

            # --- TEST DYNAMIC GOAL API ---
            if step_count == 400 and not dynamic_goal_changed and ep == 0:
                new_goal = np.array([1.0, -1.0, 1.5])
                env.set_new_goal(new_goal)
                dynamic_goal_changed = True
                print(f"⚡ [Dynamic Goal] Đột ngột đổi mục tiêu sang vị trí mới: {new_goal} tại step 400!")

            # Lưu dữ liệu tập đầu tiên để vẽ đồ thị
            if ep == 0:
                first_ep_history["steps"].append(step_count)
                first_ep_history["dist_goal"].append(dist_goal)
                first_ep_history["speed"].append(speed)

            if gui:
                import time
                time.sleep(1/240) # Giới hạn tốc độ hiển thị vật lý thực tế

        # Tính toán metric sau mỗi Episode
        hover_accuracy = (hover_steps / step_count) * 100
        is_success = info.get("at_goal", False) or (hover_steps > 50) # Thành công khi chạm goal hoặc ổn định lâu

        episode_results.append({
            "reward": ep_reward,
            "length": step_count,
            "mean_dist": np.mean(total_dist_to_goal),
            "final_dist": total_dist_to_goal[-1],
            "hover_accuracy": hover_accuracy,
            "success": is_success,
            "crashed": info.get("is_crashed", False)
        })
        
        print(f"Episode {ep+1}: Steps={step_count} | Reward={ep_reward:.1f} | Hover Accuracy={hover_accuracy:.1f}% | Crashed={info.get('is_crashed', False)}")

    # --- BÁO CÁO THỐNG KÊ TOÀN CỤC ---
    print(f"\n{'='*60}\n  BÁO CÁO THỐNG KÊ THỬ NGHIỆM CHUẨN CAO (Sau {n_episodes} Episodes)\n{'='*60}")
    print(f" 🎯 Tỷ lệ bay đạt mục tiêu (Success Rate):  {np.mean([1 if r['success'] else 0 for r in episode_results])*100:.1f}%")
    print(f" 💥 Tỷ lệ va chạm/Rơi tự do (Crash Rate):    {np.mean([1 if r['crashed'] else 0 for r in episode_results])*100:.1f}%")
    print(f" ⏱️ Thời gian giữ vững vị trí (Mean Hover): {np.mean([r['hover_accuracy'] for r in episode_results]):.1f}% tổng thời gian")
    print(f" 🗺️ Khoảng cách tới đích trung bình:        {np.mean([r['mean_dist'] for r in episode_results]):.3f} m")
    print(f" 📉 Khoảng cách khi kết thúc tập:          {np.mean([r['final_dist'] for r in episode_results]):.3f} m")
    print(f"{'='*60}\n")
    
    env.close()

    # --- VẼ ĐỒ THỊ PHÂN TÍCH (Chỉ vẽ cho Episode 0 để xem quỹ đạo phanh) ---
    if len(first_ep_history["steps"]) > 0:
        plt.figure(figsize=(12, 5))
        
        # Đồ thị khoảng cách
        plt.subplot(1, 2, 1)
        plt.plot(first_ep_history["steps"], first_ep_history["dist_goal"], color='blue', label='Khoảng cách tới Đích')
        plt.axhline(y=0.15, color='r', linestyle='--', label='Vùng Target (0.15m)')
        plt.xlabel('Steps')
        plt.ylabel('Distance (m)')
        plt.title('Đồ thị hội tụ Khoảng cách (Ep 1)')
        plt.grid(True)
        plt.legend()

        # Đồ thị vận tốc
        plt.subplot(1, 2, 2)
        plt.plot(first_ep_history["steps"], first_ep_history["speed"], color='orange', label='Vận tốc Drone')
        plt.axhline(y=0.15, color='r', linestyle='--', label='Ngưỡng Hover Yên Lặng (0.15m/s)')
        plt.xlabel('Steps')
        plt.ylabel('Speed (m/s)')
        plt.title('Đồ thị kiểm soát Vận tốc / Phanh (Ep 1)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    test_stage2(gui=args.gui, n_episodes=args.episodes, custom_radius=args.radius)