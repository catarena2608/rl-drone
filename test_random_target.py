"""
test_random_target.py
=====================
Chương trình kiểm thử KIỂM TRA KHẢ NĂNG THÍCH NGHI (Generalization) của Stage 2.
Mỗi Episode hệ thống sẽ sinh một TARGET_POS hoàn toàn ngẫu nhiên và ép Drone phải tự tìm đường đến đó.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import numpy as np
from stable_baselines3 import PPO

# Import môi trường Stage 2 gốc của bạn
from backup.nav_aviary import Stage2NavAviary

MODEL_DIR = "./models/stage2"

def test_random_target(gui=True, n_episodes=10):
    print(f"\n{'='*60}\n  EVALUATING: STAGE 2 WITH RANDOM TARGET POSITIONS\n{'='*60}\n")
    
    # 1. Khởi tạo môi trường
    env = Stage2NavAviary(gui=gui)
    
    # 2. Tự động tìm và load trọng số mô hình tốt nhất
    model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
        if not os.path.exists(model_path):
            print(f"❌ Error: Không tìm thấy mô hình tại {MODEL_DIR}.")
            env.close()
            return

    print(f" -> Đang tải trọng số mô hình từ: {model_path}\n")
    model = PPO.load(model_path, env=env)

    episode_results = []
    
    for ep in range(n_episodes):
        # =====================================================================
        # CAN THIỆP ĐỘNG: Sinh Target ngẫu nhiên cho Episode này trước khi Reset
        # Giới hạn an toàn: X [-1.5m, 1.5m], Y [-1.5m, 1.5m], Z [0.6m, 2.0m]
        # =====================================================================
        random_x = np.random.uniform(-1.5, 1.5)
        random_y = np.random.uniform(-1.5, 1.5)
        random_z = np.random.uniform(0.6, 2.0)
        
        # Ghi đè trực tiếp vào thuộc tính của môi trường
        env.TARGET_POS = np.array([random_x, random_y, random_z])
        
        # Ép bán kính spawn cố định nhỏ (ví dụ 0.2m) quanh cái Target mới 
        # Hoặc nếu bạn muốn Drone spawn cố định tại [0,0,1] và Target bay chỗ khác, hãy chỉnh tại đây
        env.spawn_radius = 0.2 
        
        obs, _ = env.reset()
        
        # Để hỗ trợ vẽ marker mục tiêu trực quan trong PyBullet GUI (Nếu bật GUI)
        if gui and hasattr(env, "CLIENT"):
            # Xóa các debug line cũ nếu có để tránh rác màn hình
            import pybullet as p
            p.removeAllUserDebugItems(physicsClientId=env.CLIENT)
            # Vẽ một chữ X màu đỏ ngay tại vị trí Target ngẫu nhiên mới để bạn dễ quan sát
            p.addUserDebugText("TARGET", env.TARGET_POS, textColorRGB=[1, 0, 0], textSize=1.5, physicsClientId=env.CLIENT)
            p.addUserDebugLine(env.TARGET_POS - np.array([0.1, 0, 0]), env.TARGET_POS + np.array([0.1, 0, 0]), lineColorRGB=[1, 0, 0], lineWidth=3, physicsClientId=env.CLIENT)
            p.addUserDebugLine(env.TARGET_POS - np.array([0, 0.1, 0]), env.TARGET_POS + np.array([0, 0.1, 0]), lineColorRGB=[1, 0, 0], lineWidth=3, physicsClientId=env.CLIENT)

        print(f"🔹 Episode {ep+1:02d} | Mục tiêu mới được cấu hình tại: {env.TARGET_POS}")
        # =====================================================================

        done = False
        ep_reward = 0.0
        step_count = 0
        dist_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            step_count += 1
            
            dist = info.get("dist", None)
            if dist is not None:
                dist_history.append(dist)
                
            done = terminated or truncated

        mean_dist = np.mean(dist_history) if dist_history else 0.0
        final_dist = dist_history[-1] if dist_history else 0.0
        reward_density = ep_reward / step_count if step_count > 0 else 0.0
        
        status = "CRASHED 💥" if step_count < env.MAX_STEPS else "COMPLETED 🟢"

        episode_results.append({
            "reward": ep_reward,
            "length": step_count,
            "mean_dist": mean_dist,
            "final_dist": final_dist,
            "density": reward_density,
            "success": step_count == env.MAX_STEPS
        })
        
        print(f" └─ KẾT QUẢ: {status} | Steps: {step_count:>4d} | Reward: {ep_reward:>7.1f} | Final Dist: {final_dist:.3f}m\n")

    # --- TỔNG HỢP THỐNG KÊ ---
    rews = [r["reward"] for r in episode_results]
    lens = [r["length"] for r in episode_results]
    dists = [r["mean_dist"] for r in episode_results]
    success_rate = np.mean([1 if r["success"] else 0 for r in episode_results]) * 100

    print(f"\n{'='*60}\n  BÁO CÁO THỐNG KÊ THÍCH NGHI (Sau {n_episodes} Episodes Đổi Target)\n{'='*60}")
    print(f" 🎯 Tỷ lệ xử lý mục tiêu thành công:     {success_rate:.1f}%")
    print(f" 📊 Chiều dài tập trung bình (Ep Length): {np.mean(lens):.1f} steps")
    print(f" 💰 Tổng điểm thưởng trung bình:         {np.mean(rews):.1f}")
    print(f" 🗺️ Khoảng cách tới các đích ngẫu nhiên:  {np.mean(dists):.3f} m")
    print(f"{'='*60}\n")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Agent Stage 2 với mục tiêu dịch chuyển ngẫu nhiên.")
    parser.add_argument("--gui", action="store_true", help="Bật giao diện đồ họa PyBullet 3D")
    parser.add_argument("--episodes", type=int, default=10, help="Số lượng episode chạy test")
    args = parser.parse_args()

    test_random_target(gui=args.gui, n_episodes=args.episodes)