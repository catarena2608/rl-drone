"""
train_stage2.py
=====================
Huấn luyện thuật toán PPO cho bài toán STAGE 2 — Navigation & Robust Hovering
Dựa trên cấu trúc chuẩn của file train.py (Stage 1).

Cập nhật tính năng:
    - Tự động kiểm tra và LOAD mô hình cũ để TRAIN TIẾP (Resume Training).
    - Giữ nguyên tiến trình đồ thị trên Tensorboard không bị reset về 0.
    - Hỗ trợ flag --fresh để ép buộc train lại từ đầu nếu muốn.
    - Tích hợp thêm chế độ Đánh giá (Evaluation) qua flag --eval.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, BaseCallback
)

# Import môi trường và bộ giám sát độ khó
from backup.nav_aviary import Stage2NavAviary
from backup.curriculum_callback import CurriculumCallback

# Cấu hình đường dẫn lưu trữ dữ liệu huấn luyện của Stage 2
LOG_DIR = "./logs/stage2"
MODEL_DIR = "./models/stage2"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# Console Logging Callback (Giữ nguyên cấu trúc của Stage 1)
# =============================================================================

class ConsoleLogCallbackStage2(BaseCallback):
    """
    In ra console sau mỗi log_interval steps:
        ep_len_mean   — độ dài episode trung bình
        ep_rew_mean   — reward trung bình
        dist_to_target — khoảng cách đến target
        speed_mean    — tốc độ trung bình (THÊM MỚI)
        crash_rate    — % episodes kết thúc do crash (THÊM MỚI)
        timeout_rate  — % episodes kết thúc do hết giờ (THÊM MỚI)
        spawn_radius  — bán kính spawn hiện tại
        
    crash_rate + timeout_rate + (các lý do khác) = 100%
    Nếu crash_rate cao: agent mất ổn định → xem lại reward att/vel
    Nếu timeout_rate thấp: agent không đến được target → xem lại reward nav
    """
    def __init__(self, log_interval: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._train_start = time.time()
 
        # Buffer tích lũy trong episode hiện tại
        self._ep_reward_buf = 0.0
        self._ep_length_buf = 0
        self._ep_dist_buf   = []
        self._ep_speed_buf  = []
 
        # Lịch sử các episodes đã hoàn thành
        self._ep_rewards   = []
        self._ep_lengths   = []
        self._ep_dists     = []
        self._ep_speeds    = []
        self._ep_crashes   = []   # True nếu episode kết thúc do crash
        self._ep_timeouts  = []   # True nếu episode kết thúc do timeout
 
    def _on_step(self) -> bool:
        reward      = self.locals["rewards"][0]
        info        = self.locals["infos"][0]
        done        = self.locals["dones"][0]
 
        self._ep_reward_buf += float(reward)
        self._ep_length_buf += 1
 
        dist  = info.get("dist",  None)
        speed = info.get("speed", None)
 
        if dist  is not None: self._ep_dist_buf.append(float(dist))
        if speed is not None: self._ep_speed_buf.append(float(speed))
 
        if done:
            # SB3 VecEnv lưu terminal info trong "terminal_observation"
            # is_crashed và is_timeout được set từ _computeInfo() trong nav_aviary
            is_crashed  = info.get("is_crashed",  False)
            is_timeout  = info.get("is_timeout",  False)
 
            self._ep_rewards.append(self._ep_reward_buf)
            self._ep_lengths.append(self._ep_length_buf)
            self._ep_dists.append(
                float(np.mean(self._ep_dist_buf))  if self._ep_dist_buf  else 0.0
            )
            self._ep_speeds.append(
                float(np.mean(self._ep_speed_buf)) if self._ep_speed_buf else 0.0
            )
            self._ep_crashes.append(bool(is_crashed))
            self._ep_timeouts.append(bool(is_timeout))
 
            # Reset buffer
            self._ep_reward_buf = 0.0
            self._ep_length_buf = 0
            self._ep_dist_buf   = []
            self._ep_speed_buf  = []
 
        if self.num_timesteps % self.log_interval == 0 and len(self._ep_rewards) > 0:
            self._print_log()
 
        return True
 
    def _print_log(self):
        elapsed = time.time() - self._train_start
        fps     = int(self.num_timesteps / max(elapsed, 1))
        n_eps   = len(self._ep_rewards)
 
        # Lấy 20 episodes gần nhất để tính mean
        recent_slice = slice(-20, None)
        mean_rew     = np.mean(self._ep_rewards[recent_slice])
        mean_len     = np.mean(self._ep_lengths[recent_slice])
        mean_dist    = np.mean(self._ep_dists[recent_slice])
        mean_speed   = np.mean(self._ep_speeds[recent_slice])
        crash_rate   = np.mean(self._ep_crashes[recent_slice])  * 100
        timeout_rate = np.mean(self._ep_timeouts[recent_slice]) * 100
 
        env            = self.training_env.envs[0].unwrapped
        current_radius = getattr(env, "spawn_radius", 0.1)
 
        bar = "-" * 52
        print(f"\n{bar}")
        print(f"| Stage 2 — Step {self.num_timesteps:>9,}  |  FPS: {fps:>5}  |")
        print(bar)
        print(f"|   ep_len_mean     | {mean_len:>10.1f} steps      |")
        print(f"|   ep_rew_mean     | {mean_rew:>10.4f}            |")
        print(f"|   dist_to_target  | {mean_dist:>10.4f} m          |")
        print(f"|   speed_mean      | {mean_speed:>10.4f} m/s        |")
        print(f"|   crash_rate      | {crash_rate:>9.1f} %           |")
        print(f"|   timeout_rate    | {timeout_rate:>9.1f} %           |")
        print(f"|   spawn_radius    | {current_radius:>10.2f} m          |")
        print(f"|   episodes        | {n_eps:>10}              |")
        print(f"|   elapsed         | {elapsed:>10.0f} s          |")
        print(f"{bar}\n")


# =============================================================================
# Hàm Huấn luyện Stage 2 (Hỗ trợ Train tiếp)
# =============================================================================

def train_stage2(args):
    print(f"\n{'='*50}\n  STARTING TRAINING: STAGE 2 NAVIGATION (STANDARD)\n{'='*50}\n")
    
    env_kwargs = dict(gui=args.gui, record=False)
    
    env      = make_vec_env(Stage2NavAviary, n_envs=1, env_kwargs=env_kwargs)
    eval_env = make_vec_env(Stage2NavAviary, n_envs=1, env_kwargs=env_kwargs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    console_callback    = ConsoleLogCallbackStage2(log_interval=2048)
    curriculum_callback = CurriculumCallback(check_freq=10000)

    # Tính toán tổng số timesteps chạy trong phiên này
    total_timesteps = args.timesteps if args.timesteps is not None else 1000000
    
    # KIỂM TRA MÔ HÌNH CŨ ĐỂ TIẾP TỤC HUẤN LUYỆN
    best_path = os.path.join(MODEL_DIR, "best_model.zip")
    final_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
    
    load_path = None
    if os.path.exists(best_path):
        load_path = best_path
    elif os.path.exists(final_path):
        load_path = final_path

    if load_path is not None and not args.fresh:
        print(f"[RESUME] Tìm thấy mô hình cũ tại: {load_path}")
        print("-> Tiến hành tải trọng số và tiếp tục huấn luyện...")
        
        # Load mô hình cũ lên gán vào môi trường mới
        model = PPO.load(
            load_path,
            env=env,
            device="cpu"  # Giữ nguyên chạy trên CPU theo cấu hình gốc của bạn
        )
        
        # Đồng bộ đường dẫn log cũ cho Tensorboard
        model.tensorboard_log = os.path.join(LOG_DIR, "tb")
    else:
        print("[FRESH] Không tìm thấy mô hình cũ hoặc bạn chọn cấu hình --fresh.")
        print("-> Khởi tạo mô hình PPO mới tinh theo kiến trúc bài báo...")
        
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 512, 256, 128],
                vf=[512, 512, 256, 128],
            ),
            activation_fn=torch.nn.ReLU,
        )
        
        lr_schedule = lambda progress_remaining: 3e-5 + (3e-4 - 3e-5) * progress_remaining

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(LOG_DIR, "tb"),
            verbose=1,
            device="cpu",
        )

    print(f"Thiết bị: {model.device}")
    print(f"Số lượng Timesteps chạy thêm trong lượt này: {total_timesteps:,}\n")
    
    # Thực hiện huấn luyện (reset_num_timesteps=False để giữ liền mạch đồ thị Tensorboard)
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([eval_callback, console_callback, curriculum_callback]),
        tb_log_name="PPO_stage2",
        reset_num_timesteps=False if (load_path is not None and not args.fresh) else True
    )

    # Lưu lại trọng số cuối cùng
    final_model_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
    model.save(final_model_path)
    print(f"Stage 2 Finished. Model saved to: {final_model_path}")
    
    env.close()
    eval_env.close()


# =============================================================================
# Hàm Đánh giá (Evaluation) Stage 2
# =============================================================================

def evaluate_stage2(gui=False, n_episodes=10):
    print(f"\n{'='*50}\n  EVALUATING: STAGE 2 NAVIGATION & ROBUST HOVERING\n{'='*50}\n")
    
    env = Stage2NavAviary(gui=gui)
    
    # Tìm kiếm mô hình tốt nhất để thực hiện đánh giá
    model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
        if not os.path.exists(model_path):
            print(f"Error: Không tìm thấy mô hình đã huấn luyện trong {MODEL_DIR}. Vui lòng chạy train trước.")
            env.close()
            return

    print(f"Loading mô hình từ: {model_path}")
    model = PPO.load(model_path, env=env)
    results = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1
            done = terminated or truncated

        results.append({"reward": ep_reward, "final_dist": info.get("dist", 0.0), "steps": step_count})
        print(f" Episode {ep+1:02d}: Reward = {ep_reward:.1f}, Khoảng cách đích cuối = {info.get('dist', 0.0):.3f}m (Steps: {step_count})")

    rews = [r["reward"] for r in results]
    dists = [r["final_dist"] for r in results]
    
    print(f"\n{'='*50}")
    print(f"  Mean reward:               {np.mean(rews):.1f} ± {np.std(rews):.1f}")
    print(f"  Mean final dist to target: {np.mean(dists):.3f} m")
    print(f"{'='*50}\n")
    
    env.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Resume PPO Agent for Stage 2 Navigation task.")
    parser.add_argument("--eval",       action="store_true",       help="Chỉ chạy chế độ đánh giá, không huấn luyện")
    parser.add_argument("--gui",        action="store_true",       help="Bật giao diện PyBullet 3D")
    parser.add_argument("--fresh",      action="store_true",       help="Ép buộc xóa bỏ mô hình cũ để train lại từ đầu")
    parser.add_argument("--episodes",   type=int, default=1000,    help="Số Episodes (dùng khi không truyền --timesteps)")
    parser.add_argument("--timesteps",  type=int, default=None,    help="Tổng số timesteps train (Ưu tiên hơn --episodes nếu được truyền)")
    args = parser.parse_args()

    if args.eval:
        evaluate_stage2(gui=args.gui, n_episodes=args.episodes)
    else:
        train_stage2(args)