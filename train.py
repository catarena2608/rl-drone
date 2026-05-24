"""
train.py
=====================
Huấn luyện thuật toán PPO cho bài toán STAGE 1 — Hover tại chỗ (Theo chuẩn Bài báo)
Sử dụng môi trường Stage1HoverAviary với không gian trạng thái 12 chiều.

Cập nhật cấu trúc Mạng thần kinh (Neural Network) sâu theo đặc tả thực nghiệm bài báo:
    Actor (pi):  512 -> 512 -> 256 -> 128
    Critic (vf): 512 -> 512 -> 256 -> 128

Cách dùng:
    python train.py                 # Huấn luyện mới Stage 1 từ đầu
    python train.py --eval          # Chỉ chạy đánh giá (Evaluation) Stage 1
    python train.py --gui           # Bật giao diện PyBullet khi chạy
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

# Import môi trường Stage 1 Hover đã được cấu hình theo chuẩn bài báo
from hover_aviary import Stage1HoverAviary

# Cấu hình đường dẫn lưu trữ dữ liệu huấn luyện
LOG_DIR = "./logs/stage1"
MODEL_DIR = "./models/stage1"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# Console Logging Callback
# =============================================================================

class ConsoleLogCallback(BaseCallback):
    """
    In ra console các chỉ số quan trọng sau mỗi rollout:
        ep_len_mean, ep_rew_mean, dist_to_target, hover_ratio, fps, elapsed
    Trung bình trên 20 episode gần nhất để tránh nhiễu.
    """

    def __init__(self, log_interval: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._ep_rewards:  list = []
        self._ep_lengths:  list = []
        self._ep_dists:    list = []
        self._ep_hovers:   list = []
        self._ep_steps:    list = []
        # buffer cho episode đang chạy
        self._ep_reward_buf = 0.0
        self._ep_length_buf = 0
        self._ep_dist_buf:  list = []
        self._ep_hover_buf  = 0
        self._train_start   = time.time()

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info   = self.locals["infos"][0]
        done   = self.locals["dones"][0]

        self._ep_reward_buf += float(reward)
        self._ep_length_buf += 1

        dist = info.get("dist_to_target", None)
        if dist is not None:
            self._ep_dist_buf.append(float(dist))

        if info.get("is_hovering", False):
            self._ep_hover_buf += 1

        # Flush khi episode kết thúc
        if done:
            self._ep_rewards.append(self._ep_reward_buf)
            self._ep_lengths.append(self._ep_length_buf)
            self._ep_dists.append(float(np.mean(self._ep_dist_buf)) if self._ep_dist_buf else 0.0)
            self._ep_hovers.append(self._ep_hover_buf)
            self._ep_steps.append(self._ep_length_buf)
            # reset buffer
            self._ep_reward_buf = 0.0
            self._ep_length_buf = 0
            self._ep_dist_buf   = []
            self._ep_hover_buf  = 0

        # In log sau mỗi log_interval steps (nếu đã có ít nhất 1 episode)
        if self.num_timesteps % self.log_interval == 0 and len(self._ep_rewards) > 0:
            self._print_log()

        return True

    def _print_log(self):
        elapsed = time.time() - self._train_start
        fps     = int(self.num_timesteps / max(elapsed, 1))
        n_eps   = len(self._ep_rewards)

        # Trung bình 20 episode gần nhất
        mean_rew  = np.mean(self._ep_rewards[-20:])
        mean_len  = np.mean(self._ep_lengths[-20:])
        mean_dist = np.mean(self._ep_dists[-20:])

        hover_ratios = [
            h / max(s, 1)
            for h, s in zip(self._ep_hovers[-20:], self._ep_steps[-20:])
        ]
        mean_hover = np.mean(hover_ratios) * 100 if hover_ratios else 0.0

        bar = "-" * 48
        print(f"\n{bar}")
        print(f"| Stage 1 — Step {self.num_timesteps:>9,}  |  FPS: {fps:>5}")
        print(bar)
        print(f"|   ep_len_mean     | {mean_len:>10.1f}          |")
        print(f"|   ep_rew_mean     | {mean_rew:>10.4f}          |")
        print(f"|   dist_to_target  | {mean_dist:>10.4f} m        |")
        print(f"|   hover_ratio     | {mean_hover:>10.1f} %        |")
        print(f"|   episodes        | {n_eps:>10}            |")
        print(f"|   elapsed         | {elapsed:>10.0f} s        |")
        print(f"{bar}\n")


# =============================================================================
# Hàm Huấn luyện Stage 1
# =============================================================================

def train_stage1(args):
    print(f"\n{'='*50}\n  STARTING TRAINING: STAGE 1 HOVER (STANDARD PAPER)\n{'='*50}\n")
    
    # Thiết lập tham số môi trường
    env_kwargs = dict(
        gui=args.gui,
        record=False,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
    )
    
    # Tạo môi trường vector hóa (Vectorized Environment)
    env      = make_vec_env(Stage1HoverAviary, n_envs=1, env_kwargs=env_kwargs)
    eval_env = make_vec_env(Stage1HoverAviary, n_envs=1, env_kwargs=env_kwargs)

    # Callback đánh giá định kỳ + console log
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    console_callback = ConsoleLogCallback(log_interval=2048)

    # Cấu hình kiến trúc mạng thần kinh theo chuẩn bài báo
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256, 128],
            vf=[512, 512, 256, 128],
        ),
        activation_fn=torch.nn.ReLU,
    )
    lr_schedule = lambda progress_remaining: 3e-5 + (3e-4 - 3e-5) * progress_remaining

    # Khởi tạo mô hình PPO
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
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        verbose=1,
        device="auto",
    )

    print(f"Thiết bị: {model.device}")
    print("Neural Network Architecture successfully configured via policy_kwargs.")
    print("Training Stage 1...\n")

    # --timesteps ưu tiên hơn --episodes
    total_timesteps = args.timesteps if args.timesteps is not None else args.episodes * 1000
    print(f"Total timesteps: {total_timesteps:,}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([eval_callback, console_callback]),
        tb_log_name="PPO_stage1",
    )

    # Lưu trọng số cuối cùng
    final_model_path = os.path.join(MODEL_DIR, "model_stage1_final.zip")
    model.save(final_model_path)
    print(f"Stage 1 Finished. Model saved to: {final_model_path}")
    
    env.close()
    eval_env.close()


# =============================================================================
# Hàm Đánh giá (Evaluation) Stage 1
# =============================================================================

def evaluate_stage1(gui=False, n_episodes=10):
    print(f"\n{'='*50}\n  EVALUATING: STAGE 1 HOVER (STANDARD PAPER)\n{'='*50}\n")
    
    # FIX: bỏ ctrl_freq/physics_freq vì Stage1HoverAviary không nhận 2 param này
    env = Stage1HoverAviary(gui=gui)
    
    # Tìm và load trọng số tốt nhất
    model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_stage1_final.zip")
        if not os.path.exists(model_path):
            print(f"Error: No trained model found in {MODEL_DIR}. Please train first.")
            env.close()
            return

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)

    results = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done        = False
        ep_reward   = 0.0
        hover_steps = 0
        step_count  = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward  += reward
            step_count += 1
            if info.get("is_hovering", False):
                hover_steps += 1
            done = terminated or truncated

        hover_ratio = hover_steps / step_count if step_count > 0 else 0
        results.append({"reward": ep_reward, "hover_ratio": hover_ratio})
        print(f" Episode {ep+1:02d}: Reward = {ep_reward:.1f}, "
              f"Hover Ratio = {hover_ratio*100:.1f}% (Steps: {step_count})")

    rews  = [r["reward"]      for r in results]
    hovrs = [r["hover_ratio"] for r in results]
    print(f"\n{'='*50}")
    print(f"  Mean reward:      {np.mean(rews):.1f} ± {np.std(rews):.1f}")
    print(f"  Mean hover ratio: {np.mean(hovrs)*100:.1f}%")
    print(f"{'='*50}\n")
    
    env.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent for Stage 1 Hovering task.")
    parser.add_argument("--eval",       action="store_true",       help="Chỉ chạy chế độ đánh giá, không huấn luyện")
    parser.add_argument("--gui",        action="store_true",       help="Bật giao diện PyBullet 3D")
    parser.add_argument("--episodes",   type=int, default=100,     help="Số Episodes (dùng khi không truyền --timesteps)")
    parser.add_argument("--timesteps",  type=int, default=None,    help="Tổng số timesteps train (ưu tiên hơn --episodes nếu truyền vào)")
    args = parser.parse_args()

    if args.eval:
        evaluate_stage1(gui=args.gui, n_episodes=args.episodes)
    else:
        train_stage1(args)
