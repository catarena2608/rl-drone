"""
train_stage2.py — thêm load spawn_radius khi resume training.

Thay đổi:
    [resume] Sau khi load model cũ, load spawn_radius từ file và
             gán vào env — tránh bị reset về 0.1 mỗi lần chạy lại
    [obs]    Observation space tăng lên 20 chiều (thêm speed_norm)
             → cần --fresh nếu load model cũ có obs 19 chiều
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

from nav_aviary import Stage2NavAviary
from curriculum_callback import CurriculumCallback, load_spawn_radius

LOG_DIR   = "./logs/stage2"
MODEL_DIR = "./models/stage2"
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# Console Logging Callback
# =============================================================================

class ConsoleLogCallbackStage2(BaseCallback):
    def __init__(self, log_interval: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval   = log_interval
        self._train_start   = time.time()
        self._ep_reward_buf = 0.0
        self._ep_length_buf = 0
        self._ep_dist_buf   = []
        self._ep_speed_buf  = []
        self._ep_rewards    = []
        self._ep_lengths    = []
        self._ep_dists      = []
        self._ep_speeds     = []
        self._ep_crashes    = []
        self._ep_timeouts   = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info   = self.locals["infos"][0]
        done   = self.locals["dones"][0]

        self._ep_reward_buf += float(reward)
        self._ep_length_buf += 1

        dist  = info.get("dist",  None)
        speed = info.get("speed", None)
        if dist  is not None: self._ep_dist_buf.append(float(dist))
        if speed is not None: self._ep_speed_buf.append(float(speed))

        if done:
            is_crashed = info.get("is_crashed", False)
            is_timeout = info.get("is_timeout", False)
            self._ep_rewards.append(self._ep_reward_buf)
            self._ep_lengths.append(self._ep_length_buf)
            self._ep_dists.append(float(np.mean(self._ep_dist_buf))  if self._ep_dist_buf  else 0.0)
            self._ep_speeds.append(float(np.mean(self._ep_speed_buf)) if self._ep_speed_buf else 0.0)
            self._ep_crashes.append(bool(is_crashed))
            self._ep_timeouts.append(bool(is_timeout))
            self._ep_reward_buf = 0.0
            self._ep_length_buf = 0
            self._ep_dist_buf   = []
            self._ep_speed_buf  = []

        if self.num_timesteps % self.log_interval == 0 and len(self._ep_rewards) > 0:
            self._print_log()
        return True

    def _print_log(self):
        elapsed      = time.time() - self._train_start
        fps          = int(self.num_timesteps / max(elapsed, 1))
        n_eps        = len(self._ep_rewards)
        s            = slice(-20, None)
        mean_rew     = np.mean(self._ep_rewards[s])
        mean_len     = np.mean(self._ep_lengths[s])
        mean_dist    = np.mean(self._ep_dists[s])
        mean_speed   = np.mean(self._ep_speeds[s])
        crash_rate   = np.mean(self._ep_crashes[s])  * 100
        timeout_rate = np.mean(self._ep_timeouts[s]) * 100
        env          = self.training_env.envs[0].unwrapped
        radius       = getattr(env, "spawn_radius", 0.1)

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
        print(f"|   spawn_radius    | {radius:>10.2f} m          |")
        print(f"|   episodes        | {n_eps:>10}              |")
        print(f"|   elapsed         | {elapsed:>10.0f} s          |")
        print(f"{bar}\n")


# =============================================================================
# Train
# =============================================================================

def train_stage2(args):
    print(f"\n{'='*50}\n  STARTING: STAGE 2 NAVIGATION\n{'='*50}\n")

    env_kwargs = dict(gui=args.gui, record=False)
    env        = make_vec_env(Stage2NavAviary, n_envs=1, env_kwargs=env_kwargs)
    eval_env   = make_vec_env(Stage2NavAviary, n_envs=1, env_kwargs=env_kwargs)

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

    total_timesteps = args.timesteps if args.timesteps is not None else 1_000_000

    best_path  = os.path.join(MODEL_DIR, "best_model.zip")
    final_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
    load_path  = None
    if os.path.exists(best_path):  load_path = best_path
    elif os.path.exists(final_path): load_path = final_path

    if load_path and not args.fresh:
        print(f"[RESUME] Load model từ: {load_path}")
        model = PPO.load(load_path, env=env, device="cpu")
        model.tensorboard_log = os.path.join(LOG_DIR, "tb")
        
        model.clip_range = lambda _: 0.15   # giảm từ 0.2 xuống 0.15
        model.n_epochs   = 5                 # giảm từ 10 xuống 5
        model.learning_rate = lambda _: 1e-4 # flat thay vì schedule

        # FIX BUG 2: Load và áp spawn_radius đã lưu vào env
        saved_radius = load_spawn_radius(default=0.1)
        env.envs[0].unwrapped.spawn_radius     = saved_radius
        eval_env.envs[0].unwrapped.spawn_radius = saved_radius
        print(f"[RESUME] Restored spawn_radius: {saved_radius:.2f} m")

        reset_timesteps = False
    else:
        print("[FRESH] Khởi tạo model mới...")
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128]),
            activation_fn=torch.nn.ReLU,
        )
        lr_schedule = lambda p: 3e-5 + (3e-4 - 3e-5) * p

        model = PPO(
            "MlpPolicy", env,
            learning_rate   = lr_schedule,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 5,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.15,
            ent_coef        = 0.05,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = os.path.join(LOG_DIR, "tb"),
            verbose         = 1,
            device          = "cpu",
        )
        reset_timesteps = True

    print(f"Device: {model.device} | Timesteps: {total_timesteps:,}\n")

    model.learn(
        total_timesteps     = total_timesteps,
        callback            = CallbackList([eval_callback, console_callback, curriculum_callback]),
        tb_log_name         = "PPO_stage2",
        reset_num_timesteps = reset_timesteps,
    )

    save_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
    model.save(save_path)
    print(f"Saved: {save_path}")
    env.close()
    eval_env.close()


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_stage2(gui=False, n_episodes=10):
    env        = Stage2NavAviary(gui=gui)
    model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_stage2_final.zip")
    model   = PPO.load(model_path, env=env)
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward, steps = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1
            done = terminated or truncated
        status = "TIMEOUT 🟢" if steps == env.MAX_STEPS else "CRASHED 💥"
        results.append({"reward": ep_reward, "dist": info.get("dist", 0), "steps": steps})
        print(f"  Ep {ep+1:02d} | {status} | Steps: {steps:>4d} | "
              f"Reward: {ep_reward:.1f} | Dist: {info.get('dist',0):.3f}m")

    print(f"\n  Mean reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Mean dist:   {np.mean([r['dist']   for r in results]):.3f}m")
    env.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--gui",       action="store_true")
    parser.add_argument("--fresh",     action="store_true",
                        help="Train lại từ đầu. Dùng khi obs space thay đổi (16→19→20)")
    parser.add_argument("--episodes",  type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate_stage2(gui=args.gui, n_episodes=args.episodes)
    else:
        train_stage2(args)
