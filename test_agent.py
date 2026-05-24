"""
test_agent.py
=============
Test và visualize agent Stage 1 đã train xong.

Cách dùng:
    python test_agent.py                        # test nhanh 5 ep, spawn gần (±0.1m)
    python test_agent.py --radius 0.5           # spawn cách target 0.5m
    python test_agent.py --radius 1.5 --gui     # spawn xa 1.5m, bật GUI xem
    python test_agent.py --gui --slow           # chạy chậm 30fps để quan sát
    python test_agent.py --episodes 20          # chạy 20 episodes
    python test_agent.py --model path/to/model.zip  # chỉ định model cụ thể
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from hover_aviary import Stage1HoverAviary

MODEL_DIR = "./models/stage1"
TARGET_POS = np.array([0.0, 0.0, 1.0])


# =============================================================================
# Helpers
# =============================================================================

def find_model(model_arg):
    if model_arg:
        if not os.path.exists(model_arg):
            raise FileNotFoundError(f"Không tìm thấy model: {model_arg}")
        return model_arg
    candidates = [
        os.path.join(MODEL_DIR, "best_model.zip"),
        os.path.join(MODEL_DIR, "model_stage1_final.zip"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Không tìm thấy model trong {MODEL_DIR}.\n"
        "Hãy train trước: python train.py"
    )


def random_spawn(radius: float) -> np.ndarray:
    """
    Sinh vị trí spawn ngẫu nhiên trên mặt cầu bán kính `radius` quanh target.
    Đảm bảo Z >= 0.15 để không spawn dưới đất.
    """
    while True:
        offset = np.random.uniform(-1, 1, size=3)
        # Chuẩn hóa về đúng bán kính rồi thêm noise nhỏ
        offset = offset / max(np.linalg.norm(offset), 1e-6) * radius
        offset *= np.random.uniform(0.7, 1.0)
        pos = TARGET_POS + offset
        if pos[2] >= 0.15:   # tránh spawn dưới đất
            return pos


def print_header(model_path: str, n_episodes: int, radius: float):
    print(f"\n{'='*60}")
    print(f"  TEST AGENT — STAGE 1 HOVER")
    print(f"{'='*60}")
    print(f"  Model        : {model_path}")
    print(f"  Episodes     : {n_episodes}")
    print(f"  Spawn radius : {radius}m từ target {TARGET_POS.tolist()}")
    difficulty = (
        "🟢 Easy   (train distribution)"  if radius <= 0.15 else
        "🟡 Medium (gần ngoài train)"     if radius <= 0.5  else
        "🟠 Hard   (ngoài train dist)"    if radius <= 1.0  else
        "🔴 Extreme (rất xa target)"
    )
    print(f"  Độ khó       : {difficulty}")
    print(f"{'='*60}\n")


def print_episode(ep: int, spawn: np.ndarray, reward: float,
                  steps: int, hover_steps: int, final_dist: float, crashed: bool):
    hover_pct = hover_steps / max(steps, 1) * 100
    status    = "💥 CRASH" if crashed else ("✅ OK" if hover_pct > 50 else "⚠️  DRIFT")
    spawn_str = f"[{spawn[0]:+.2f},{spawn[1]:+.2f},{spawn[2]:+.2f}]"
    print(
        f"  Ep {ep:02d} | {status} | spawn={spawn_str} | "
        f"reward={reward:8.1f} | steps={steps:4d} | "
        f"hover={hover_pct:5.1f}% | dist_final={final_dist:.3f}m"
    )


def print_summary(results: list, radius: float):
    rews    = [r["reward"]      for r in results]
    hovrs   = [r["hover_ratio"] for r in results]
    dists   = [r["final_dist"]  for r in results]
    crashes = sum(1 for r in results if r["crashed"])

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({len(results)} episodes, spawn radius={radius}m)")
    print(f"{'='*60}")
    print(f"  Mean reward      : {np.mean(rews):8.1f} ± {np.std(rews):.1f}")
    print(f"  Best reward      : {np.max(rews):8.1f}")
    print(f"  Mean hover ratio : {np.mean(hovrs)*100:7.1f}%")
    print(f"  Mean final dist  : {np.mean(dists):8.3f} m")
    print(f"  Crashes          : {crashes}/{len(results)}")

    mean_hover = np.mean(hovrs) * 100
    if mean_hover >= 80:
        grade = "🏆 Excellent"
    elif mean_hover >= 50:
        grade = "👍 Good"
    elif mean_hover >= 20:
        grade = "🔧 Fair"
    else:
        grade = "❌ Poor"
    print(f"\n  Đánh giá: {grade} (hover {mean_hover:.1f}%)")
    print(f"{'='*60}\n")


# =============================================================================
# Main test loop
# =============================================================================

def run_test(args):
    model_path = find_model(args.model)
    print_header(model_path, args.episodes, args.radius)

    model = PPO.load(model_path)
    results = []

    for ep in range(1, args.episodes + 1):
        # Sinh spawn mới mỗi episode
        spawn_pos = random_spawn(args.radius)
        spawn_xyz = spawn_pos.reshape(1, 3)

        env = Stage1HoverAviary(gui=args.gui, initial_xyzs=spawn_xyz)
        obs, _ = env.reset()

        done        = False
        ep_reward   = 0.0
        step_count  = 0
        hover_steps = 0
        final_dist  = 0.0
        crashed     = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward  += reward
            step_count += 1
            final_dist  = info.get("dist_to_target", 0.0)

            if info.get("is_hovering", False):
                hover_steps += 1
            if terminated:
                crashed = True

            done = terminated or truncated

            if args.slow and args.gui:
                time.sleep(1 / 30)

        env.close()

        hover_ratio = hover_steps / max(step_count, 1)
        results.append({
            "reward":      ep_reward,
            "steps":       step_count,
            "hover_ratio": hover_ratio,
            "final_dist":  final_dist,
            "crashed":     crashed,
        })
        print_episode(ep, spawn_pos, ep_reward, step_count,
                      hover_steps, final_dist, crashed)

    print_summary(results, args.radius)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Stage 1 Hover Agent")
    parser.add_argument("--model",    type=str,   default=None,
                        help="Đường dẫn model .zip (mặc định: tự tìm best_model)")
    parser.add_argument("--episodes", type=int,   default=5,
                        help="Số episodes test (mặc định: 5)")
    parser.add_argument("--radius",   type=float, default=0.1,
                        help="Bán kính spawn quanh target (mặc định: 0.1m)")
    parser.add_argument("--gui",      action="store_true",
                        help="Bật giao diện PyBullet 3D")
    parser.add_argument("--slow",     action="store_true",
                        help="Chạy chậm 30fps để quan sát (dùng với --gui)")
    args = parser.parse_args()

    run_test(args)