"""
test_agent.py — Demo GUI nhanh cho Stage 2 Navigation & Robust Hovering
===========================================================================

Bản demo này giải quyết 4 vấn đề:
1. Agent bị chậm khi goal xa: demo dùng waypoint lookahead lớn hơn train một chút.
2. Không bị cắt ở 2000 step: demo có --demo-max-steps riêng.
3. Train vẫn kết thúc sớm, demo thì được nới bounds/step để thử waypoint xa.
4. Dynamic goal được thể hiện rõ bằng đổi goal giữa chừng + marker + vạch trên biểu đồ.

Cách chạy quay video 8 giây:
    python test_agent.py --gui --demo-steps 1400 --target-seconds 6 --plot-hold 2

Nếu vẫn muốn drone chạy nhanh hơn trên màn hình:
    python test_agent.py --gui --demo-steps 1800 --target-seconds 6 --demo-lookahead 0.55
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from nav_aviary import Stage2NavAviary

try:
    import pybullet as p
except Exception:
    p = None


MODEL_DIR = "./models/stage2"


@dataclass
class DemoColors:
    goal: tuple = (1.0, 0.1, 0.1, 1.0)
    new_goal: tuple = (1.0, 0.45, 0.05, 1.0)
    waypoint: tuple = (0.1, 0.4, 1.0, 1.0)
    trail: tuple = (0.0, 0.9, 0.2)
    decision: tuple = (1.0, 0.85, 0.0)
    goal_line: tuple = (1.0, 0.2, 0.2)


class PyBulletDemoOverlay:
    def __init__(self, env, draw_every=3, trail_every=2, trail_keep=260):
        self.env = env
        self.draw_every = max(1, int(draw_every))
        self.trail_every = max(1, int(trail_every))
        self.trail_keep = max(10, int(trail_keep))
        self.colors = DemoColors()
        self.client = getattr(env, "CLIENT", 0)
        self.ids = []
        self.trail_ids = []
        self.prev_pos = None
        self.step = 0
        self.goal_visual_id = None
        self.wp_visual_id = None
        self.dynamic_goal_step = None

    def _safe(self, fn, *args, **kwargs):
        if p is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    def setup_camera(self):
        target = np.array(getattr(self.env, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=float)
        self._safe(
            p.resetDebugVisualizerCamera,
            cameraDistance=1.35,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=target.tolist(),
            physicsClientId=self.client,
        )

    def reset(self):
        self.clear_dynamic()
        self.prev_pos = None
        self.step = 0
        self.setup_camera()
        self._create_goal_and_waypoint_balls()

    def clear_dynamic(self):
        for item_id in self.ids + self.trail_ids:
            self._safe(p.removeUserDebugItem, item_id, physicsClientId=self.client)
        self.ids.clear()
        self.trail_ids.clear()

    def _create_goal_and_waypoint_balls(self):
        if p is None:
            return
        goal = np.array(getattr(self.env, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=float)
        waypoint = self._get_waypoint(goal)

        goal_shape = self._safe(p.createVisualShape, p.GEOM_SPHERE, radius=0.045, rgbaColor=self.colors.goal, physicsClientId=self.client)
        wp_shape = self._safe(p.createVisualShape, p.GEOM_SPHERE, radius=0.035, rgbaColor=self.colors.waypoint, physicsClientId=self.client)

        if goal_shape is not None:
            self.goal_visual_id = self._safe(p.createMultiBody, baseMass=0, baseVisualShapeIndex=goal_shape, basePosition=goal.tolist(), physicsClientId=self.client)
        if wp_shape is not None:
            self.wp_visual_id = self._safe(p.createMultiBody, baseMass=0, baseVisualShapeIndex=wp_shape, basePosition=waypoint.tolist(), physicsClientId=self.client)

    def mark_dynamic_goal(self, step):
        self.dynamic_goal_step = int(step)
        if self.goal_visual_id is not None:
            goal = np.array(getattr(self.env, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=float)
            self._safe(p.resetBasePositionAndOrientation, self.goal_visual_id, goal.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        self.setup_camera()

    def _get_state(self):
        s = self.env._getDroneStateVector(0)
        return np.array(s[0:3], dtype=float), np.array(s[10:13], dtype=float)

    def _get_waypoint(self, fallback):
        waypoint = getattr(self.env.planner, "waypoint", None)
        if waypoint is None:
            return np.array(fallback, dtype=float)
        return np.array(waypoint, dtype=float)

    def update(self, action, info):
        self.step += 1
        if p is None:
            return

        pos, vel = self._get_state()
        goal = np.array(getattr(self.env, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=float)
        waypoint = self._get_waypoint(goal)

        if self.goal_visual_id is not None:
            self._safe(p.resetBasePositionAndOrientation, self.goal_visual_id, goal.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        if self.wp_visual_id is not None:
            self._safe(p.resetBasePositionAndOrientation, self.wp_visual_id, waypoint.tolist(), [0, 0, 0, 1], physicsClientId=self.client)

        if self.step % self.trail_every == 0 and self.prev_pos is not None:
            line_id = self._safe(
                p.addUserDebugLine,
                self.prev_pos.tolist(), pos.tolist(),
                lineColorRGB=self.colors.trail,
                lineWidth=3,
                lifeTime=0,
                physicsClientId=self.client,
            )
            if line_id is not None:
                self.trail_ids.append(line_id)
            while len(self.trail_ids) > self.trail_keep:
                old_id = self.trail_ids.pop(0)
                self._safe(p.removeUserDebugItem, old_id, physicsClientId=self.client)
        self.prev_pos = pos.copy()

        if self.step % self.draw_every != 0:
            return

        for item_id in self.ids:
            self._safe(p.removeUserDebugItem, item_id, physicsClientId=self.client)
        self.ids.clear()

        self.ids.append(self._safe(
            p.addUserDebugLine,
            pos.tolist(), waypoint.tolist(),
            lineColorRGB=self.colors.decision,
            lineWidth=4,
            lifeTime=0,
            physicsClientId=self.client,
        ))
        self.ids.append(self._safe(
            p.addUserDebugLine,
            waypoint.tolist(), goal.tolist(),
            lineColorRGB=self.colors.goal_line,
            lineWidth=2,
            lifeTime=0,
            physicsClientId=self.client,
        ))

        action = np.asarray(action).flatten()
        action_mean = float(np.mean(action)) if action.size else 0.0
        action_std = float(np.std(action)) if action.size else 0.0
        dist_goal = float(info.get("dist_goal", np.linalg.norm(goal - pos)))
        dist_wp = float(info.get("dist", np.linalg.norm(waypoint - pos)))
        speed = float(info.get("speed", np.linalg.norm(vel)))
        lookahead = float(info.get("lookahead", getattr(self.env.planner, "lookahead", 0.0)))
        dynamic_text = "\nDYNAMIC GOAL: ON" if self.dynamic_goal_step is not None and self.step >= self.dynamic_goal_step else ""

        text = (
            "AGENT DECISION\n"
            f"Goal dist: {dist_goal:.2f} m\n"
            f"Waypoint dist: {dist_wp:.2f} m\n"
            f"Lookahead: {lookahead:.2f} m\n"
            f"Speed: {speed:.2f} m/s\n"
            f"Action mean/std: {action_mean:+.2f}/{action_std:.2f}"
            f"{dynamic_text}"
        )
        self.ids.append(self._safe(
            p.addUserDebugText,
            text,
            (pos + np.array([0.0, 0.0, 0.35])).tolist(),
            textColorRGB=(1, 1, 1),
            textSize=1.08,
            lifeTime=0,
            physicsClientId=self.client,
        ))

        self.ids = [x for x in self.ids if x is not None]


def find_model_path():
    candidates = [
        os.path.join(MODEL_DIR, "best_model.zip"),
        os.path.join(MODEL_DIR, "latest_model.zip"),
        os.path.join(MODEL_DIR, "model_stage2_final.zip"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def make_behavior_plot(history, summary_text, hold_seconds=2.0, save_path="demo_behavior_plot.png", dynamic_goal_step=None):
    if len(history["steps"]) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    axes[0].plot(history["steps"], history["dist_goal"], label="Distance to active goal")
    axes[0].plot(history["steps"], history["dist_wp"], label="Distance to waypoint")
    axes[0].axhline(0.15, linestyle="--", label="Hover zone 0.15m")
    axes[0].set_title("Waypoint -> goal behavior")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("m")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history["steps"], history["speed"], label="Speed")
    axes[1].axhline(0.15, linestyle="--", label="Stable hover speed")
    axes[1].set_title("Braking / hover behavior")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("m/s")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(history["steps"], history["action_mean"], label="Action mean")
    axes[2].plot(history["steps"], history["action_std"], label="Action std")
    axes[2].set_title("Policy action behavior")
    axes[2].set_xlabel("Step")
    axes[2].grid(True)
    axes[2].legend()

    if dynamic_goal_step is not None:
        for ax in axes:
            ax.axvline(dynamic_goal_step, linestyle=":", label="Dynamic goal switch")

    fig.suptitle(summary_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)

    plt.show(block=False)
    plt.pause(max(0.1, float(hold_seconds)))
    plt.close(fig)


def run_demo(gui=True,
             radius=1.2,
             demo_steps=1400,
             target_seconds=6.0,
             plot_hold=2.0,
             draw_every=3,
             demo_max_steps=6000,
             demo_lookahead=0.45,
             dynamic_goal_step=650,
             start_pos=None,
             first_goal=None,
             second_goal=None):
    print("\n" + "=" * 74)
    print("STAGE 2 FAST PYBULLET DEMO — WAYPOINT + DYNAMIC GOAL + BEHAVIOR PLOT")
    print("=" * 74)

    env = Stage2NavAviary(
        gui=gui,
        max_steps=demo_max_steps,
        lookahead=demo_lookahead,
        demo_mode=True,
        bounds_xy=10.0,
        bounds_z_max=6.0,
    )
    env.spawn_radius = float(radius)
    env.configure_demo(max_steps=demo_max_steps, lookahead=demo_lookahead, bounds_xy=10.0, bounds_z_max=6.0)

    model_path = find_model_path()
    if model_path is None:
        env.close()
        raise FileNotFoundError(f"Không tìm thấy model trong {MODEL_DIR}")

    if start_pos is None:
        start_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if first_goal is None:
        first_goal = np.array([1.8, -1.2, 1.35], dtype=np.float64)
    if second_goal is None:
        second_goal = np.array([-1.6, 1.3, 1.15], dtype=np.float64)

    print(f"Model: {model_path}")
    print(f"Demo steps: {demo_steps} | Env max steps: {demo_max_steps} | Target animation: {target_seconds:.1f}s")
    print(f"Start: {np.round(start_pos, 2)} | Goal 1: {np.round(first_goal, 2)} | Goal 2: {np.round(second_goal, 2)}")
    print(f"Demo lookahead: {demo_lookahead:.2f}m — tăng nhẹ để agent di chuyển nhanh hơn trong demo")

    model = PPO.load(model_path, env=env)

    obs, info = env.reset(options={
        "target_pos": first_goal,
        "initial_xyz": start_pos,
        "spawn_near_goal": False,
    })

    overlay = PyBulletDemoOverlay(env, draw_every=draw_every, trail_every=2, trail_keep=280)
    if gui:
        overlay.reset()

    history = {
        "steps": [],
        "dist_goal": [],
        "dist_wp": [],
        "speed": [],
        "action_mean": [],
        "action_std": [],
        "reward": [],
    }

    start = time.perf_counter()
    total_reward = 0.0
    final_info = {}
    goal_switched = False

    for step in range(1, int(demo_steps) + 1):
        if (not goal_switched) and dynamic_goal_step is not None and step == int(dynamic_goal_step):
            env.set_new_goal(second_goal)
            goal_switched = True
            if gui:
                overlay.mark_dynamic_goal(step)
            print(f"[DEMO] Dynamic goal switched at step {step}: {np.round(second_goal, 2)}")

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        total_reward += float(reward)

        action_flat = np.asarray(action).flatten()
        history["steps"].append(step)
        history["dist_goal"].append(float(info.get("dist_goal", 0.0)))
        history["dist_wp"].append(float(info.get("dist", 0.0)))
        history["speed"].append(float(info.get("speed", 0.0)))
        history["action_mean"].append(float(np.mean(action_flat)))
        history["action_std"].append(float(np.std(action_flat)))
        history["reward"].append(float(reward))

        if gui:
            overlay.update(action, info)
            expected_elapsed = target_seconds * step / max(1, demo_steps)
            real_elapsed = time.perf_counter() - start
            delay = expected_elapsed - real_elapsed
            if delay > 0:
                time.sleep(min(delay, 0.006))

        if terminated:
            print(f"[DEMO] Kết thúc sớm do crash/flipped ở step {step}.")
            break
        # Cố tình không dừng theo truncated nếu demo_steps nhỏ hơn demo_max_steps.
        if truncated:
            print(f"[DEMO] Chạm demo_max_steps ở step {step}.")
            break

    elapsed = time.perf_counter() - start
    final_dist = float(final_info.get("dist_goal", history["dist_goal"][-1] if history["dist_goal"] else 0.0))
    final_speed = float(final_info.get("speed", history["speed"][-1] if history["speed"] else 0.0))
    hover_frames = sum(d < 0.15 and s < 0.15 for d, s in zip(history["dist_goal"], history["speed"]))
    hover_ratio = hover_frames / max(1, len(history["steps"])) * 100

    summary = (
        f"Steps={len(history['steps'])}, animation={elapsed:.1f}s, "
        f"reward={total_reward:.1f}, final_dist={final_dist:.2f}m, "
        f"final_speed={final_speed:.2f}m/s, hover={hover_ratio:.1f}%"
    )
    print("\n" + summary)
    print("Trail xanh = đường drone đã đi | bi đỏ = active goal | bi xanh dương = waypoint | line vàng = quyết định hiện tại")

    env.close()
    make_behavior_plot(history, summary, hold_seconds=plot_hold, dynamic_goal_step=dynamic_goal_step if goal_switched else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--radius", type=float, default=1.2)
    parser.add_argument("--demo-steps", type=int, default=1400)
    parser.add_argument("--target-seconds", type=float, default=6.0)
    parser.add_argument("--plot-hold", type=float, default=2.0)
    parser.add_argument("--draw-every", type=int, default=3)
    parser.add_argument("--demo-max-steps", type=int, default=6000)
    parser.add_argument("--demo-lookahead", type=float, default=0.45)
    parser.add_argument("--dynamic-goal-step", type=int, default=650)
    args = parser.parse_args()

    run_demo(
        gui=args.gui,
        radius=args.radius,
        demo_steps=args.demo_steps,
        target_seconds=args.target_seconds,
        plot_hold=args.plot_hold,
        draw_every=args.draw_every,
        demo_max_steps=args.demo_max_steps,
        demo_lookahead=args.demo_lookahead,
        dynamic_goal_step=args.dynamic_goal_step,
    )
