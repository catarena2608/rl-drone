"""
WaypointPlanner — Moving Waypoint Planner
===========================================================================
Tầng 1 của kiến trúc hybrid:
    - Nhận current_pos và goal_pos
    - Trả về waypoint di chuyển liên tục về phía goal
    - RL controller chỉ cần học bay đến waypoint trong LOOKAHEAD radius

Cơ chế:
    waypoint = current_pos + direction * min(dist_to_goal, LOOKAHEAD)

    Khi dist > LOOKAHEAD: waypoint ở phía trước current_pos đúng LOOKAHEAD
    Khi dist < LOOKAHEAD: waypoint = goal (agent đang trong vùng hover)

Lý do dùng moving waypoint thay vì direct goal:
    - RL controller chỉ thấy rel_waypoint nhỏ (≤ LOOKAHEAD = 0.3m)
    - Không bao giờ thấy vector lớn → không bị out-of-distribution
    - Dynamic goal: planner update waypoint ngay bước tiếp theo tự động
"""

import numpy as np


class WaypointPlanner:
    """
    Moving waypoint planner.

    Parameters
    ----------
    lookahead : float
        Khoảng cách tối đa từ current_pos đến waypoint (mét).
        Nên bằng spawn_radius tối đa mà RL controller đã học tốt.
        Default: 0.3m
    goal_threshold : float
        Khi dist_to_goal < threshold thì coi là đã đến goal.
        Default: 0.05m
    """

    def __init__(self, lookahead: float = 0.3, goal_threshold: float = 0.05):
        self.lookahead       = lookahead
        self.goal_threshold  = goal_threshold
        self._goal           = None   # goal hiện tại
        self._waypoint       = None   # waypoint hiện tại

    # ── Public API ────────────────────────────────────────────────────────────

    def set_goal(self, goal: np.ndarray):
        """
        Đặt goal mới. Có thể gọi bất kỳ lúc nào — dynamic goal.
        Waypoint sẽ tự động hướng về goal mới từ bước tiếp theo.
        """
        self._goal = np.array(goal, dtype=np.float64)

    def update(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Tính waypoint mới dựa trên vị trí hiện tại.

        Parameters
        ----------
        current_pos : np.ndarray shape (3,)
            Vị trí hiện tại của drone.

        Returns
        -------
        waypoint : np.ndarray shape (3,)
            Điểm mục tiêu gần nhất mà RL controller cần bay đến.
        """
        if self._goal is None:
            # Chưa có goal: waypoint = current_pos (đứng yên)
            self._waypoint = current_pos.copy()
            return self._waypoint

        pos  = np.array(current_pos, dtype=np.float64)
        goal = self._goal

        delta = goal - pos
        dist  = np.linalg.norm(delta)

        if dist < 1e-6:
            # Đã ở đúng goal
            self._waypoint = goal.copy()
        elif dist <= self.lookahead:
            # Trong vùng LOOKAHEAD: waypoint = goal
            # Agent học hover tại đây
            self._waypoint = goal.copy()
        else:
            # Ngoài vùng LOOKAHEAD: waypoint cách current_pos đúng LOOKAHEAD
            # theo hướng goal
            direction      = delta / dist
            self._waypoint = pos + direction * self.lookahead

        return self._waypoint.copy()

    def is_at_goal(self, current_pos: np.ndarray) -> bool:
        """Kiểm tra drone đã đến goal chưa (dùng cho logic bên ngoài)."""
        if self._goal is None:
            return False
        return float(np.linalg.norm(current_pos - self._goal)) < self.goal_threshold

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy() if self._goal is not None else None

    @property
    def waypoint(self) -> np.ndarray:
        return self._waypoint.copy() if self._waypoint is not None else None
