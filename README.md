# Drone Reinforcement Learning Project (UAV-RL)

Đồ án này triển khai thuật toán học tăng cường sâu (Deep Reinforcement Learning - DRL) để huấn luyện Drone tự hành trong môi trường mô phỏng PyBullet. Dự án tập trung vào hai giai đoạn chính: Hovering (Giữ vị trí) và Navigation (Di chuyển đến đích) sử dụng kiến trúc điều khiển Hybrid.

## 🚀 Tính năng chính

- **Thuật toán PPO (Proximal Policy Optimization):** Sử dụng Stable-Baselines3 để huấn luyện policy điều khiển Drone.
- **Kiến trúc Hybrid 2 tầng (Stage 2):**
    - **Tầng 1 (Waypoint Planner):** Tính toán lộ trình bằng toán học, tạo ra các điểm waypoint trung gian giúp ổn định quá trình bay.
    - **Tầng 2 (RL Controller):** Điều khiển Drone bám theo waypoint một cách linh hoạt và chính xác.
- **Curriculum Learning:** Tự động điều chỉnh độ khó (bán kính spawn quanh mục tiêu) dựa trên hiệu suất của Drone (Reward, Distance, Speed).
- **Resume Training:** Hỗ trợ tiếp tục huấn luyện từ mô hình đã lưu tốt nhất.
- **Console Monitoring:** Hệ thống log chi tiết các chỉ số: Crash rate, Timeout rate, Hover ratio, Spawn radius...

## 📂 Cấu trúc thư mục

- `hover_aviary.py`: Môi trường Stage 1 - Tập trung vào bài toán Hover (giữ vị trí cố định).
- `nav_aviary.py`: Môi trường Stage 2 - Tích hợp Waypoint Planner để Navigation đến mục tiêu ngẫu nhiên.
- `waypoint_planner.py`: Thuật toán toán học điều phối điểm waypoint trung gian.
- `curriculum_callback.py`: Logic tự động tăng/giảm độ khó khi huấn luyện.
- `train.py`: Script huấn luyện cho Stage 1.
- `train_stage2.py`: Script huấn luyện cho Stage 2 (hỗ trợ Resume, Curriculum).
- `test_agent.py` / `test_agent2.py`: Script kiểm tra mô hình đã huấn luyện.
- `models/`: Lưu trữ các trọng số mô hình tốt nhất (`.zip`).
- `logs/`: Lưu trữ log Tensorboard và dữ liệu đánh giá.

## 🛠 Cài đặt

Yêu cầu Python 3.8+ và môi trường Conda.

1. **Khởi tạo môi trường:**
   ```bash
   conda create -n rl-drone python=3.10
   conda activate rl-drone
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install numpy torch gymnasium stable-baselines3 pybullet gym-pybullet-drones
   ```

## 🎮 Hướng dẫn sử dụng

### Giai đoạn 1: Hovering (Stage 1)
Huấn luyện Drone giữ vị trí ổn định tại một điểm cố định.

- **Huấn luyện:**
  ```bash
  python train.py --timesteps 500000
  ```
- **Kiểm tra (GUI):**
  ```bash
  python train.py --eval --gui
  ```

### Giai đoạn 2: Navigation (Stage 2)
Huấn luyện Drone di chuyển đến các đích ngẫu nhiên trong không gian.

- **Huấn luyện (Tiếp tục từ mô hình cũ):**
  ```bash
  python train_stage2.py --timesteps 1000000
  ```
- **Huấn luyện mới hoàn toàn:**
  ```bash
  python train_stage2.py --timesteps 1000000 --fresh
  ```
- **Kiểm tra (GUI):**
  ```bash
  python train_stage2.py --eval --gui
  ```

### Theo dõi quá trình huấn luyện (TensorBoard)
Bạn có thể theo dõi trực quan các chỉ số như `reward`, `loss`, `episode length` qua TensorBoard:

- **Bật TensorBoard cho Stage 1:**
  ```bash
  tensorboard --logdir ./logs/stage1/tb
  ```
- **Bật TensorBoard cho Stage 2:**
  ```bash
  tensorboard --logdir ./logs/stage2/tb
  ```
Sau đó truy cập địa chỉ: `http://localhost:6006/` trên trình duyệt.

## 📊 Kiến trúc Mạng thần kinh (Neural Network)

Theo đặc tả thực nghiệm, cả Actor và Critic đều sử dụng mạng MLP sâu:
- **Input:** State vector (12 chiều cho Stage 1, 19 chiều cho Stage 2).
- **Hidden Layers:** `512 -> 512 -> 256 -> 128` (ReLU).
- **Output:** Action (4 tín hiệu điều khiển động cơ).

## 📝 Ghi chú
- Nếu gặp lỗi `KMP_DUPLICATE_LIB_OK`, file code đã tích hợp sẵn lệnh xử lý môi trường.
- Bán kính spawn (`spawn_radius`) trong Stage 2 được lưu tại `models/stage2/spawn_radius.txt` để đảm bảo tính liên tục khi resume training.
