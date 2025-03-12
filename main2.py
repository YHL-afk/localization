import subprocess
import sys
import json
import numpy as np
import yaml
import matplotlib.pyplot as plt

# 假设在 plot1.py 中已经实现了 draw_ekf_objects
from plot1 import setup_realtime_plot, update_realtime_plot, draw_radar_position, draw_ekf_objects
from local import triangulate_radar_position
from initinal1 import polar_clustering_until_two_valid
from detection import local_tracking_bfs
from ekf import EKFTracker
from fliter import ParticleFilter  # 导入粒子滤波器

def main():
    while True:  # 外层循环：每次重新开始整个流程
        restart = False  # 标志位，指示是否需要重新初始化
        print("Running the compiled program...")
        process = subprocess.Popen(
            ["./lds_driver"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        map_image_path = "map.png"
        map_yaml_path = "map.yaml"

        try:
            fig, ax, scatter_lidar, scatter_pose, map_metadata = setup_realtime_plot(
                map_image_path, map_yaml_path
            )
        except FileNotFoundError as e:
            print(f"Error loading map: {e}", file=sys.stderr)
            process.terminate()
            continue
        except yaml.YAMLError as e:
            print(f"Error reading map metadata: {e}", file=sys.stderr)
            process.terminate()
            continue

        print("Waiting for initial radar data...")
        while True:
            line = process.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.strip())
                ranges = np.array(data['ranges']) * 100.0
                angles = np.array(data['angles'])

                # 初始聚类并得到目标点
                points, labels, bounding_boxes, target_centers = polar_clustering_until_two_valid(
                    ranges, angles,
                    radius_threshold=10, 
                    min_cluster_size=5, 
                    max_cluster_size=14,
                    dist_min=30,
                    dist_max=280,
                    min_separation_distance=200.0
                )

                # 如果成功检测到两个目标，就完成初始化
                if target_centers[0] is not None and target_centers[1] is not None:
                    print(f"Initialization successful: {target_centers}")
                    break
            except Exception as e:
                print(f"Initialization error: {e}", file=sys.stderr)

        print("Entering tracking mode...")

        # EKF 参数
        process_noise = np.diag([0.1, 0.1, 0.1, 0.1])
        measurement_noise = np.diag([0.1, 0.1])
        initial_covariance = np.eye(4) * 0.1
        trackers = {}
        next_id = 1
        # 初始化粒子滤波器
        particle_filter = ParticleFilter()

        fix_false_count = 0  # 记录连续 fix 为 False 的次数

        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.strip())
                    ranges = np.array(data['ranges']) * 100.0
                    angles = np.array(data['angles'])
                    points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))

                    # 实时更新雷达散点图
                    lidar_data = np.column_stack((ranges, angles))
                    update_realtime_plot(scatter_lidar, lidar_data, map_metadata)

                    # 本地跟踪
                    new_centers, labels, bounding_boxes, fix = local_tracking_bfs(points, target_centers)
                    print(f"Detected targets from local_tracking_bfs: {new_centers}")
                    print(f"fix: {fix}")

                    # 检查 fix 变量，更新连续 False 的计数
                    if fix is False:
                        fix_false_count += 1
                    else:
                        fix_false_count = 0

                    # 当 fix 连续为 False 三次时，设置 restart 标志，并跳出内层循环
                    if fix_false_count >= 3:
                        print("Detected fix==False three consecutive times. Restarting initialization...")
                        restart = True
                        break

                    # EKF 预测
                    for tracker in trackers.values():
                        tracker.predict(dt=0.1)

                    # 数据关联
                    matched = set()
                    for obj_id, tracker in trackers.items():
                        best_match = None
                        best_score = float('inf')

                        for i, center in enumerate(new_centers):
                            if i in matched:
                                continue
                            score = tracker.calculate_association_score({'center': center})
                            if score < best_score and score < 50.0:  # 关联阈值
                                best_match = i
                                best_score = score

                        if best_match is not None:
                            tracker.update(np.array(new_centers[best_match]))
                            matched.add(best_match)
                        else:
                            # 如果没有匹配到，就只执行预测不更新
                            tracker.update()

                    # 创建新的 EKF 追踪器
                    for i, center in enumerate(new_centers):
                        if i not in matched:
                            initial_state = np.array([*center, 0, 0])
                            tracker_id = next_id
                            new_tracker = EKFTracker(
                                tracker_id,
                                initial_state,
                                initial_covariance,
                                process_noise,
                                measurement_noise
                            )
                            trackers[tracker_id] = new_tracker
                            next_id += 1

                    # 移除未匹配多次的追踪器
                    for obj_id in list(trackers.keys()):
                        if trackers[obj_id].should_remove():
                            del trackers[obj_id]

                    # 得到 EKF 跟踪后的目标中心（与 new_centers 格式相同）
                    ekf_tracked_targets = [
                        (float(tracker.state[0]), float(tracker.state[1]))
                        for tracker in trackers.values()
                    ]
                    print(f"EKF tracked targets: {ekf_tracked_targets}")
                    
                    # 更新 target_centers，供下一帧用
                    target_centers = ekf_tracked_targets
                    
                    # ---------- 绘制目标 ----------
                    ekf_bounding_boxes = []
                    for (cx, cy) in ekf_tracked_targets:
                        half_size = 6.5
                        bottom_left = (cx - half_size, cy)
                        top_right = (cx + half_size, cy + 13)
                        ekf_bounding_boxes.append((bottom_left, top_right))

                    # 绘制蓝色 EKF 框
                    draw_ekf_objects(ax, ekf_bounding_boxes, map_metadata)

                    # 计算雷达在全局坐标系中的位置
                    if len(ekf_tracked_targets) >= 2:
                        local_target1 = ekf_tracked_targets[0]
                        local_target2 = ekf_tracked_targets[1]
                        radar_position = triangulate_radar_position(local_target1, local_target2)

                        # 使用粒子滤波器进行平滑处理
                        particle_filter.predict()
                        particle_filter.update(*radar_position)
                        particle_filter.resample()
                        radar_position = particle_filter.estimate()
                        
                        print(f"Filtered Radar estimated global position: {radar_position}")
                    else:
                        print("Not enough EKF-tracked targets for triangulation.")
                    
                    draw_radar_position(ax, map_metadata, radar_position=radar_position)
                   
                    plt.pause(0.001)

                except Exception as e:
                    print(f"Tracking error: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("Terminating...")
            break
        finally:
            process.terminate()
            process.wait()
            plt.close(fig)

        # 如果 restart 标志为 True，则继续外层循环，重新初始化整个流程
        if restart:
            continue
        else:
            # 如果不是因为 restart 而退出内层循环，则退出整个程序
            break

if __name__ == "__main__":
    main()
