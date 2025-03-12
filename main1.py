import subprocess
import sys
import json
import numpy as np
import yaml
import matplotlib.pyplot as plt


from plot1 import setup_realtime_plot, update_realtime_plot, draw_radar_position, draw_ekf_objects
from local import triangulate_radar_position
from initinal1 import polar_clustering_until_two_valid
from detection import local_tracking_bfs
from ekf import EKFTracker
from fliter import ParticleFilter 

def main():
    while True:  
        restart = False  
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


                points, labels, bounding_boxes, target_centers = polar_clustering_until_two_valid(
                    ranges, angles,
                    radius_threshold=10, 
                    min_cluster_size=5, 
                    max_cluster_size=14,
                    dist_min=30,
                    dist_max=280,
                    min_separation_distance=200.0
                )


                if target_centers[0] is not None and target_centers[1] is not None:
                    print(f"Initialization successful: {target_centers}")
                    break
            except Exception as e:
                print(f"Initialization error: {e}", file=sys.stderr)

        print("Entering tracking mode...")


        process_noise = np.diag([0.1, 0.1, 0.1, 0.1])
        measurement_noise = np.diag([0.1, 0.1])
        initial_covariance = np.eye(4) * 0.1
        trackers = {}
        next_id = 1

        particle_filter = ParticleFilter()

        fix_false_count = 0  

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


                    lidar_data = np.column_stack((ranges, angles))
                    update_realtime_plot(scatter_lidar, lidar_data, map_metadata)


                    new_centers, labels, bounding_boxes, fix, xy_pos = local_tracking_bfs(points, target_centers)
                    print(f"Detected targets from local_tracking_bfs: {new_centers}")
                    print(f"fix: {fix}")
                    print(f"xy_pos: {xy_pos}")
                    if len(xy_pos) == 2:
                        print(f"xy, pos: {xy_pos[0]}, {xy_pos[1]}")


                    if fix is False:
                        fix_false_count += 1
                    else:
                        fix_false_count = 0


                    if fix_false_count >= 3:
                        print("Detected fix==False three consecutive times. Restarting initialization...")
                        restart = True
                        break

                    for tracker in trackers.values():
                        tracker.predict(dt=0.1)


                    matched = set()
                    for obj_id, tracker in trackers.items():
                        best_match = None
                        best_score = float('inf')

                        for i, center in enumerate(new_centers):
                            if i in matched:
                                continue
                            score = tracker.calculate_association_score({'center': center})
                            if score < best_score and score < 50.0:  
                                best_match = i
                                best_score = score

                        if best_match is not None:
                            tracker.update(np.array(new_centers[best_match]))
                            matched.add(best_match)
                        else:
                            tracker.update()


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


                    for obj_id in list(trackers.keys()):
                        if trackers[obj_id].should_remove():
                            del trackers[obj_id]

                    ekf_tracked_targets = [
                        (float(tracker.state[0]), float(tracker.state[1]))
                        for tracker in trackers.values()
                    ]
                    print(f"EKF tracked targets: {ekf_tracked_targets}")
                    
                    for tracker in trackers.values():
                        speed, direction = tracker.get_velocity_and_direction()
                        vx = speed * np.cos(direction)
                        vy = speed * np.sin(direction)
                        print(f"Tracker {tracker.tracker_id} velocity: vx={vx:.2f}, vy={vy:.2f}, speed={speed:.2f}, direction={direction:.2f} radians")


                    target_centers = ekf_tracked_targets
                    

                    ekf_bounding_boxes = []
                    for (cx, cy) in ekf_tracked_targets:
                        half_size = 6.5
                        bottom_left = (cx - half_size, cy - half_size)
                        top_right = (cx + half_size, cy + half_size)
                        ekf_bounding_boxes.append((bottom_left, top_right))

                    draw_ekf_objects(ax, ekf_bounding_boxes, map_metadata)


                    if len(ekf_tracked_targets) >= 2:
                        local_target1 = ekf_tracked_targets[0]
                        local_target2 = ekf_tracked_targets[1]
                        radar_position = triangulate_radar_position(local_target1, local_target2)

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

        if restart:
            continue
        else:
            break

if __name__ == "__main__":
    main()
