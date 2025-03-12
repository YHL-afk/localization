import numpy as np
from collections import deque

def _single_pass_polar_clustering(ranges,
                                  angles,
                                  radius_threshold=10,
                                  min_cluster_size=5,
                                  max_cluster_size=14,
                                  dist_min=50,    
                                  dist_max=280,   
                                  min_separation_distance=200.0
                                  ):

    valid_mask = (ranges >= dist_min) & (ranges <= dist_max)
    ranges_f = ranges[valid_mask]
    angles_f = angles[valid_mask]

    if len(ranges_f) == 0:
        print("No valid points; all have been filtered out.")
        return (False, (np.empty((0,2)),
                        np.array([], dtype=int),
                        [None, None],
                        [None, None]))


    x = ranges_f * np.cos(angles_f)
    y = ranges_f * np.sin(angles_f)

    valid_mask_xy = (x >= 20) & (y >= -235) & (y <= 235)
    x = x[valid_mask_xy]
    y = y[valid_mask_xy]
    ranges_f = ranges_f[valid_mask_xy]
    angles_f = angles_f[valid_mask_xy]

    if len(x) == 0:
        print("No valid points; all have been filtered out by x/y conditions.")
        return (False, (np.empty((0, 2)),
                        np.array([], dtype=int),
                        [None, None],
                        [None, None]))

    points = np.column_stack((x, y))
    N = len(points)

    labels = -1 * np.ones(N, dtype=int)
    bounding_boxes_assigned = [None, None]
    target_centers = [None, None]

    
    visited = np.zeros(N, dtype=bool)
    cluster_id = 0
    clusters_info = []  

    def bfs_local(start_idx, cid):
        queue = deque([start_idx])
        cluster_indices = [start_idx]
        visited[start_idx] = True
        labels[start_idx] = cid

        while queue:
            cur = queue.popleft()
            dist = np.linalg.norm(points[cur] - points, axis=1)
            neighbors = np.where((dist <= radius_threshold) & (~visited))[0]
            for nb in neighbors:
                visited[nb] = True
                labels[nb] = cid
                queue.append(nb)
                cluster_indices.append(nb)
        return cluster_indices

    for i in range(N):
        if not visited[i]:
            cidx = bfs_local(i, cluster_id)
            cluster_id += 1

            idx_min = np.argmin(points[cidx, 0])
            cx = points[cidx, 0][idx_min]
            cy_mean = points[cidx, 1][idx_min]
            y_max = np.max(points[cidx, 1])
            y_min = np.min(points[cidx, 1])
            y_mean = y_max - y_min

            avg_range = np.mean(ranges_f[cidx])

            if 150 < avg_range < 200:
                y_mean = y_mean * 1.5
            elif 200 < avg_range < 260:
                y_mean = y_mean * 1.8
            elif 30 < avg_range < 80:
                y_mean = y_mean * 0.8

            if y_mean > 6.5:
                cy = cy_mean
            else:
                if cy_mean> 0:
                    cy = cy_mean + 6.5
                else:
                     cy = cy_mean - 6.5
            clusters_info.append((cluster_id - 1, cidx, cx, cy))

            clusters_info.append((cluster_id - 1, cidx, cx, cy))



    valid_clusters = []
    for cid, cidx, cx, cy in clusters_info:
        csize = len(cidx)

        avg_range = np.mean(ranges_f[cidx])

        if 150 < avg_range < 200:
            effective_csize = csize * 1.5
        elif 200 < avg_range < 260:
            effective_csize = csize * 1.8
        elif 30 < avg_range < 80:
            effective_csize = csize * 0.8
        else:
            effective_csize = csize

        if min_cluster_size <= effective_csize <= max_cluster_size:
            valid_clusters.append((cid, cidx, cx, cy))
        else:

            labels[cidx] = -1

    if len(valid_clusters) < 2:
        labels[:] = -1
        return (False, (points, labels, bounding_boxes_assigned, target_centers))


    valid_clusters_sorted = sorted(valid_clusters, key=lambda t: t[2])
    found_pair = False
    pair_indices = None
    num_valid = len(valid_clusters_sorted)

    for i in range(num_valid):
        for j in range(i+1, num_valid):
            _, _, cx1, cy1 = valid_clusters_sorted[i]
            _, _, cx2, cy2 = valid_clusters_sorted[j]
            dist = np.linalg.norm([cx1 - cx2, cy1 - cy2])
            if dist >= min_separation_distance:
                pair_indices = (i, j)
                found_pair = True
                break
        if found_pair:
            break

    if not found_pair:
        labels[:] = -1
        return (False, (points, labels, bounding_boxes_assigned, target_centers))

    i_sel, j_sel = pair_indices
    cluster1 = valid_clusters_sorted[i_sel]  
    cluster2 = valid_clusters_sorted[j_sel]


    labels[:] = -1

    for lb_idx, cluster in enumerate([cluster1, cluster2]):
        cid, c_indices, cx, cy = cluster
        labels[c_indices] = lb_idx

        half_size = 6.5
        bottom_left = (cx - half_size, cy)
        top_right = (cx + half_size, cy + 13)
        bounding_boxes_assigned[lb_idx] = (bottom_left, top_right)

        center_x = cx
        center_y = cy
        target_centers[lb_idx] = (center_x, center_y)

    return (True, (points, labels, bounding_boxes_assigned, target_centers))


def polar_clustering_until_two_valid(ranges,
                                     angles,
                                     radius_threshold=10,
                                     min_cluster_size=8,
                                     max_cluster_size=13,
                                     dist_min=100,
                                     dist_max=300,
                                     min_separation_distance=200.0,
                                     max_retries=10
                                     ):

    retry_count = 0
    while True:
        retry_count += 1
        found_two, result = _single_pass_polar_clustering(
            ranges, angles,
            radius_threshold=radius_threshold,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            dist_min=dist_min,
            dist_max=dist_max,
            min_separation_distance=min_separation_distance
        )

        if found_two:
            return result  
        else:
            print(f"Try {retry_count}: Not found 2 valid targets, retrying...")

        if retry_count >= max_retries:
            print("Max retries reached, still not found 2 valid targets.")
            return result  

