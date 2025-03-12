import numpy as np
from collections import deque

def local_tracking_bfs(
    points,
    old_centers,
    radius_threshold=10,
    dynamic_radius=40.0,
    min_cluster_size=5,
    max_cluster_size=14,
    min_separation_distance=200.0
):
    

    N = len(points)
    labels = -1 * np.ones(N, dtype=int) 
    bounding_boxes = [None, None]
   
    new_centers = [(cx, cy) for (cx, cy) in old_centers]  
    fix = False  

    
    valid_centers = [(cx, cy) for (cx, cy) in old_centers if cx is not None and cy is not None]
    if len(valid_centers) == 0:
        return new_centers, labels, bounding_boxes, fix, []


    local_point_indices = set()
    for (cx, cy) in valid_centers:
        dists = np.linalg.norm(points - np.array([cx, cy]), axis=1)
        near_idxs = np.where(dists <= dynamic_radius)[0]
        for idx in near_idxs:
            local_point_indices.add(idx)

    if len(local_point_indices) == 0:
        return new_centers, labels, bounding_boxes, fix, []

    local_point_indices = sorted(list(local_point_indices))
    local_points = points[local_point_indices] 


    visited = np.zeros(len(local_points), dtype=bool)
    cluster_id = 0
    
    clusters_info = []

    def bfs_local(start_idx, cid):
        queue = deque([start_idx])
        cluster_indices = [start_idx]
        visited[start_idx] = True

        while queue:
            cur = queue.popleft()
            dist = np.linalg.norm(local_points[cur] - local_points, axis=1)
            neighbors = np.where((dist <= radius_threshold) & (~visited))[0]
            for nb in neighbors:
                visited[nb] = True
                cluster_indices.append(nb)
                queue.append(nb)
        return cluster_indices

    for i in range(len(local_points)):
        if not visited[i]:
            cidx = bfs_local(i, cluster_id)
            cluster_id += 1

            cx_mean = np.mean(local_points[cidx, 0])
            cy_mean = np.mean(local_points[cidx, 1])

            y_extent = np.ptp(local_points[cidx, 1])
            avg_range  = np.linalg.norm([cx_mean, cy_mean])
            if 150 < avg_range < 200:
                y_extent = y_extent * 1.5
            elif 200 < avg_range < 260:
                y_extent = y_extent * 1.8
            elif 30 < avg_range < 80:
                y_extent = y_extent * 0.8

            if y_extent > 6.5:
                xy_pos_val = True
            else:
                xy_pos_val = False
                if cy_mean >0:
                    cy_mean = cy_mean + 6.5
                else:
                    cy_mean = cy_mean - 6.5
            
            clusters_info.append((cluster_id - 1, cidx, cx_mean, cy_mean, xy_pos_val))


    valid_clusters = []
    for cid, cidx, cx_mean, cy_mean, xy_pos_val in clusters_info:
        csize = len(cidx)

        distance = np.linalg.norm([cx_mean, cy_mean])
        if 150 < distance < 200:
            effective_size = csize * 1.5
        elif 200 < distance < 260:
            effective_size = csize * 1.8
        elif 30 < distance < 80:
            effective_size = csize * 0.8
        else:
            effective_size = csize
        if min_cluster_size <= effective_size <= max_cluster_size:
            valid_clusters.append((cid, cidx, cx_mean, cy_mean, xy_pos_val))

    if len(valid_clusters) < 2:
        return new_centers, labels, bounding_boxes, fix, []


    valid_clusters_sorted = sorted(valid_clusters, key=lambda t: t[2]) 
    found_pair = False
    pair_indices = None
    num_valid = len(valid_clusters_sorted)

    for i in range(num_valid):
        for j in range(i+1, num_valid):
            _, _, cx1, cy1, _ = valid_clusters_sorted[i]
            _, _, cx2, cy2, _ = valid_clusters_sorted[j]
            dist_ = np.linalg.norm([cx1 - cx2, cy1 - cy2])
            if dist_ >= min_separation_distance:
                pair_indices = (i, j)
                found_pair = True
                break
        if found_pair:
            break

    if not found_pair:
        return new_centers, labels, bounding_boxes, fix, []


    i_sel, j_sel = pair_indices
    cluster1 = valid_clusters_sorted[i_sel] 
    cluster2 = valid_clusters_sorted[j_sel]

    csize1 = len(cluster1[1])
    csize2 = len(cluster2[1])
    print("Cluster 1 csize:", csize1)
    print("Cluster 2 csize:", csize2)

    labels[:] = -1
    xy_pos_out = []
    for lb_idx, cluster in enumerate([cluster1, cluster2]):
        cid_, local_indices_c, cx_, cy_, xy_pos_val = cluster
        new_centers[lb_idx] = (cx_, cy_)
        for loc_i in local_indices_c:
            global_i = local_point_indices[loc_i]
            labels[global_i] = lb_idx
        half_size = 6.5
        bottom_left = (cx_ - half_size, cy_)
        top_right = (cx_ + half_size, cy_ + 13)
        bounding_boxes[lb_idx] = (bottom_left, top_right)
        xy_pos_out.append(xy_pos_val)

    fix = True

    return new_centers, labels, bounding_boxes, fix, xy_pos_out
