import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
from matplotlib.patches import Polygon


def setup_realtime_plot(map_image_path, map_yaml_path):
    """
    Initialize the real-time plotting environment.
    :param map_image_path: Path to the map image file
    :param map_yaml_path: Path to the map configuration file
    :return: figure, ax, scatter_lidar, scatter_obstacles, scatter_pose, map_metadata
    """
    # Load map and metadata
    with open(map_yaml_path, 'r') as file:
        map_metadata = yaml.safe_load(file)

    map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
    if map_image is None:
        raise FileNotFoundError(f"Cannot load map image from {map_image_path}")

    resolution = map_metadata["resolution"] * 100  # Convert to cm
    origin = np.array(map_metadata["origin"][:2]) * 100  # Convert to cm, take x and y
    map_height, map_width = map_image.shape

    # Set plot range (in cm)
    fig, ax = plt.subplots()
    ax.imshow(map_image, cmap='gray', origin='lower', extent=[
        0, map_width * resolution,
        0, map_height * resolution
    ])
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Real-Time Lidar Data and Obstacles on Map")

    # Initialize plot elements
    scatter_lidar, = ax.plot([], [], 'bo', markersize=2, label='Lidar Points')
    scatter_pose, = ax.plot([], [], 'gx', markersize=10, label='Lidar Position')
    ax.legend()

    return fig, ax, scatter_lidar, scatter_pose, map_metadata

def update_realtime_plot(scatter_lidar, lidar_data, map_metadata):
    """
    Update the real-time plot with lidar and obstacle data.
    :param scatter_lidar: Lidar points scatter object
    :param scatter_obstacles: Detected obstacles scatter object
    :param lidar_data: Current frame lidar data
    :param detected_obstacles: Detected obstacles
    :param map_metadata: Map metadata
    """
    resolution = map_metadata["resolution"] * 100  # Convert to cm
    origin = np.array(map_metadata["origin"][:2]) * 100  # Convert to cm

    # Convert lidar data to Cartesian coordinates
    ranges, angles = lidar_data[:, 0], lidar_data[:, 1]  # In meters
    x_coords = ranges * np.cos(angles)  # Convert to cm
    y_coords = ranges * np.sin(angles) # Convert to cm
    lidar_points = np.column_stack((x_coords, y_coords))

    # Transform lidar points to map coordinates
    lidar_points_map = lidar_points + origin

    # Update lidar points on plot
    scatter_lidar.set_data(lidar_points_map[:, 0], lidar_points_map[:, 1])


    plt.pause(0.01)


# plot1.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def draw_radar_position(ax, map_metadata, radar_position=None):
    """
    在地图上绘制雷达的全局位置及其坐标文本，坐标单位与地图保持一致 (都以厘米为单位)。
    如果 radar_position 以 "米" 为单位，则需要在此处做米->厘米的转换。

    :param ax: Matplotlib Axes 对象
    :param map_metadata: 地图的元数据，含 resolution(米/像素) 和 origin(米)，但在外部已转成厘米
    :param radar_position: (X, Y) - 雷达在全局坐标系 (米) 下的位置（若已是厘米，可相应调整逻辑）
    """
    # 如果在 setup_realtime_plot 中，map_metadata["origin"] 已经 * 100，单位则是厘米
    origin = np.array(map_metadata.get("origin", [0, 0])[:2])  # 这里直接是厘米
    # resolution = map_metadata["resolution"] * 100  # 若需要，可再次获取（单位：厘米/像素）

    # 1) 不清除 ax.patches，因为这个函数只负责画雷达位置，可能你还要在别处画别的东西
    #   若想只保留最新的雷达位置，可在此先移除旧的 scatter/text

    # 只删除旧的雷达标记（scatter_radar）与文字（text_radar）
    if hasattr(ax, 'scatter_radar'):
        ax.scatter_radar.remove()
    if hasattr(ax, 'text_radar'):
        ax.text_radar.remove()

    if radar_position is not None:
        radar_x_m, radar_y_m = radar_position  # 这里假设传进来的雷达坐标是“米”
        # 转换到“地图绘制使用的厘米坐标”
        radar_x_map = radar_x_m  + origin[0]
        radar_y_map = radar_y_m  + origin[1]

        # 绘制雷达位置
        ax.scatter_radar = ax.scatter(
            radar_x_map,
            radar_y_map,
            c='green',
            s=100,
            marker='*',
            label='Radar Position'
        )

        # 显示雷达坐标 (以米为单位标注，或你也可改为以厘米输出)
        ax.text_radar = ax.text(
            radar_x_map, radar_y_map,
            f"Radar: ({radar_x_m:.2f}, {radar_y_m:.2f})",
            color='green', fontsize=10, ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0)
        )
        # 标记该文本为雷达位置，以便在下一帧更新时不被误删
        ax.text_radar.is_radar_position = True

    plt.draw()
    plt.pause(0.01)



def draw_ekf_objects(ax, bounding_boxes, map_metadata, radar_position=None):
    """
    在地图上绘制由 EKF 跟踪得到的目标框，显示它们的中心坐标和标签。
    以蓝色标记区分，与原始检测框(红色)相区别。

    :param ax: Matplotlib Axes 对象
    :param bounding_boxes: 列表，包含若干目标框，每个框由 (bottom_left, top_right) 组成
    :param map_metadata: 地图的元数据
    :param radar_position: Tuple (X, Y) - 雷达在全局坐标系中的位置（可选）
    :return: 返回每个目标中心的坐标列表，例如 [ (cx1, cy1), (cx2, cy2), ... ]
    """
    # 从地图元数据获取分辨率与原点（并转换为厘米）
    resolution = map_metadata.get("resolution", 0.05) * 100  # 假设 resolution 单位是米，转换为厘米
    origin = np.array(map_metadata.get("origin", [0, 0])[:2]) * 100

    # 1) 只清除先前由本函数绘制的 EKF patches 和 texts
    #   通过给 patch 和 text 设置一个自定义的属性 is_ekf 来实现区分
    for p in reversed(ax.patches):
        if getattr(p, 'is_ekf', False):
            p.remove()

    ekf_texts_to_remove = []
    for txt in ax.texts:
        if getattr(txt, 'is_ekf', False):
            ekf_texts_to_remove.append(txt)
    for txt in ekf_texts_to_remove:
        txt.remove()

    # 2) 绘制新的 EKF 目标框和标签
    centers = []
    for idx, bbox in enumerate(bounding_boxes):
        if bbox is None:
            continue  # 如果没有目标框，跳过

        bottom_left, top_right = bbox
        # 计算中心坐标（局部坐标）
        cx = 0.5 * (bottom_left[0] + top_right[0])
        cy = 0.5 * (bottom_left[1] + top_right[1])
        centers.append((cx, cy))

        # 转换到地图坐标系
        bottom_left_map = np.array(bottom_left) * resolution + origin
        top_right_map = np.array(top_right) * resolution + origin
        cx_map = cx * resolution + origin[0]
        cy_map = cy * resolution + origin[1]

        # 计算矩形宽度和高度
        width = top_right_map[0] - bottom_left_map[0]
        height = top_right_map[1] - bottom_left_map[1]

        # 画矩形（蓝色）
        rect = Rectangle(
            bottom_left_map,
            width,
            height,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        # 自定义属性，用于下一次清除
        rect.is_ekf = True
        ax.add_patch(rect)

        # 显示中心坐标
        text_center = ax.text(
            cx_map,
            cy_map,
            f"({cx:.2f}, {cy:.2f})",
            color='blue',
            fontsize=8,
            ha='center',
            va='center'
        )
        text_center.is_ekf = True  # 标记为 EKF 文本

        # 显示标签（如 EKF-1, EKF-2 ...），在目标框上方
        label = idx + 1  # 标签从1开始
        label_text = f"EKF-{label}"
        label_x = (bottom_left_map[0] + top_right_map[0]) / 2
        label_y = top_right_map[1] + (height * 0.1)
        text_label = ax.text(
            label_x,
            label_y,
            label_text,
            color='blue',
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0)
        )
        text_label.is_ekf = True  # 标记为 EKF 文本

    # 3) 如果需要，可在此处更新或绘制雷达位置（可选）
    #    一般情况下 draw_detected_objects() 已经会绘制雷达位置，如果需要单独在本函数绘制，
    #    可参考 draw_detected_objects 的写法

    # 强制刷新绘图
    plt.draw()
    plt.pause(0.01)

    return centers
