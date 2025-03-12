import numpy as np
import cv2
import yaml

def generate_simple_map(output_image, output_yaml, map_size, resolution, obstacles):
    """
    Generate a simple occupancy grid map and save as an image and YAML file.
    :param output_image: Output image filename (e.g., 'map.png')
    :param output_yaml: Output YAML filename (e.g., 'map.yaml')
    :param map_size: Map size (width, height) in grid units
    :param resolution: Size of each grid cell in meters per grid unit
    :param obstacles: List of obstacles, each defined as (x, y, w, h) in grid units
    """
    # Create a blank map (all white)
    grid_map = np.ones(map_size, dtype=np.uint8) * 255

    # Add obstacles (fill with black)
    for obs in obstacles:
        x, y, w, h = obs
        grid_map[int(y):int(y+h), int(x):int(x+w)] = 0

    # Save the image
    cv2.imwrite(output_image, grid_map)

    # Set map origin (assuming the origin is at the center of the map)
    origin_x = map_size[1] * resolution / 2  # Height direction
    origin_y = map_size[0] * resolution / 2  # Width direction

    # Create YAML file content
    yaml_data = {
        "image": output_image,
        "resolution": resolution,
        "origin": [origin_x, origin_y, 0.0],  # Origin at map center
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.2
    }

    # Save the YAML file
    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    print(f"Map image saved to {output_image}")
    print(f"YAML file saved to {output_yaml}")

# Example usage
if __name__ == "__main__":
    map_size = (500, 500)  # Map size in grid units (width, height, in centimeters)
    resolution = 0.01  # Each grid unit represents 1 cm

    # Obstacles defined based on the image (converted from cm to grid units)
    # All distances are in centimeters, and block sizes are square
    obstacles = [
        (400, 130, 12.5, 12.5),  # Bottom-left obstacle (x, y, w, h)
        (450, 130, 6, 6),  
        (450, 399, 6, 6), 
        (450, 264.5, 6, 6), # Bottom-right obstacle (x, y, w, h)
        (400, 392.5, 12.5, 12.5),  # Top-left obstacle (x, y, w, h)
        #(400, 130, 50, 6),    # Top-right obstacle (x, y, w, h)
        #(400, 399, 50, 6),     # Far-right middle obstacle (x, y, w, h)
        (330, 110,20,10),
        (330, 420,20,10),
        (330, 110,10,310),
    ]

    generate_simple_map("map.png", "map.yaml", map_size, resolution, obstacles)
