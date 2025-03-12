import subprocess
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from matplotlib.patches import Polygon
from collections import deque
import math
from matplotlib.patches import Rectangle

def initialize_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Laser Scan Visualization")
    ax.set_xlim(-10, 10)
    ax.set_ylim(10, 10)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    return fig, ax

def compile_cpp_program():
    print("Compiling the C++ program...")
    result = subprocess.run(["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        sys.exit(1)
    print("Compilation successful!")

def update_plot(ranges, angles, ax):
    points = []
    for r, theta in zip(ranges, angles):
        if math.isfinite(r):
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            points.append((x, y))
    points = np.array(points)
    ax.clear()
    ax.set_title("Laser Scan Visualization")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=10, label="Lidar Points")
    ax.legend(loc="upper right")

def main():
    compile_cpp_program()
    chmod_result = subprocess.run(["chmod", "+x", "./lds_driver"],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
    if chmod_result.returncode != 0:
        print("Failed to add executable permission to lds_driver:")
        print(chmod_result.stderr)
        sys.exit(1)
    fig, ax = initialize_plot()
    process = subprocess.Popen(
        ["./lds_driver"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.strip())
                ranges = data['ranges']
                angles = data['angles']
                update_plot(ranges, angles, ax)
                plt.draw()
                plt.pause(0.001)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing data: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nVisualization terminated by user.")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        process.terminate()
    finally:
        process.wait()

if __name__ == "__main__":
    main()
