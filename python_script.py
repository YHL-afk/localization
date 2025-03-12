import subprocess
import sys
import json
import matplotlib.pyplot as plt
from math import sin, cos  # 添加这一行



def initialize_plot():
    """Initialize the plot for real-time visualization."""
    plt.ion()
    plt.figure()
    plt.title("Laser Scan Visualization")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.figure(figsize=(10, 10))
    ##plt.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.5, label='Original Points')  # 显示所有点


def update_plot(ranges, angles):
    """Update the plot with the new laser scan data."""
    x = [r * cos for r, cos in zip(ranges, [cos(a) for a in angles])]
    y = [r * sin for r, sin in zip(ranges, [sin(a) for a in angles])]
    plt.clf()
    plt.title("Laser Scan Visualization")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x, y, s=1)
    plt.draw()
    plt.pause(0.001)


def compile_cpp_program():
    """Run the `make` command to compile the C++ program."""
    print("Compiling the C++ program...")
    result = subprocess.run(["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        sys.exit(1)
    print("Compilation successful!")


def run_cpp_program():
    """Run the compiled C++ program and process its output."""
    print("Running the compiled program...")
    process = subprocess.Popen(
        ["./lds_driver"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        initialize_plot()
        while True:
            line = process.stdout.readline()
            if not line:  # End of output
                break
            try:

                # 解析 JSON 数据
                data = json.loads(line.strip())
                ranges = data['ranges']
                angles = data['angles']

                # 打印解析后的数据
                print(f"Ranges: {ranges}")
                print(f"Angles: {angles}")

                # 更新绘图
                update_plot(ranges, angles)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("Terminating...")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        process.terminate()
    finally:
        process.wait()


if __name__ == "__main__":
    compile_cpp_program()
    run_cpp_program()