# visualize_quaternion.py

from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_quaternion(quat):
    """3D 可视化四元数旋转"""
    
    q = Quaternion(quat)
    
    # 创建 3D 图
    fig = plt.figure(figsize=(12, 5))
    
    # 左图：旋转前
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("旋转前", fontsize=14)
    
    # 右图：旋转后
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"旋转后\n角度: {q.degrees:.1f}°", fontsize=14)
    
    # 原始坐标轴
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    # 旋转后的坐标轴
    x_rotated = q.rotate(x_axis)
    y_rotated = q.rotate(y_axis)
    z_rotated = q.rotate(z_axis)
    
    # 绘制原始坐标轴
    for ax in [ax1, ax2]: 
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.15, label='X' if ax == ax1 else None)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=2, arrow_length_ratio=0.15, label='Y' if ax == ax1 else None)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=2, arrow_length_ratio=0.15, label='Z' if ax == ax1 else None)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # 绘制旋转后的坐标轴
    ax2.quiver(0, 0, 0, x_rotated[0], x_rotated[1], x_rotated[2], 
              color='r', linewidth=3, linestyle='--', arrow_length_ratio=0.15, alpha=0.7)
    ax2.quiver(0, 0, 0, y_rotated[0], y_rotated[1], y_rotated[2], 
              color='g', linewidth=3, linestyle='--', arrow_length_ratio=0.15, alpha=0.7)
    ax2.quiver(0, 0, 0, z_rotated[0], z_rotated[1], z_rotated[2], 
              color='b', linewidth=3, linestyle='--', arrow_length_ratio=0.15, alpha=0.7)
    
    # 绘制旋转轴
    if q.degrees > 0.1:
        axis = q.axis
        ax2.quiver(0, 0, 0, axis[0], axis[1], axis[2], 
                  color='black', linewidth=4, arrow_length_ratio=0.2, label='旋转轴')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印分析
    print(f"\n四元数:  [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
    print(f"旋转角度: {q.degrees:.2f}°")
    print(f"旋转轴:  [{q.axis[0]:.4f}, {q.axis[1]:.4f}, {q.axis[2]:.4f}]")
    print(f"\nX轴 [1,0,0] → [{x_rotated[0]:.3f}, {x_rotated[1]:.3f}, {x_rotated[2]:.3f}]")
    print(f"Y轴 [0,1,0] → [{y_rotated[0]:.3f}, {y_rotated[1]:.3f}, {y_rotated[2]:.3f}]")
    print(f"Z轴 [0,0,1] → [{z_rotated[0]:.3f}, {z_rotated[1]:.3f}, {z_rotated[2]:.3f}]")

# 测试
if __name__ == '__main__':
    print("测试几个常见四元数\n")
    
    test_cases = [
        ("绕Y轴-60°", Quaternion(axis=[0, 1, 0], degrees=-60)),
        ("绕Z轴90°", Quaternion(axis=[0, 0, 1], degrees=90)),
        ("绕X轴45°", Quaternion(axis=[1, 0, 0], degrees=45)),
    ]
    
    for name, q in test_cases:
        print(f"\n{'='*70}")
        print(name)
        print('='*70)
        visualize_quaternion(q.elements)
        input("按回车继续...")