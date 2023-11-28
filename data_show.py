import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件加载IMU数据
csv_file = 'data/018_1.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 提取数据列（假设列名称是 'timestamp'、'angular_velocity_x' 和 'linear_acceleration_x'）
timestamps = df['timestamp']
angular_velocity_x = df['angular_velocity_x']
angular_velocity_y = df['angular_velocity_y']
angular_velocity_z = df['angular_velocity_z']
# 创建一个Matplotlib图形窗口
plt.figure(figsize=(10, 6))

# 绘制角速度和线性加速度数据
plt.plot(timestamps, angular_velocity_x, label='Angular Velocity (X)')
plt.plot(timestamps, angular_velocity_y, label='Angular Velocity (Y)')
plt.plot(timestamps, angular_velocity_z, label='Angular Velocity (Z)')

# 添加标题和标签
plt.title('IMU Data')
plt.xlabel('Timestamp')
plt.ylabel('Value')

# 添加图例
plt.legend()

# 保存图像到文件
output_image_file = 'imu_data_plot.png'  # 替换为你想要保存的图像文件路径
plt.savefig(output_image_file)

# 显示图形
plt.show()
