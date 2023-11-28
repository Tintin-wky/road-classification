import pandas as pd
import os

# 列出CSV文件所在的文件夹
csv_folder = 'data'  # 替换为包含CSV文件的文件夹路径

# 创建一个空的DataFrame来存储统计特征
statistics_df = pd.DataFrame(columns=['File', 'Mean', 'Std', 'Min', 'Max'])

# 循环处理每个CSV文件
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        csv_file = os.path.join(csv_folder, filename)

        # 从CSV文件加载IMU数据
        df = pd.read_csv(csv_file)

        angular_velocity_z = df['angular_velocity_z']

        # 计算统计特征
        mean = angular_velocity_z.mean()
        std = angular_velocity_z.std()
        min_value = angular_velocity_z.min()
        max_value = angular_velocity_z.max()

        # 将统计特征添加到统计DataFrame
        statistics_df = pd.concat([statistics_df, pd.DataFrame({'File': [filename], 'Mean': [mean], 'Std': [std], 'Min': [min_value], 'Max': [max_value]})], ignore_index=True)

# 保存统计特征DataFrame为CSV文件
statistics_df = statistics_df.sort_values(by='File')
output_csv_file = 'imu_statistics.csv'  # 替换为输出的CSV文件路径
statistics_df.to_csv(output_csv_file, index=False)

print(f'Statistics features saved to {output_csv_file}')
