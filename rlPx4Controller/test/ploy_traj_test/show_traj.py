import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取CSV文件
csv_file_path = 'example.csv'
df = pd.read_csv(csv_file_path)

# 分离两条轨迹的数据
trajectory1 = df[['e_x', 'e_y', 'e_z']]
trajectory2 = df[['r_x', 'r_y', 'r_z']]

# 绘制3D轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory1['e_x'], trajectory1['e_y'], trajectory1['e_z'], label='Reference trajectory')
ax.plot(trajectory2['r_x'], trajectory2['r_y'], trajectory2['r_z'], label='Real trajectory')

# 添加标签和标题
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Lemniscate Trajectories')
ax.legend()

# 显示图形
plt.show()