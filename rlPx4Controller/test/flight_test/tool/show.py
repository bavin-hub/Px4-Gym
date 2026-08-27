import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from lemniscate import Lemniscate
import numpy as np

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Lemniscate Curve Over Time')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

l = Lemniscate(scale=2)
t_values = torch.linspace(0, 20*np.pi, 1000)  # Vary the time parameter
pos = l.update(t_values).numpy()


# 绘制 Lemniscate
ax.plot(pos[:,0],pos[:,1],pos[:,2])
plt.title('Lemniscate of Bernoulli')
plt.show()