from rlPx4Controller.traj_tools.polyTrajGen import PolyTrajGen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

key_points = np.array([
        [3,4,2],
        [5,-4,2],
        [8,2,3],
        [10,-2,3],
        [12,8,2],
        [20,-8,2],
        # [3,4,2]
    ])
dt_vec = np.array([3,10,5,6,6])
zero_vec = np.array([0,0,0])

traj = PolyTrajGen(key_points,dt_vec,zero_vec,zero_vec,zero_vec,zero_vec)



# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Lemniscate Curve Over Time')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_zlim(-1.5, 1.5)
pos_list = []
vel_list = []
acc_list = []
for i in np.arange(0.0,float(np.sum(dt_vec)),0.01):
    pos_list.append(traj.sample(i)) 
    vel_list.append(traj.sample_vel(i))
    acc_list.append(traj.sample_acc(i))

pos_array = np.asarray(pos_list)
vel_array = np.asarray(vel_list)
acc_array = np.asarray(acc_list)

ax.plot(pos_array[:,0],pos_array[:,1],pos_array[:,2], label='pos')
ax.plot(vel_array[:,0],vel_array[:,1],vel_array[:,2], label='vel')
ax.plot(acc_array[:,0],acc_array[:,1],acc_array[:,2], label='acc')

plt.title('Ploy Traj')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.show()



