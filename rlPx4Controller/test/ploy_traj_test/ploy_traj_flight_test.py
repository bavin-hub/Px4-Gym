import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
import math
from rlPx4Controller.pyControl import PosControl,AttiControl,RateControl,Mixer
import torch
# udp
import sys 
sys.path.append("..") 
from simple_sim.udp_comm import CmdSend,ObsRecv
from simple_sim.record import Plotjuggler
from rlPx4Controller.traj_tools.polyTrajGen import PolyTrajGen
import csv
class Controller():
    def __init__(self) -> None:
        self.pos_ctl = PosControl()
        self.atti_ctl = AttiControl()
        self.rate_ctl = RateControl()
        self.mix_ctl = Mixer()

        self.cmd_send   = CmdSend()
        self.obs_recv = ObsRecv()



if __name__ == "__main__":
    pj = Plotjuggler()
    controller = Controller()
    controller.rate_ctl.set_pid_params(np.array([0.5,0.5,0.2]),np.array([0.08,0.08,0.1]),np.array([0.001,0.001,0.0]))
    
    # init time
    last_rate_control_time =  controller.obs_recv.wait_data()["current_time"]
    poly_start_time = last_rate_control_time
    # init yaw
    last_vel_z = 0
    hover_time = 10
    yaw= 0

    key_points = np.array([
            [0,0,2],
            [5,-4,2],
            [8,2,3],
            [10,-2,3],
            [12,8,2],
            [20,-8,2],
        ])
    dt_vec = np.array([10,20,20,20,10])
    zero_vec = np.array([0,0,0])

    traj = PolyTrajGen(key_points,dt_vec,zero_vec,zero_vec,zero_vec,zero_vec)


    # 指定保存的CSV文件名
    csv_file_path = 'example.csv'
    # 写入CSV文件
    csvfile =  open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow( ['e_x', 'e_y', 'e_z','r_x','r_y','r_z'])
    while True:
        # get obs
        obs = controller.obs_recv.wait_data()
        rot_quat = np.array([obs["quaternion"]["w"],obs["quaternion"]["x"],obs["quaternion"]["y"],obs["quaternion"]["z"]])
        pos_world = np.array([obs["pos"]["x"],obs["pos"]["y"],obs["pos"]["z"]])
        velocity_world =  np.array([obs["velocity"]["x"],obs["velocity"]["y"],obs["velocity"]["z"]])
        angular_velocity_world =  np.array([obs["angular_velocity"]["x"],obs["angular_velocity"]["y"],obs["angular_velocity"]["z"]])
        current_time = obs["current_time"]

        # control   
        if(current_time-poly_start_time<hover_time):
            # just hover
            exp_pos = np.array([0,0,2.0])
            exp_vel = np.array([0,0,0])
            yaw = np.pi
        elif(current_time-poly_start_time-hover_time < np.sum(dt_vec) ):
            t = current_time - poly_start_time - hover_time
            # exp_pos = lemniscate.update(torch.tensor([t])).numpy()
            # exp_pos_2 = lemniscate.update(torch.tensor([t+0.1])).numpy()
            # exp_vel = lemniscate.update_vel(torch.tensor([t])).numpy()
            exp_pos = traj.sample(t)
            exp_pos_2s = traj.sample(t+0.1)
            exp_vel = traj.sample_vel(t)

            yaw_vector = exp_pos_2s - exp_pos
            yaw = math.atan2(yaw_vector[1], yaw_vector[0])

        print(f"exp_pos{exp_pos} {pos_world}")
        csv_writer.writerow([exp_pos[0],exp_pos[1],exp_pos[2],
                             pos_world[0],pos_world[1],pos_world[2]
                             ])  
 


        # pos ctrl
        controller.pos_ctl.set_status(pos_world,velocity_world,angular_velocity_world,rot_quat,current_time-last_rate_control_time)
        atti_thrust_sp = controller.pos_ctl.update(exp_pos,exp_vel,np.array([0,0,0]),float(yaw))
        print(controller.pos_ctl.get_hover_thrust())
        # atti ctrl
        rate_sp = controller.atti_ctl.update(atti_thrust_sp[:4],rot_quat)

        # rate ctrl
        controller.rate_ctl.set_q_world(rot_quat)
        thrust_3 = controller.rate_ctl.update(rate_sp,angular_velocity_world,np.array([0,0,0]),current_time-last_rate_control_time)
        
        # mixer
        thrust = controller.mix_ctl.update(np.array([thrust_3[0],thrust_3[1],thrust_3[2],atti_thrust_sp[4]]))

        # send_cmd
        # print(thrust)
        controller.cmd_send.publish(0,thrust=thrust)
        
        last_rate_control_time = current_time
        last_vel_z = velocity_world[2]

        #####################################
        # for plot
        # print(rot_quat)
        q = rot_quat[[1, 2,3,0]]
        rot = Rotation.from_quat(q)# x,y,z,w
        atti_euler = rot.as_euler("xyz", degrees=False) # 对应 PlotJuggler 

        # pj.publish(current_time,np.array([0,0,0]),atti_euler,controller.px4.get_rate_des()*57.3,angular_velocity_world*57.3,0)  
        # pj.publish_q(current_time,estim_z,velocity_world[2],current_time-last_rate_control_time,estim_z)        
