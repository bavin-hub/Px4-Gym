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
from rlPx4Controller.traj_tools import Lemniscate

from rlPx4Controller.pyParallelControl import ParallelPosControl

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
    parallel_pos_control = ParallelPosControl(2)
    # init time
    last_rate_control_time =  controller.obs_recv.wait_data()["current_time"]
    poly_start_time = last_rate_control_time
    # init yaw
    last_vel_z = 0
    lemniscate = Lemniscate(scale=3)
    hover_time = 10
    yaw= 0
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
            exp_pos = np.array([0,0,2])
        else:
            t = current_time - poly_start_time - hover_time
            exp_pos = lemniscate.update(torch.tensor([t])).numpy()
            exp_pos_2 = lemniscate.update(torch.tensor([t+0.1])).numpy()
            yaw_vector = exp_pos_2[0,:2] - exp_pos[0,:2]
            yaw = math.atan2(yaw_vector[1], yaw_vector[0])
            exp_pos = exp_pos[0] + np.array([0,0,2])
            yaw = np.pi

        # pos ctrl
        # controller.pos_ctl.set_status(pos_world,velocity_world,angular_velocity_world,rot_quat,current_time-last_rate_control_time)
        # atti_thrust_sp = controller.pos_ctl.update(exp_pos,np.array([0,0,0]),np.array([0,0,0]),float(yaw))

        # atti ctrl
        # rate_sp = controller.atti_ctl.update(atti_thrust_sp[:4],rot_quat)

        # rate ctrl
        # controller.rate_ctl.set_q_world(rot_quat)
        # thrust_3 = controller.rate_ctl.update(rate_sp,angular_velocity_world,np.array([0,0,0]),current_time-last_rate_control_time)
        
        # mixer
        # thrust = controller.mix_ctl.update(np.array([thrust_3[0],thrust_3[1],thrust_3[2],atti_thrust_sp[4]]))

        # replace rate->mixer
        root_rot_cpu = np.zeros((2,4))
        root_rot_cpu[0,:] = rot_quat

        # root_ang_vel_cpu = np.zeros((2,4))
        root_ang_vel_cpu = np.zeros((2,3))
        root_ang_vel_cpu[0,:] = angular_velocity_world
        
        # root_ang_vel_cpu = np.zeros((2,4))
        root_pos_cpu = np.zeros((2,3))
        root_pos_cpu[0,:] = pos_world

        root_vel_cpu = np.zeros((2,3))
        root_vel_cpu[0,:] = velocity_world

        actions_cpu = np.zeros((2,4))
        actions_cpu[0,:] = np.array([0,0,2,0])
        # actions_cpu[0,:3] = exp_pos
        actions_cpu[0,3] = yaw

        
        print(actions_cpu)

        parallel_pos_control.set_status(root_pos_cpu,root_rot_cpu,root_vel_cpu,root_ang_vel_cpu,0.01)
        thrust = parallel_pos_control.update(actions_cpu)
        thrust = thrust[0]
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
      