import socket
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from PolyGen import PolyGen
from test.simple_sim.udp_comm import CmdSend,ObsRecv
from test.simple_sim.record import Plotjuggler
import math
from pyControl import PosControl,AttiControl,RateControl,Mixer
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
    # _gain_p << 0.07,0.07,0.2;
    # _gain_i << 0.25,0.2,0.1;
    # _gain_d << 0.001,0.001,0.0;
    controller.rate_ctl.set_pid_params(np.array([0.5,0.5,0.2]),np.array([0.08,0.08,0.1]),np.array([0.001,0.001,0.0]))
    last_rate_control_time =  controller.obs_recv.wait_data()["current_time"]
    poly_start_time = last_rate_control_time
    yaw_tile = 0
    last_vel_z = 0

    poly_gen = PolyGen()

    while True:
        # get obs
        obs = controller.obs_recv.wait_data()
        rot_quat = np.array([obs["quaternion"]["w"],obs["quaternion"]["x"],obs["quaternion"]["y"],obs["quaternion"]["z"]])
        pos_world = np.array([obs["pos"]["x"],obs["pos"]["y"],obs["pos"]["z"]])
        velocity_world =  np.array([obs["velocity"]["x"],obs["velocity"]["y"],obs["velocity"]["z"]])
        angular_velocity_world =  np.array([obs["angular_velocity"]["x"],obs["angular_velocity"]["y"],obs["angular_velocity"]["z"]])
        current_time = obs["current_time"]
    


        yaw= 0
        # control   
        if(current_time-poly_start_time<20):
            exp_pos = np.array([0,0,1])
        else:
            exp_pos = poly_gen.sample(current_time-poly_start_time-20)
            exp_pos_2s = poly_gen.sample(current_time-poly_start_time+0.2-20)

            yaw_vector = exp_pos_2s - exp_pos
            yaw = math.atan2(yaw_vector[1], yaw_vector[0])
        controller.pos_ctl.set_status(pos_world,velocity_world,angular_velocity_world,rot_quat,current_time-last_rate_control_time)
        # atti_thrust_sp = controller.pos_ctl.update(np.array([0,0,1]),np.array([0,0,0]),np.array([0,0,0]),float(0))
        atti_thrust_sp = controller.pos_ctl.update(exp_pos,np.array([0,0,0]),np.array([0,0,0]),float(yaw))

        print("atti_thrust_sp {} ".format(atti_thrust_sp))
        # print("pos_error{}".format(exp_pos-pos_world))
        # print("hover_thrust {} ".format(controller.pos_ctl.get_hover_thrust()))


        rate_sp = controller.atti_ctl.update(atti_thrust_sp[:4],rot_quat)
        # print("rate_sp{}".format(rate_sp))

        controller.rate_ctl.set_q_world(rot_quat)
        thrust_3 = controller.rate_ctl.update(rate_sp,angular_velocity_world,np.array([0,0,0]),current_time-last_rate_control_time)
        # print("thrust_3{}".format(thrust_3))

        thrust = controller.mix_ctl.update(np.array([thrust_3[0],thrust_3[1],thrust_3[2],atti_thrust_sp[4]]))

        # send_cmd
        print(thrust)

        # print(rot_quat)
        q = rot_quat[[1, 2,3,0]]
        rot = Rotation.from_quat(q)# x,y,z,w
        atti_euler = rot.as_euler("xyz", degrees=False) # 对应 PlotJuggler 

        # pj.publish(current_time,np.array([0,0,0]),atti_euler,controller.px4.get_rate_des()*57.3,angular_velocity_world*57.3,0)  
        # pj.publish_q(current_time,estim_z,velocity_world[2],current_time-last_rate_control_time,estim_z)        
      
        controller.cmd_send.publish(0,thrust=thrust)
        
        # controller.cmd_send.publish(0,thrust=[0,0,0,0])     
        last_rate_control_time = current_time
        last_vel_z = velocity_world[2]
