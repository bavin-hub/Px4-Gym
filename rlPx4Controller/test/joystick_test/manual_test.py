import socket
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation

from Px4Controller import Px4Controller
# TODO: 将在之后移除 Px4Controller
from ..simple_sim.udp_comm import CmdSend,ObsRecv
import js_test as js_test
from ..simple_sim.record import Plotjuggler
class Controller():
    def __init__(self) -> None:
        self.px4 = Px4Controller()
        self.cmd_send   = CmdSend()
        self.obs_recv = ObsRecv()
if __name__ == "__main__":
    pj = Plotjuggler()
    controller = Controller()
    # _gain_p << 0.07,0.07,0.2;
    # _gain_i << 0.25,0.2,0.1;
    # _gain_d << 0.001,0.001,0.0;
    controller.px4.set_rate_pid(np.array([0.5,0.5,0.2]),np.array([0.08,0.08,0.1]),np.array([0.001,0.001,0.0]))
    js = js_test.joystick(range=[0,2048])
    last_rate_control_time =  controller.obs_recv.wait_data()["current_time"]
    yaw_tile = 0
    last_vel_z = 0

    while True:
        max_tile = 10/57.3
        roll_tile = js.get_state()[0]*max_tile
        pitch_tile = js.get_state()[1]*max_tile
        yaw_tile -= js.get_state()[2]*1/57.3

        # print(yaw_tile)
        rot = Rotation.from_euler("zyx", np.array([yaw_tile,pitch_tile , roll_tile]), degrees=False)
        quat_des = rot.as_quat()
        quat_des = np.array([quat_des[3],quat_des[0],quat_des[1],quat_des[2]])
        # print(quat_des)
        thrust_set = (js.get_state()[3]+1)/2
        # controller.px4.set_thrust(0)
        controller.px4.set_thrust((js.get_state()[3]+1)/2)

        # print(cr.wait_data())
        # get obs
        obs = controller.obs_recv.wait_data()
        rot_quat = np.array([obs["quaternion"]["w"],obs["quaternion"]["x"],obs["quaternion"]["y"],obs["quaternion"]["z"]])
        pos_world = np.array([obs["pos"]["x"],obs["pos"]["y"],obs["pos"]["z"]])
        velocity_world =  np.array([obs["velocity"]["x"],obs["velocity"]["y"],obs["velocity"]["z"]])
        angular_velocity_world =  np.array([obs["angular_velocity"]["x"],obs["angular_velocity"]["y"],obs["angular_velocity"]["z"]])
        current_time = obs["current_time"]

        
        thrust_z = np.matmul(rot.as_matrix(),np.array([0,0,thrust_set]))[2]
        # estim_z = (velocity_world[2] - last_vel_z)/ (current_time-last_rate_control_time)
        # estim_z = 0
        estim_z = controller.px4.hover_thrust_estimator(current_time-last_rate_control_time, velocity_world[2],thrust_z)
        print('estimator Hover t_z {:.2f} v_z {:.3f} h_e{:.2f}'.format(thrust_z,velocity_world[2],0))



        # control
        controller.px4.atti_update(quat_des,rot_quat)
        controller.px4.rate_update(angular_velocity_world,np.array([0,0,0]),current_time-last_rate_control_time,False)
        # send_cmd
        thrust = controller.px4.get_thrust_output()
        thrust = thrust*5
        # print(thrust)

        # print(rot_quat)
        q = rot_quat[[1, 2,3,0]]
        rot = Rotation.from_quat(q)# x,y,z,w
        atti_euler = rot.as_euler("xyz", degrees=False) # 对应 PlotJuggler 

        pj.publish(current_time,np.array([0,0,0]),atti_euler,controller.px4.get_rate_des()*57.3,angular_velocity_world*57.3,0)  
        pj.publish_q(current_time,estim_z,velocity_world[2],current_time-last_rate_control_time,estim_z)        
      
        controller.cmd_send.publish(0,thrust=thrust)
        
        # controller.cmd_send.publish(0,thrust=[0,0,0,0])     
        # controller.cmd_send.publish(0,thrust=[0.2,0,0,0.2])           
        print(current_time-last_rate_control_time)
        last_rate_control_time = current_time
        last_vel_z = velocity_world[2]
