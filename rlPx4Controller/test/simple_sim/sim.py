from record import Plotjuggler

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView
import numpy as np
import omni.replicator.core as rep
import cv2
import random
import time

from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

from robot.drone import X152b
from robot.drone_view import X152bView

from udp_comm import ObsSend,CmdRecv
my_world = World(stage_units_in_meters=1.0)

def setup_scene():
    local_assets_path = "/home/hao/Documents/isaac_sim_sense/Flyscene"
    _usd_path = local_assets_path + "/little_scene/LGround.usd"
    add_reference_to_stage(_usd_path, prim_path="/World/Lground")



def sim_step(thrust):
    pass


def reset(copter):
    copter.post_reset()
    # copter.set_world_poses(np.array([[0,0,0.2]]), np.array([[1,0,0,0]]))
    # copter.set_velocities(np.array([[0,0,0.0]]))
    copter.physics_rotors[0].apply_forces(np.array([0,0,0]))
    copter.physics_rotors[1].apply_forces(np.array([0,0,0]))
    copter.physics_rotors[2].apply_forces(np.array([0,0,0]))
    copter.physics_rotors[3].apply_forces(np.array([0,0,0]))
    copter.set_velocities(np.array([[0,0,0,0,0,0]]))
    
    # print(copter.get_velocities())

def get_observations(copter,current_time) -> dict:
    root_pos, root_rot = copter.get_world_poses(clone=False)
    # pos(x,y,z) units=m ,rot(w,x,y,z) frame(world)
    root_velocity = copter.get_velocities(clone=False) 
    # relocity units rad/s  frame(world)
    root_angular_velocity = copter.get_angular_velocities(clone=False)
    # relocity units rad/s  frame(world)

    observations = {
        "current_time": current_time,
        "pos": {
            "x": float(root_pos[0][0]),
            "y": float(root_pos[0][1]),
            "z": float(root_pos[0][2]),
        },
        "quaternion": 
        {
            "w": float(root_rot[0][0]),
            "x": float(root_rot[0][1]),
            "y": float(root_rot[0][2]),
            "z": float(root_rot[0][3]),
        },
        "velocity": {
            "x": float(root_velocity[0][0]),
            "y": float(root_velocity[0][1]),
            "z": float(root_velocity[0][2]),
        },
        "angular_velocity":{
            "x": float(root_angular_velocity[0][0]),
            "y": float(root_angular_velocity[0][1]),
            "z": float(root_angular_velocity[0][2]),
        } ,
    }
    return observations


# if __name__ == "__name__":

copter_m = X152b(prim_path="/World/X152b", name="X152b",translation=np.array([0,0,0.2]))
copter = X152bView(prim_paths_expr="/World/X152b", name="X152b_view")
obs_send = ObsSend()
my_world.scene.add(copter)
for i in range(4):
    my_world.scene.add(copter.physics_rotors[i])

# 对应了[Warning] [omni.isaac.core.prims.rigid_prim_view] Physics Simulation View is not created yet

# my_world.scene.add_default_ground_plane()
setup_scene()

my_world.reset()
my_world.set_simulation_dt(physics_dt=0.01,rendering_dt=0.01)

pj = Plotjuggler()

cmd_recv = CmdRecv()

last_render_time = 0
last_rate_control_time= 0 
while simulation_app.is_running():
    # print(my_world.current_time_step_index)
    if my_world.is_playing() is False:
        my_world.step(render=True)
        reset(copter)
        continue

    if( my_world.current_time - last_render_time > 1/30):
        my_world.step(render=True)
        last_render_time = my_world.current_time
    else:
        my_world.step(render=False)

    observations = get_observations(copter,my_world.current_time)
    obs_send.publish(observations)

    last_rate_control_time = my_world.current_time

    cmd = cmd_recv.wait_data()

    if(cmd is None):
        my_world.step(render=True)
        continue
    cmd_sq = cmd*cmd
    rot_matrix = rot_utils.quats_to_rot_matrices(np.array([observations["quaternion"]["w"],observations["quaternion"]["x"],observations["quaternion"]["y"],observations["quaternion"]["z"]]))
    copter.physics_rotors[0].apply_forces(rot_matrix.dot(np.array([0,0,cmd_sq[0]*5])))
    copter.physics_rotors[1].apply_forces(rot_matrix.dot(np.array([0,0,cmd_sq[1]*5])))
    copter.physics_rotors[2].apply_forces(rot_matrix.dot(np.array([0,0,cmd_sq[2]*5])))
    copter.physics_rotors[3].apply_forces(rot_matrix.dot(np.array([0,0,cmd_sq[3]*5])))
    efforts = np.array([[-cmd_sq[0],-cmd_sq[1],cmd_sq[2],cmd_sq[3]]])*0.2
    copter.set_joint_efforts(efforts)
    # copter.get_jacobians()


    
simulation_app.close()



