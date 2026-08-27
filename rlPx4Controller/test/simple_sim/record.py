import socket
import json
import time
import numpy as np

# Plotjuggler 可视化

class Plotjuggler():
    def __init__(self) -> None:
        self.address = '127.0.0.1'
        self.port = 9870
        self.sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
    def publish(self,time_stamp, atti_target,atti,rate_target,rate,thrust):
        data = {
            "atti_target_msg": {
                "timestamp": time_stamp,
                "body_rate": {
                    "x": float(rate_target[0]),
                    "y": float(rate_target[1]),
                    "z": float(rate_target[2]),
                },
                "angle": {
                    "roll":  float(atti_target[0]),
                    "pitch": float(atti_target[1]),
                    "yaw":   float(atti_target[2]),
                },
            #     # "thrust": atti_target_msg.thrust
            },
            "atti_msg": {
                "timestamp": time_stamp,
                "body_rate": {
                    "x": float(rate[0]),
                    "y": float(rate[1]),
                    "z": float(rate[2]),
                },
                "angle": {
                    "roll":  float(atti[0]),
                    "pitch": float(atti[1]),
                    "yaw":   float(atti[2]),
                },
                # "thrust": atti_target_msg.thrust
            },
            # "delta_x": pos_target_msg.position.x - odom_msg.pose.pose.position.x,
            # "delta_y": pos_target_msg.position.y - odom_msg.pose.pose.position.y,
            # "delta_z": pos_target_msg.position.z - odom_msg.pose.pose.position.z,
            # "delta_ang_x": get_rpy(atti_target_msg.orientation)[0] - get_rpy(odom_msg.pose.pose.orientation)[0],
            # "delta_ang_y": get_rpy(atti_target_msg.orientation)[1] - get_rpy(odom_msg.pose.pose.orientation)[1],
            # "delta_ang_z": get_rpy(atti_target_msg.orientation)[2] - get_rpy(odom_msg.pose.pose.orientation)[2],   
        }
        # print(data)

        self.sock.sendto( json.dumps(data).encode(), (self.address, self.port) )
    def publish_q(self,time_stamp, w,x,y,z):
        data = {
            "quaternion": {
                "timestamp": time_stamp,
                    "w": float(w),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                
            #     # "thrust": atti_target_msg.thrust
            },
            # "delta_x": pos_target_msg.position.x - odom_msg.pose.pose.position.x,
            # "delta_y": pos_target_msg.position.y - odom_msg.pose.pose.position.y,
            # "delta_z": pos_target_msg.position.z - odom_msg.pose.pose.position.z,
            # "delta_ang_x": get_rpy(atti_target_msg.orientation)[0] - get_rpy(odom_msg.pose.pose.orientation)[0],
            # "delta_ang_y": get_rpy(atti_target_msg.orientation)[1] - get_rpy(odom_msg.pose.pose.orientation)[1],
            # "delta_ang_z": get_rpy(atti_target_msg.orientation)[2] - get_rpy(odom_msg.pose.pose.orientation)[2],   
        }
        if(data["quaternion"]["timestamp"] > 0.0026 or data["quaternion"]["timestamp"] < 0.0024):
            print(data)

        self.sock.sendto( json.dumps(data).encode(), (self.address, self.port) )
if __name__ == "__main__":
    pj = Plotjuggler()
    while True:
        pj.publish(time.time(),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0.2)
        time.sleep(0.02)