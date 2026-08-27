import json
import socket
import numpy as np
import threading
class CmdSend():
    def __init__(self,port=20010) -> None:
        self.address = '127.0.0.1'
        self.port = port
        self.sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
    def publish(self,time_stamp,thrust):
        data = {
            "thrust":{
                "m_1": float(thrust[0]),
                "m_2": float(thrust[1]),
                "m_3": float(thrust[2]),
                "m_4": float(thrust[3]),
            }  
        }
        # print(data)
        self.sock.sendto( json.dumps(data).encode(), (self.address, self.port) )
class CmdRecv():
    def __init__(self,port=20010) -> None:
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.settimeout(0.02)

        # 绑定服务器地址和端口
        server_address = ('localhost', port)
        self.udp_socket.bind(server_address)
        self.latest_data = None

        # my_thread = threading.Thread(target=self.wait_data)
        # my_thread.start()
    def wait_data(self):
        try:
            data, client_address = self.udp_socket.recvfrom(1024)
            # 解析JSON数据
            try:
                json_data = json.loads(data.decode('utf-8'))
                thrust = json_data["thrust"]
                self.latest_data =  np.array([thrust["m_1"],thrust["m_2"],thrust["m_3"],thrust["m_4"]])
                return self.latest_data
            except json.JSONDecodeError as e:
                print("Error decoding JSON data: {}".format(e))
        except socket.timeout:
            # print("timeout")
            return None        


class ObsRecv():
    def __init__(self,port=30100) -> None:
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定服务器地址和端口
        server_address = ('localhost', port)
        self.udp_socket.bind(server_address)
    def wait_data(self):
        # 接收数据和客户端地址
        data, client_address = self.udp_socket.recvfrom(1024)
        # 解析JSON数据
        try:
            json_data = json.loads(data.decode('utf-8'))
            # print("Received JSON data from {}: {}".format(client_address, json_data))
            return json_data
        except json.JSONDecodeError as e:
            print("Error decoding JSON data: {}".format(e))
class ObsSend():
    def __init__(self,port=30100) -> None:
        self.address = '127.0.0.1'
        self.port = port
        self.sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
    def publish(self,obs_dict):
        # print(data)
        self.sock.sendto( json.dumps(obs_dict).encode(), (self.address, self.port) )