"""
MyModule

This module provides functionality related to ...
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
# 3d
class Lemniscate:
    def __init__(self,scale=1.0,c=1.0):
        self.feq = 10 #s
        self.t_offset = 3.141593/2
        self.scale = scale
        self.c = c
    def update(self,t): #[n,1] | [n,]
        if(t.dim() == 1):
            t = t.unsqueeze(1)
        # now t.dim =2
            
        sin_t = torch.sin(t/self.feq + self.t_offset)
        cos_t = torch.cos(t/self.feq + self.t_offset)
        sin2p1 = torch.square(sin_t) + 1/self.scale

        x = cos_t / sin2p1
        y = (sin_t * cos_t) / sin2p1
        z = (self.c * sin_t) / sin2p1
        return torch.cat((x, y, z), 1)
    def update_de(self,t):
        if(t.dim() == 1):
            t = t.unsqueeze(1)
        sin_t = torch.sin(t/self.feq + self.t_offset)
        cos_t = torch.cos(t/self.feq + self.t_offset)
        sin2p1 = torch.square(sin_t) + 1/self.scale

        x =  - sin_t/(self.feq * (sin2p1)) - (2 * sin_t * torch.square(cos_t))/(self.feq * torch.square(sin2p1))
        y =  - torch.square(sin_t)/(self.feq * (sin2p1)) + torch.square(cos_t)/(self.feq * (sin2p1)) - (2 *torch.square(sin_t) * torch.square(cos_t) )/(self.feq * torch.square(sin2p1))
        z =  (self.c * cos_t)/(self.feq * (sin2p1)) - (2 * self.c * torch.square(sin_t)*cos_t)/(self.feq * torch.square(sin2p1))
        return torch.cat((x, y, z), 1)

if __name__ == "__main__":
    l = Lemniscate(scale=2)
    t = torch.zeros([2])
    while True:
        t = t + 0.01
        # print(l.update(t))
        print(l.update_de(t))
        time.sleep(0.01)