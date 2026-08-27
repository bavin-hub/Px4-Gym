from inputs import get_gamepad
import time
import threading

# 遥控器读取
class joystick():
    def __init__(self,range=[21,232]):
        self.joy_value = [0, 0, 0, 0]
        self.range = range
        my_thread = threading.Thread(target=self.update)
        my_thread.start()
    def get_state(self):
        return self.joy_value
    def update(self):
        while True:
            events = get_gamepad()
            for event in events:
                # print(event)
                if(event.device.device_type == "joystick" and event.ev_type != "Sync" and event.ev_type != "Misc"):
                    # print("{} {} {}".format(event.state,event.ev_type,event.code))
                    # range[21,232]
                    value_norm = min(1, max(-1, (event.state-(self.range[1] - self.range[0])/2)/(self.range[1] -self.range[0])*2))
                    if(event.code == "ABS_Z"):
                        self.joy_value[3]= value_norm
                    elif(event.code == "ABS_X"):
                        self.joy_value[0]= value_norm
                    elif(event.code == "ABS_Y"):
                        self.joy_value[1]= value_norm
                    elif(event.code == "ABS_RX"):
                        self.joy_value[2]= value_norm
                    # print(self.joy_value)

if __name__ == "__main__":
    js = joystick(range=[0,2048])
    while True:
        # js.update()
        print(js.get_state())
        time.sleep(0.02)