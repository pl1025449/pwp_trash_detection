import time
import processing_parallel
from processing_parallel import process_frame
from motor_steering import set_motor_speeds
from Motordriver import stop_all,_send_command,turn_right

def avoid_obstacle():
    _send_command('forward')
    time.sleep(2.8)
    set_motor_speeds(25.0)
    time.sleep(1.0)
    _send_command('forward')
    time.sleep(3.3)
    set_motor_speeds(-30.0)
    time.sleep(2.8)
    _send_command('forward')
    time.sleep(2.8)
    set_motor_speeds(25.0)
    time.sleep(1.0)
