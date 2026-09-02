import sys

sys.path.append("/home/vivek/Desktop/Fairino/fairino-python-sdk-main/linux")

from fairino import Robot

robot = Robot.RPC("192.168.57.2")

if not getattr(robot, "is_connect", False):
    Robot.RPC.is_connect = True

print(robot.GetSDKVersion())
