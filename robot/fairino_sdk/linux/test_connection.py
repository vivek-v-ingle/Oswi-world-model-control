import sys

sys.path.append("/home/vivek/Desktop/Fairino/fairino-python-sdk-main/linux")

from fairino import Robot

print("Connecting...")

robot = Robot.RPC("192.168.57.2")

print("SDK connect flag:", robot.is_connect)

if not getattr(robot, "is_connect", False):
    print("Forcing XML-RPC mode")
    Robot.RPC.is_connect = True

try:
    print("Controller IP:", robot.robot.GetControllerIP())
except Exception as e:
    print("Controller IP error:", e)

try:
    print("RobotEnable:", robot.RobotEnable(1))
except Exception as e:
    print("RobotEnable error:", e)

try:
    print("Mode Auto:", robot.Mode(0))
except Exception as e:
    print("Mode error:", e)
