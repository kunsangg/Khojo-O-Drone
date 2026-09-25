<h1>PID Tuning GUI for Quadcopters and Differential Drive Robots</h1>
A proportional–integral–derivative controller (PID controller or three-term controller) is a control loop mechanism employing feedback that is widely used in industrial control systems and a variety of other applications requiring continuously modulated control. A PID controller continuously calculates an error value 
e(t) as the difference between a desired setpoint (SP) and a measured process variable (PV) and applies a correction based on proportional, integral, and derivative terms (denoted P, I, and D respectively), hence the name.

<h2>Prerequisites</h2>

- Ubuntu 22.04 LTS
- ROS2 Humble

<h2>Installation Instructions</h2>

```
mkdir ~/pid_ws
cd pid_ws
git clone https://github.com/erts-RnD/pid_tune.git -b ros2 src
```
```
cd ~/pid_ws
colcon build
source install/setup.bash
```

To run the Differential Drive PID Tuning GUI, use the following command;

```
ros2 run pid_tune pid_tune_differential.py 
```

For Quadcopters, there is two different GUIs for PID Tuning:

1. Button GUI
2. Slider GUI

To launch Quadcopter PID Tuning Button GUI, use the following command:
```
ros2 launch pid_tune pid_tune_drone.launch.py node_name:=button_ui
```

To launch Quadcopter PID Tuning Slider GUI, use the following command:
```
ros2 launch pid_tune pid_tune_drone.launch.py node_name:=slider_ui
```
