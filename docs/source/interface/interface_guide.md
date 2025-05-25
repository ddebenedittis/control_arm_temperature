# Jacopino Pane e Vino Tiffany Ciony's Interface Usage Guide

## Guide's Instructions

Read everything. Stupid.

## Preliminaries

Install chrony:
```shell
sudo apt install chrony
```

Understand the Ethernet and/or WiFi address of the Raspberry.
Do this by connecting via ssh to it and running `ip a`.
If no Ethernet or not WiFi is set up, you will not see the corresponding address.

Decide whether to use the Ethernet or the WiFi. You will use different IP addresses for the two connections.

Edit the config file and allow the pi3hat address. I.e., add `allow 100.100.100.3/24` (or whatever it is) at the end of the file. You can do it in the terminal with
```shell
sudo nano /etc/chrony/chrony.conf
```
Then, close the file with CTRL+X, and save (Y+ENTER).

## Usage

Turn on the battery (quick press followed by a long press). Turn on the Pi3Hat (press the damn button).

The interface can be used with the WiFi or with a LAN connection. The only difference is the IP address used when connecting with SSH. I used only the SSH cause Jacopino Pane e Vino Tiffany Ciony (hereafter only Tiffany) was angry and did not want to set up the WiFi. Deal with it.

Connect to the LAN.
Go to wired settings, open the settings, go to the `IPv4` tab, choose the `manual` method.
Then, use `100.100.100.4` as the `Address` and `255.255.255.0` as the `Netmask`. Leave the gateway empty.
The Address last entry needs to be different from the Pi3Hat address.
Click `apply`, and then check that you are connected to the LAN.

Verify that in the LAN settings (top right) the IPv4 address is `100.100.100.4`.

In the following, two terminals will be needed. One for connecting to the Pi3Hat and launching the interface (which is on the Pi3Hat), and the other to load the controllers and launch the shit in your local machine.

### In a First Terminal

Connect with ssh to the Pi3Hat with
```shell
ssh jacopocioni@100.100.100.3
```
If the Raspberry has a different user, use that one instead of `jacopocioni`.

Enable superuser in the Pi3Hat terminal
```shell
sudo su
```
then insert the password (e.g. `123456`).

Verify that the Pi3Hat is synchronized (`Leap status` is `normal`) with
```shell
chronyc tracking
```
if not, follow [Troubleshoot: Leap Status Not Synchronized](#troubleshoot-leap-status-not-synchronized).

Navigate to the Pi3Hat workspace
```shell
cd mulinex_ws
```

Source ROS and the Pi3Hat workspace
```shell
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Export the ROS_DOMAIN_ID env variable
```shell
export ROS_DOMAIN_ID=10
```
this variable needs to be the same in both terminals.

Check in the urdf file that the joints have the correct names, order, etc with
```shell
nano src/pi3hat_hw_interface/urdf/test_int.urdf.xacro
```

Check the interface kp and kd gains with
```shell

```

Run the hardware interface with
```shell
ros2 launch pi3hat_hw_interface test_0.launch.py
```
If everything goes correctly, in the terminal the package losses of the motors will be printed repeatedly. If this does not happen, there was a communication error with the motors. CTRL-C and run the command again.

The joints zero position is set when the interface activates. If the kp and kd are different from zero, the robot will resist movements.

### In a Second Terminal

Navigate to your workspace.

Source ROS and your workspace
```shell
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Export the same ROS_DOMAIN_ID env variable as the one in the pi3hat
```shell
export ROS_DOMAIN_ID=10
```

Check that there are no ROS topics with
```shell
ros2 topic list
```

Load the ROS controller to move the robot
```shell
ros2 control load_controller joint_controller --set-state active
```

Load the ROS controller to publish the robot state
```shell
ros2 control load_controller state_broadcaster --set-state active
```

Check that there are new topics with
```shell
ros2 topic list
```
like `/joint_controller/command`,`joint_controller/transition_event`, etc.

Do your shit...

## Shut Down

In the Pi3Hat terminal run
```shell
shutdown -h now
```

Wait a bit, and then turn off the Pi3Hat using the button. Otherwise, Tiffany will ********** you.

## Troubleshoot: Leap Status Not Synchronized

Run
```shell
nano /etc/chrony/chrony.conf
```
and set the correct IP of the OPC.

Run
```shell
systemctl restart chrony
systemctl status chrony     # check the activation of chrony daemon
chronyc tracking            # chech that Leap status is normal
```