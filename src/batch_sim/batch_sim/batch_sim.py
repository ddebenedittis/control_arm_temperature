import os
import signal
import subprocess
import time


source_cmd = "source /opt/ros/jazzy/setup.bash && source ./install/setup.bash && "

for i in range(3):
    # Define your arguments
    launch_args = "ros2 launch robot_gazebo arm.launch.py" + \
        " task:=circle" + \
        " log:=False"

    # Start the launch
    process = subprocess.Popen([
        "bash", "-c",
        source_cmd + launch_args,
    ], preexec_fn=os.setsid)

    time.sleep(5)

    subprocess.run([
        "bash", "-c",
        source_cmd +
        " gz service -s /world/default/control" +
        " --reqtype gz.msgs.WorldControl" +
        " --reptype gz.msgs.Boolean" +
        " --timeout 3000" +
        " --req 'pause: false'"
    ])

    # Wait for a while
    time.sleep(5)

    # Send SIGINT to cleanly shut down
    os.killpg(os.getpgid(process.pid), signal.SIGINT)

    time.sleep(5)
