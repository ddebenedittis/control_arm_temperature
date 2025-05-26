from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import rosbag2_py
from rosbag2_py import Recorder, RecordOptions


class BagRecorder(Node):
    def __init__(self):
        super().__init__('BagRecorder')
        
        # ============================ Parameters ============================ #
        
        self.declare_parameter('task', 'point')
        task = str(self.get_parameter('task').get_parameter_value().string_value)
        
        self.declare_parameter('time', 'yyyy-mm-dd-hh-mm-ss')
        time = str(self.get_parameter('time').get_parameter_value().string_value)
        
        workspace_directory = f"{get_package_share_directory('whole_body_controller')}/../../../../"
        bag_filepath = f"{workspace_directory}/bags/{time}-{task}"
        
                
        # Create the bag reader.
        self.recorder = Recorder()
        
        self.record_options = RecordOptions()
        self.record_options.all_topics = True
        
        self.storage_options: rosbag2_py.StorageOptions = rosbag2_py._storage.StorageOptions(
            uri=bag_filepath, storage_id='mcap')
        
        self.get_name()
        
    def __call__(self):
        try:
            self.recorder.record(self.storage_options, self.record_options)
        except KeyboardInterrupt:
            pass


def main(args=None):
    rclpy.init(args=args)
    
    recorder = BagRecorder()
    recorder()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
