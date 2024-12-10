import os
import yaml

from ament_index_python.packages import get_package_share_directory
import pinocchio as pin


class RobotWrapper(pin.RobotWrapper):
    def __init__(self, robot_name: str):
        
        # ======================== Read The YAML File ======================== #
        
        package_share_directory = get_package_share_directory('robot_model')
        
        yaml_path = package_share_directory + "/robots/all_robots.yaml"
        
        with open(yaml_path, 'r') as f:
            doc = yaml.load(f, yaml.SafeLoader)
            
        pkg_name = doc[robot_name]["pkg_name"]
        urdf_path = doc[robot_name]["urdf_path"]
        
        full_urdf_path = get_package_share_directory(pkg_name) + urdf_path
        
        # ============ Create The Robot Model And The Data Objects =========== #
        
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            full_urdf_path,
            get_package_share_directory(pkg_name),
        )
        
        super().__init__(model)
            
        # ==================================================================== #
        
        self.base_name = doc[robot_name]["base_name"]
        self.ee_name = doc[robot_name]["ee_name"]
        
        self.joint_names = doc[robot_name]["ordered_joint_names"]
