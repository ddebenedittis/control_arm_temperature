import os

import numpy as np


class KPI:
    def __init__(self, subdir: str):
        self.foldername = 'log'
        self.joint_names = [r'$q_1$', r'$q_2$', r'$q_3$']

        self.subdir = subdir

        self.npzfile = np.load(f"{self.foldername}/csv/{self.subdir}/log.npz")
        
    def compute_ee_error(self):
        ee_position = self.npzfile['ee_position']
        reference_position = self.npzfile['reference_position']
        
        # Compute the RMSE for each dimension (x, y, z)
        errors = ee_position - reference_position
        squared_errors = np.square(errors)
        mean_squared_error = np.mean(np.sum(squared_errors, axis=1))
        rmse = np.sqrt(mean_squared_error)
        
        # Compute the max error
        max_error = np.max(
            np.sqrt(np.sum(squared_errors, axis=1))
        )
        
        return rmse, max_error
    
    def compute_temp_max_and_mean(self):
        temp = self.npzfile['temperatures']
        temp_max = np.max(temp, axis=0)
        temp_mean = np.mean(temp, axis=0)
        return temp_max, temp_mean
    
    def print_all_kpi(self):
        rmse, max_error = self.compute_ee_error()
        print(f"End-effector position RMSE: {rmse}")
        print(f"End-effector position max error: {max_error}")
        
        temp_max, temp_mean = self.compute_temp_max_and_mean()
        print(f"Max temperatures: {temp_max}")
        print(f"Mean temperatures: {temp_mean}")
    
    
def main():
    # Get the number of folders to process and create the output directories
    rootdir_csv = "log/csv"
    n_to_process = 0
    for file in os.listdir(rootdir_csv):
        d = os.path.join(rootdir_csv, file)
        if os.path.isdir(d):
            n_to_process += 1

    # Process each folder
    counter = 0
    for file in os.listdir(rootdir_csv):
        d = os.path.join(rootdir_csv, file)
        if os.path.isdir(d):
            counter += 1
            print(f"\nProcessing the {counter}-th folder out of {n_to_process} total folders ({file})...")

            try:
                plot = KPI(file)
                plot.print_all_kpi()
            except Exception as e:
                print(f"Pass")

    print("\nFinished.\n")
