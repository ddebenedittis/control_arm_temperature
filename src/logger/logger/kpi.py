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
    
    def compute_kpi_barrier(self):
        x_max =  0.375
        y_min = -0.425
        
        ee_position = self.npzfile['ee_position']
        
        x = ee_position[:, 0] - x_max
        x = np.clip(x, a_min=0, a_max=None)
        
        y = ee_position[:, 1] - y_min
        y = np.clip(y, a_min=None, a_max=0)
        
        return np.sum(x**2 + y**2)
    
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
        
        # kpi_barrier = self.compute_kpi_barrier()
        # print(f"Barrier KPI: {kpi_barrier}")
    
    
def main():
    # Get list of subdirectories and sort them
    rootdir_csv = "log/csv"
    subdirs = sorted([
        d for d in os.listdir(rootdir_csv)
        if os.path.isdir(os.path.join(rootdir_csv, d))
    ])

    n_to_process = len(subdirs)

    # Process each folder in sorted order
    for counter, folder in enumerate(subdirs, 1):
        print(f"\nProcessing the {counter}-th folder out of {n_to_process} total folders ({folder})...")

        try:
            plot = KPI(folder)
            plot.print_all_kpi()
        except Exception as e:
            print(f"Pass")

    print("\nFinished.\n")
