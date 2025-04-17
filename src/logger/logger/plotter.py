from cycler import cycler
import os

import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, subdir: str):
        [self.x_size_def, self.y_size_def] = plt.rcParams.get('figure.figsize')

        self.foldername = 'log'
        self.joint_names = [r'$q_1$', r'$q_2$', r'$q_3$']

        self.subdir = subdir
        
        self.npzfile = np.load(f"{self.foldername}/csv/{self.subdir}/log.npz")
        
    @staticmethod
    def process_y_axis_labels(name):
        if name == "joint_positions":
            return r"Joint Positions [rad]"
        if name == "joint_velocities":
            return r"Joint Velocities [rad/s]"
        if name == "joint_torques":
            return r"Torques [Nm]"
        if name == "temperatures":
            return r"Temperatures [°C]"

        return name
        
    def save_all_plots(self):
        times = self.npzfile['times']
        
        for name in sorted(self.npzfile.files):
            if name == 'times':
                continue
            
            arr = self.npzfile[name]
            
            plt.figure(figsize=(self.x_size_def, self.y_size_def))
            plt.plot(times, arr)
            
            plt.xlabel('Time [s]')
            plt.ylabel(self.process_y_axis_labels(name))
            plt.xlim([times[0], times[-1]])
            
            plt.legend(self.joint_names)
            
            plt.savefig(
                os.path.join(self.foldername, 'pdf', self.subdir, name + ".pdf"),
                bbox_inches='tight',
            )
            plt.close()

def main():
    # ======================== Plots Style Definition ======================== #

    default_cycler = (
        cycler(color=['#0072BD', '#D95319', '#EDB120', '#7E2F8E']) +
        cycler('linestyle', ['-', '--', '-', '--'])
    )

    textsize = 16
    labelsize = 18

    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', titlesize=labelsize, labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)

    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle='dotted', linewidth=0.25)

    plt.rcParams['figure.constrained_layout.use'] = True
    
    plots_format = 'pdf'
    
    # Get the number of folders to process and create the output directories
    rootdir_csv = "log/csv"
    rootdir_plt = "log/" + plots_format
    n_to_process = 0
    for file in os.listdir(rootdir_csv):
        d = os.path.join(rootdir_csv, file)
        if os.path.isdir(d):
            os.makedirs(os.path.join(rootdir_plt, file), exist_ok=True)
            if len(os.listdir(os.path.join(rootdir_plt, file))) == 0:
                n_to_process += 1

    # Process each folder
    counter = 0
    for file in os.listdir(rootdir_csv):
        d = os.path.join(rootdir_csv, file)
        if os.path.isdir(d) \
            and len(os.listdir(os.path.join(rootdir_plt, file))) == 0:

            counter += 1
            print(f"Processing the {counter}-th folder out of {n_to_process} total folders ({file})...")

            plot = Plot(file)
            plot.save_all_plots()

    print("\nFinished.\n")
    

if __name__ == '__main__':
    main()
