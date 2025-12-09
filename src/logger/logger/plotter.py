from cycler import cycler
import os

import numpy as np
from matplotlib.cm import get_cmap, viridis
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pinocchio as pin

from robot_model.robot_wrapper import RobotWrapper


def plot_colourline(x, y, c, vmin=25, vmax=35.1):
    cmap = get_cmap('viridis').copy()
    cmap.set_over('red')

    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

    col = cmap(norm(c))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                c=col[i], linestyle='-')

    im = ax.scatter(x, y, c=c, s=0, cmap=cmap, norm=norm)

    cbar = plt.colorbar(im, ax=ax, extend='max')
    cbar.set_label(r"Max $T$ [°C]")
    im.set_clim(vmin, vmax)

    return im


class Plot:
    def __init__(self, subdir: str):
        [self.x_size_def, self.y_size_def] = plt.rcParams.get('figure.figsize')

        self.foldername = 'log'
        self.joint_names = [r'$q_1$', r'$q_2$', r'$q_3$']

        self.subdir = subdir
        
        self.npzfile = np.load(f"{self.foldername}/csv/{self.subdir}/log.npz")
        
        self.ee_limits = {
            'x': {'min': 100.0, 'max': -100.0},
            'y': {'min': 100.0, 'max': -100.0},
        }
        
        self.robot_name = 'arm'
        self.robot_wrapper = RobotWrapper(self.robot_name)
        
        self.t_max = 300
        mask = self.npzfile['times'] <= self.t_max

        self.npzfile = {
            key: self.npzfile[key][mask]
            for key in self.npzfile.files
        }
        for key in self.npzfile.keys():
            self.npzfile[key] = np.where(
                self.npzfile[key] == 0, np.nan, self.npzfile[key]
            )
        
        
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
        if name == "joint_currents":
            return r"Motor Currents [A]"
        if name == "ee_position":
            return r"End-Effector Position [m]"

        return name
    
    def compute_task_space_vel(self, q: np.ndarray, q_dot: np.ndarray):
        self.robot_wrapper.forwardKinematics(q)
        pin.computeJointJacobians(self.robot_wrapper.model, self.robot_wrapper.data)
        pin.updateFramePlacements(self.robot_wrapper.model, self.robot_wrapper.data)
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        J_ee = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_ee = J_ee[[0, 2], :]
        
        return (np.linalg.pinv(J_ee) @ J_ee) @ q_dot
        
    def compute_nullspace_vel(self, q: np.ndarray, q_dot: np.ndarray):
        self.robot_wrapper.forwardKinematics(q)
        pin.computeJointJacobians(self.robot_wrapper.model, self.robot_wrapper.data)
        pin.updateFramePlacements(self.robot_wrapper.model, self.robot_wrapper.data)
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        J_ee = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_ee = J_ee[[0, 2], :]
        
        return (np.eye(3) - np.linalg.pinv(J_ee) @ J_ee) @ q_dot
    
    def save_ee_traj(self):
        ee_pos = self.npzfile['ee_position']
        ee_ref = self.npzfile['reference_position']
        temp = self.npzfile['temperatures']
        
        temp_max = np.max(temp, axis=1)
        
        fig = plt.figure(figsize=(self.x_size_def, self.y_size_def))
        ax = plt.gca()
        
        im = plot_colourline(
            ee_pos[:, 0],
            ee_pos[:, 2],
            temp_max,
        )
        
        plt.plot(
            ee_ref[:, 0], ee_ref[:, 2],
            linestyle=':', color='blue', alpha=0.5,
            # label=r'Reference Trajectory',
        )
        
        ax.set(
            xlabel=r'$x$-coordinate [m]',
            ylabel=r'$z$-coordinate [m]',
        )
        
        delta_x = self.ee_limits['x']['max'] - self.ee_limits['x']['min']
        ax.set_xlim([
            self.ee_limits['x']['min'] - delta_x * 0.05,
            self.ee_limits['x']['max'] + delta_x * 0.05,
        ])
        delta_y = self.ee_limits['y']['max'] - self.ee_limits['y']['min']
        ax.set_ylim([
            self.ee_limits['y']['min'] - delta_y * 0.05,
            self.ee_limits['y']['max'] + delta_y * 0.05,
        ])
        
        plt.savefig(
            os.path.join(self.foldername, 'pdf', self.subdir, "ee_traj.pdf"),
            bbox_inches='tight',
        )
        plt.close()
        
    def save_nullspace_vel(self):
        times = self.npzfile['times']
        q = self.npzfile['joint_positions']
        q_dot = self.npzfile['joint_velocities']
        
        nullspace_vel = np.array([
            self.compute_nullspace_vel(q[i], q_dot[i])
            for i in range(len(q))
        ])
        
        task_space_vel = np.array([
            self.compute_task_space_vel(q[i], q_dot[i])
            for i in range(len(q))
        ])
        
        fig, axs = plt.subplots(2, 1, figsize=(self.x_size_def, self.y_size_def), sharex=True)
        
        axs[0].plot(times, nullspace_vel)
        axs[0].set_ylabel(r"Nullspace Vel. [rad/s]")
        axs[0].legend([
            r'$\dot{q}_{\tiny N, 1}$',
            r'$\dot{q}_{\tiny N, 2}$',
            r'$\dot{q}_{\tiny N, 3}$',
        ])

        axs[1].plot(times, task_space_vel)
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel(r"Task Space Vel. [rad/s]")
        axs[1].legend([
            r'$\dot{q}_{\tiny T, 1}$',
            r'$\dot{q}_{\tiny T, 2}$',
            r'$\dot{q}_{\tiny T, 3}$',
        ])
        
        axs[0].set_xlim([0, 300])
        axs[1].set_xlim([0, 300])
        
        plt.savefig(
            os.path.join(self.foldername, 'pdf', self.subdir, "nullspace_vel.pdf"),
            bbox_inches='tight',
        )
        plt.close()
        
    def save_all_plots(self):
        times = self.npzfile['times']
        
        for name in sorted(self.npzfile.keys()):
            if name == 'times' or name == 'reference_position':
                continue
            
            arr = self.npzfile[name]
            
            plt.figure(figsize=(self.x_size_def, self.y_size_def))
            plt.plot(times, arr)
            
            plt.xlabel('Time [s]')
            plt.ylabel(self.process_y_axis_labels(name))
            plt.xlim([0, self.t_max])
            
            if name == 'ee_position':
                arr2 = self.npzfile['reference_position']
                plt.plot(times, arr2, linestyle=':', color='black', alpha=0.5)
            
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

    textsize = 20
    labelsize = 22

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', titlesize=labelsize, labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)

    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle='dotted', linewidth=0.25)

    plt.rcParams['figure.constrained_layout.use'] = True
    
    plt.rcParams.update({
        # Outer padding when saving
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,   # default is 0.1

        # Constrained layout padding
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.02,
        "figure.constrained_layout.wspace": 0.02,

        # Distance between axis and labels
        "axes.labelpad": 2.0,         # default ~4

        # Distance between ticks and their labels
        "xtick.major.pad": 2.0,       # default ~3.5
        "ytick.major.pad": 2.0,

        # Legend padding
        "legend.borderaxespad": 0.2,  # distance from axes
        "legend.borderpad": 0.2,      # inside the legend box
        "legend.labelspacing": 0.2,   # vertical space between entries
    })
    
    plots_format = 'pdf'
    
    # ======================================================================= #
    
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
                
    # Get x and y min and max values across all folders
    counter = 0
    ee_limits = {
        'x': {'min': 100.0, 'max': -100.0},
        'y': {'min': 100.0, 'max': -100.0},
    }
    for file in os.listdir(rootdir_csv):
        d = os.path.join(rootdir_csv, file)
        if os.path.isdir(d) \
            and len(os.listdir(os.path.join(rootdir_plt, file))) == 0:
        
            foldername = 'log'
            npzfile = np.load(f"{foldername}/csv/{file}/log.npz")
            ee_limits['x']['min'] = min(ee_limits['x']['min'], np.min(npzfile['ee_position'][:, 0]))
            ee_limits['x']['max'] = max(ee_limits['x']['max'], np.max(npzfile['ee_position'][:, 0]))
            ee_limits['y']['min'] = min(ee_limits['y']['min'], np.min(npzfile['ee_position'][:, 2]))
            ee_limits['y']['max'] = max(ee_limits['y']['max'], np.max(npzfile['ee_position'][:, 2]))

    # Process each folder
    counter = 0
    for file in os.listdir(rootdir_csv):
        d = os.path.join(rootdir_csv, file)
        if os.path.isdir(d) \
            and len(os.listdir(os.path.join(rootdir_plt, file))) == 0:

            counter += 1
            print(f"Processing the {counter}-th folder out of {n_to_process} total folders ({file})...")

            plot = Plot(file)
            plot.ee_limits = ee_limits
            
            plot.save_all_plots()
            plot.save_ee_traj()
            # plot.save_nullspace_vel()

    print("\nFinished.\n")
    

if __name__ == '__main__':
    main()
