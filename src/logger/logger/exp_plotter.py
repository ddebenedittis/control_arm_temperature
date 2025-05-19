from cycler import cycler
import os

import numpy as np
from matplotlib.cm import get_cmap, viridis
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


def plot_colourline(x, y, c, vmin=25, vmax=40):
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
    def __init__(self):
        [self.x_size_def, self.y_size_def] = plt.rcParams.get('figure.figsize')

        self.foldername = 'log'
        self.joint_names = [r'$q_1$', r'$q_2$', r'$q_3$']

        self.npzfile = np.load(f"{self.foldername}/csv/log.npz")
        
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
        if name == "ee_position":
            return r"End-Effector Position [m]"

        return name
        
    def save_all_plots(self):
        q = self.npzfile['q']
        q_dot = self.npzfile['q_dot']
        T = self.npzfile['T']
        tau = self.npzfile['tau']
        
        times = np.arange(len(q)) * 0.01
        
        dict = {
            'joint_positions': q,
            'joint_velocities': q_dot,
            'joint_torques': tau,
            'temperatures': T,
        }
        
        for name, arr in dict.items():
            
            plt.figure(figsize=(self.x_size_def, self.y_size_def))
            plt.plot(times, arr)
            
            plt.xlabel('Time [s]')
            plt.ylabel(self.process_y_axis_labels(name))
            plt.xlim([times[0], times[-1]])
            
            plt.legend(self.joint_names)
            
            plt.savefig(
                os.path.join(self.foldername, 'pdf', name + ".pdf"),
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

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', titlesize=labelsize, labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)

    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle='dotted', linewidth=0.25)

    plt.rcParams['figure.constrained_layout.use'] = True
    
    plots_format = 'pdf'
    
    plot = Plot()
            
    plot.save_all_plots()
    

if __name__ == '__main__':
    main()
