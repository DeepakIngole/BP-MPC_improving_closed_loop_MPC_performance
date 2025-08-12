import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from src.sim_var import SimVar

# TODO: adapt to python notation convention

class Plotter:

    def __init__(self):
        pass

    @staticmethod
    def colors() -> dict:
        # save colors
        violet = (0.4940, 0.1840, 0.5560)
        blue = (0, 0.4470, 0.7410)
        orange = (0.8500, 0.3250, 0.0980)
        yellow = (0.9290, 0.6940, 0.1250)
        red = (0.6350, 0.0780, 0.1840)
        green = (0.4660, 0.6740, 0.1880)
        lblue = (0.3010, 0.7450, 0.9330)

        return {'violet':violet,'blue':blue,'orange':orange,'yellow':yellow,'red':red,'green':green,'lblue':lblue}

    @staticmethod
    def plot_trajectory(s:SimVar,options:dict=None,show:bool=False) -> None:

        if options is None:
            options = {}

        # extract options
        options = {'x':[0,1,2,3],'x_legend':['Position','Velocity','Angle','Angular velocity'],'u':[0],'u_legend':['Force'],'plot_constraints':True, 'color':'blue'} | options

        # extract colors
        colors = Plotter.colors()

        # extract state trajectory
        x = s.x
        u = s.u

        # get dimension
        T = x.shape[1] - 1

        # create time vector
        t = np.arange(T + 1)

        ### 1. STATE FIGURE

        if len(options['x']) > 0:
            # Check if figure 1 exists, if not create it
            if not plt.fignum_exists(1):
                # Create a new figure and subplots
                fig, axs = plt.subplots(len(options['x']), 1, num=1, figsize=(5, 5))
            else:
                # Use the existing figure and axes
                fig = plt.figure(1)
                axs = fig.get_axes()

            # Ensure axs is always a list (even if it's just one subplot)
            if not (isinstance(axs, np.ndarray) or isinstance(axs, list)):
                axs = [axs]

            # plot state trajectory
            for i in range(len(options['x'])):
                ax = axs[i]
                ax.plot(t, np.array(x[i, :]).squeeze(), label=options['x_legend'][i], color=colors[options['color']])
                ax.legend()
            axs[0].set_title('State trajectory')

            # adjust layout
            plt.tight_layout()

        ### 2. INPUT FIGURE

        if len(options['u']) > 0:
            # Check if figure 2 exists, if not create it
            if not plt.fignum_exists(2):
                # Create a new figure and subplots
                fig, axs = plt.subplots(len(options['u']), 1, num=2, figsize=(5, 5))
            else:
                # Use the existing figure and axes
                fig = plt.figure(2)
                axs = fig.get_axes()

            # Ensure axs is always a list (even if it's just one subplot)
            if not (isinstance(axs, np.ndarray) or isinstance(axs, list)):
                axs = [axs]

            # plot input trajectory
            for i in range(len(options['u'])):
                ax = axs[i]
                ax.plot(t[:-1], np.array(u[i, :]).squeeze(), label=options['u_legend'][i], color=colors[options['color']])
                ax.legend()
            axs[0].set_title('Input trajectory')

        # adjust layout
        plt.tight_layout()

        # update plot without blocking
        plt.draw()

        if show:
            plt.show()  # Only call plt.show() after plotting both trajectories

    @staticmethod
    def plot_car_trajectory(
        waypoints:Union[ca.DM,np.ndarray],
        tangent_direction:Union[ca.DM,np.ndarray],
        sim:SimVar,
        path_constraint:Union[ca.DM,np.ndarray]=None,
        show:bool=False,
        options:dict=None,
    ) -> None:
        
        if options is None:
            options = {}

        # extract options
        options = {'legend':'Optimal','color':'orange','color_quiver':'red','linestyle':'--'} | options

        # extract colors
        colors = Plotter.colors()
        
        # extract orthogonal distance from the center of the path
        e_cg = np.array(sim.x[0,:]).squeeze()[:-1]

        # extract orientation error
        theta_e = np.array(sim.x[2,:]).squeeze()[:-1]

        # compute absolute angle
        theta = theta_e + tangent_direction

        # extract orthogonal direction
        nx, ny = -np.sin(tangent_direction), np.cos(tangent_direction)

        # extract car orientation
        nx_car, ny_car = -np.sin(theta), np.cos(theta)

        # extract waypoints
        x_r, y_r = waypoints[0,:], waypoints[1,:]

        # obtain absolute position
        x, y = x_r + e_cg * nx, y_r + e_cg * ny

        # form lane if required
        if path_constraint is not None:
            x_lane_left, y_lane_left = x_r - path_constraint * nx, y_r - path_constraint * ny
            x_lane_right, y_lane_right = x_r + path_constraint * nx, y_r + path_constraint * ny

        # plot lane
        plt.figure(2,figsize=(5, 5))

        # plot lane
        if path_constraint is not None:

            # create coordinates to fill lane space
            x_poly = np.concatenate([x_lane_left, x_lane_right[::-1]])
            y_poly = np.concatenate([y_lane_left, y_lane_right[::-1]])

            # fill the lane
            plt.fill(x_poly, y_poly, color='tab:gray', alpha=0.2)

            # plot lane boundaries
            plt.plot(x_lane_left,y_lane_left,color='tab:gray',linewidth=1)
            plt.plot(x_lane_right,y_lane_right,color='tab:gray',linewidth=1)

        # plot absolute 
        plt.plot(x,y,label=options['legend'],linestyle=options['linestyle'],color=colors[options['color']],linewidth=2)

        # plot orientation
        # plt.quiver(x,y,nx_car,ny_car,linewidth=1.5,color=colors[options['color_quiver']])

        if show:
            plt.show()