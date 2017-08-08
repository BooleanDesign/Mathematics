from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from colorama import Fore, Back, Style
from mpl_toolkits.mplot3d import axes3d

"""
Sympy Symbol definitions
"""
x = Symbol('x')
y = Symbol('y')

"""
Integral definitions
"""


def gradient(function):
    """
    Returns the gradient of the function.
    :param function: The function that is being calculated.
    :return: [dif_x,dif_y],[lambda_x,lambda_y]
    """
    deriv_x = diff(function, x)
    deriv_y = diff(function, y)
    return [deriv_x, deriv_y], [lambdify((x, y), deriv_x, 'numpy'), lambdify((x, y), deriv_y, 'numpy')]


def get_inputs():
    """

    :return: The function and the bounds of graphing.
    """
    function_loop = False
    while function_loop is False:
        try:
            function_loop = True
            function = input('What is the function in terms of x and y? ')
        except NameError:
            function_loop = False
            print 'The function must be in terms of x and y.'
        except SyntaxError:
            function_loop = False
            print "Syntax Error: please re-enter the function using the proper format."
    bounds_loop = False
    while bounds_loop is False:
        try:
            bounds_loop = True
            bounds = [float(element) for element in
                      raw_input('What are the bounds of the function, <min_x,max_x,min_y,max_y,min_z,max_z>? ').split(
                          ',')]
            if len(bounds) != 6:
                raise SyntaxError()
            else:
                pass
        except ValueError:
            print 'Must be float valued inputs.'
            bounds_loop = False
        except SyntaxError:
            print 'Length of bounds must be = to 6.'
            bounds_loop = False
    return [function, bounds]


def main(function, bounds):
    """

    :param function:
    :param bounds:
    :return:
    """
    """
    Print the data for the original function
    """
    functional_gradient = gradient(function)
    print 'The partial derivative of %s with respect to x is %s.' % (str(function), str(functional_gradient[0][0]))
    print 'The partial derivative of %s with respect to y is %s.' % (str(function), str(functional_gradient[0][1]))
    """
    Create the meshgrid
    """
    grid_x, grid_y = np.meshgrid(np.arange(bounds[0], bounds[1], abs(bounds[1] - bounds[0]) / 500),
                                 np.arange(bounds[2], bounds[3], abs(bounds[3] - bounds[2]) / 500))
    """
    Original Graph
    """
    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    """
    graphing the original function
    """
    lambda_function = lambdify((x, y), function, 'numpy')
    function_z = lambda_function(grid_x, grid_y)
    original_plot = ax.plot_surface(grid_x, grid_y, function_z, linewidth=0, cmap=cm.seismic)
    fig1.colorbar(original_plot)
    """
    customizing the plot
    """
    ax.axis = bounds
    ax.set_title(r'$\mathit{f(x) = z}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    """
    plotting the extra stuff
    """
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(221)  # gradient quiver plot
    ax2 = fig2.add_subplot(222, projection='3d')  # x partial
    ax3 = fig2.add_subplot(223, projection='3d')  # y partial
    ax4 = fig2.add_subplot(224)  # contour plot of f(x,y)
    n = 5  # this is the scale factor for quiver plots
    """
    Gradient Plot
    """
    quiver_grid_x, quiver_grid_y = np.meshgrid(np.arange(bounds[0], bounds[1], abs(bounds[1] - bounds[0]) / 50),
                                               np.arange(bounds[2], bounds[3], abs(bounds[3] - bounds[2]) / 50))
    quiver_u = gradient(function)[1][0](quiver_grid_x, quiver_grid_y)
    quiver_v = gradient(function)[1][1](quiver_grid_x, quiver_grid_y)
    distance = np.sqrt(quiver_u ** 2 + quiver_v ** 2) * n
    quiver = ax1.quiver(quiver_grid_x, quiver_grid_y, quiver_u / distance, quiver_v / distance, distance, angles='xy',
                        scale_units='xy', scale=1, cmap=cm.jet)
    plt.colorbar(quiver, ax=ax1)
    ax1.axis(bounds[:4])
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_title(r'$Q(x,y)= \nabla f(x,y)$')
    ax1.set_aspect('equal', adjustable='box')
    ax1.spines['bottom'].set_position('center')
    ax1.spines['left'].set_position('center')
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.grid()
    """
    X partial graph
    """
    grad = gradient(function)
    x_part_z = grad[1][0](grid_x, grid_y)
    x_part_surf = ax2.plot_surface(grid_x, grid_y, x_part_z, linewidth=0, cmap=cm.seismic)
    plt.colorbar(x_part_surf, ax=ax2)
    ax2.axis(bounds[:4])
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_title(r'$Q(x,y)= \frac{\partial f}{\partial x}$')
    ax2.set_aspect('equal', adjustable='box')
    """
    Y partial graph
    """
    y_part_z = grad[1][1](grid_x, grid_y)
    y_part_surf = ax3.plot_surface(grid_x, grid_y, y_part_z, linewidth=0, cmap=cm.seismic)
    plt.colorbar(y_part_surf, ax=ax3)
    ax3.axis(bounds[:4])
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')
    ax3.set_title(r'$Q(x,y)= \frac{\partial f}{\partial y}$')
    ax3.set_aspect('equal', adjustable='box')
    """
    Contour plot
    """
    cont = ax4.contour(grid_x, grid_y, function_z, cmap=cm.jet)
    quiver2 = ax4.quiver(quiver_grid_x, quiver_grid_y, quiver_u / distance, quiver_v / distance, distance, angles='xy',
                         scale_units='xy', scale=1, cmap=cm.jet)
    plt.colorbar(quiver2, ax=ax4)
    ax4.axis(bounds[:4])
    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$y$')
    ax4.set_title(r'$Q(x,y)= \nabla f(x,y)$')
    ax4.set_aspect('equal', adjustable='box')
    ax4.spines['bottom'].set_position('center')
    ax4.spines['left'].set_position('center')
    ax4.spines['top'].set_color('none')
    ax4.spines['right'].set_color('none')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.yaxis.set_ticks_position('left')
    ax4.grid()

    plt.show()


operation_loop = False
while operation_loop is False:
    inputs = get_inputs()
    main(inputs[0], inputs[1])
    exit_loop = False
    while exit_loop is False:
        try:
            exit_loop = True
            exit_data = raw_input('Would you like to exit the program <y/n>? ')
            if exit_data != 'y' and exit_data != 'n':
                raise ValueError('Input is Invalid.')
            elif exit_data == 'y':
                exit()
            else:
                pass
        except ValueError:
            print "The input must be either <y/n>."
            exit_loop = False
