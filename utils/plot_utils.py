import matplotlib.pyplot as plt
import numpy as np

def show_3d_points(points):
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    axes.scatter(x, y, z)
    axes.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    axes.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # axes.set_ylim(-1, 1)
    axes.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # axes.set_xlim(-1, 1)
    plt.show()
    plt.close()

def show_3d_points_and_plane(points, plane):
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    axes.scatter(x, y, z)
    axes.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    axes.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    axes.set_ylim(-1, 1)
    axes.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    axes.set_xlim(-1, 1)
    axes.set_zlim(z.min(), z.min() + 2)
    A, B, C, D = plane
    
    plane_xx, plane_yy = np.meshgrid((np.array(range(20)) - 10) / 10, (np.array(range(20)) - 10) / 10)
    plane_Z = (- A * plane_xx - B * plane_yy - D) / C
    axes.plot_surface(plane_xx, plane_yy, plane_Z, alpha=0.3)

    plt.show()
    plt.close()

def show_2d_points(points):
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y, )
    plt.xlabel('Depth')
    plt.ylabel('Angle')
    plt.show()
    plt.close()