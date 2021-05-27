import matplotlib.pyplot as plt

def show_3d_points(points):
    print(points)
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

def show_2d_points(points):
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y, )
    plt.xlabel('Depth')
    plt.ylabel('Angle')
    plt.show()
    plt.close()