from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from SPM_utilities import ParticleCloud, Slick


def generate_mesh(parcels, x_num=50, y_num=50, z_num=50):
    if not(isinstance(x_num, int) and isinstance(y_num, int) and isinstance(z_num, int)):
        print(f'Invalid Types of Arguments - x:{type(x_num)}, y:{type(y_num)}, z:{type(z_num)}')
        raise TypeError('Incompatible types of arguments, must be integers')

    x, y, z = [], [], []
    for parcel in parcels:
        x.append(parcel.x)
        y.append(parcel.y)
        z.append(parcel.z)

    x_left_bound, x_right_bound, y_left_bound, y_right_bound, z_top_bound, z_bottom_bound \
        = min(x), max(x), min(y), max(y), min(z), max(z)

    extent = [x_left_bound, x_right_bound, y_left_bound,
              y_right_bound, z_bottom_bound, z_top_bound]

    print('Meshgrid ranging from Latitude: {0}-{1}, Longitude: {2}-{3}, Depth: {4}-{5} meters'.
          format(y_left_bound, y_right_bound, x_left_bound,
                 x_right_bound, z_bottom_bound, z_top_bound))

    element_volume = abs(x_left_bound - x_right_bound) / x_num * 111.1 * 1e3\
                     * abs(y_left_bound - y_right_bound) / y_num * 111.1 * 1e3\
                     * abs(z_bottom_bound - z_top_bound) / z_num

    meshgrid = []
    x_ticks = np.linspace(x_left_bound, x_right_bound, x_num+1)
    y_ticks = np.linspace(y_left_bound, y_right_bound, y_num+1)
    z_ticks = np.linspace(z_bottom_bound, z_top_bound, z_num+1)

    for i in zip(x_ticks[0:-1], x_ticks[1:]):
        for j in zip(y_ticks[0:-1], y_ticks[1:]):
            for k in zip(z_ticks[0:-1], z_ticks[1:]):
                grid = {'x_bound': i, 'y_bound': j, 'z_bound': k, 'parcels': [],
                        'm_PAHs': 0, 'C_PAHs': 0, 'age': 0, 'A': 0}
                meshgrid.append(grid)

    for parcel in parcels:
        for grid in meshgrid:
            x_left, x_right, y_left, y_right, z_bottom, z_top = \
                grid['x_bound'][0], grid['x_bound'][-1], \
                grid['y_bound'][0], grid['y_bound'][-1], \
                grid['z_bound'][0], grid['z_bound'][-1]
            if parcel.x is not None:
                if (x_left <= parcel.x <= x_right) and (y_left <= parcel.y <= y_right) \
                        and (z_top <= parcel.z <= z_bottom):
                    grid['parcels'].append(parcel)
                    if isinstance(parcel, ParticleCloud):
                        grid['m_PAHs'] += sum(parcel.m_Cloud[[-13, -11, -9, -7]])
                        grid['age'] += parcel.age
                    elif isinstance(parcel, Slick):
                        grid['m_PAHs'] += sum(parcel.m[[-13, -11, -9, -7]])
                        grid['A'] += parcel.A
                        grid['age'] += parcel.age

                    break

    return meshgrid, element_volume, extent


def compute_PAHs(meshgrid, grid_volume):
    #  1 kg / m3 = 1e6 ug / L

    for grid in meshgrid:
        if len(grid['parcels']) != 0:
            grid['C_PAHs'] = grid['m_PAHs'] / grid_volume * 1e6


def plot_PAHs(meshgrid, extent):

    total_bins = len(meshgrid)
    bins = ceil(total_bins ** (1/3))
    x_num, y_num, z_num = bins, bins, bins
    x_left, x_right, y_left, y_right, z_bottom, z_top = extent

    C_PAHs = np.zeros((z_num, x_num, y_num))

    count = 0
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                C_PAHs[k, i, j] = meshgrid[count]['C_PAHs']
                count += 1

    # C_bottom = C_PAHs[0, :, :]
    C_middle = C_PAHs[ceil(z_num/2), :, :]
    C_top = C_PAHs[-1, :, :]
    print(np.max(C_middle), np.max(C_top))
    # data_max = max(np.max(C_bottom), np.max(C_middle), np.max(C_top))
    # data_min = min(np.min(C_bottom), np.min(C_middle), np.min(C_top))
    # print(ceil(z_num/2))
    # print(data_min, data_max)
    # print(np.max(C_bottom), np.min(C_bottom))
    # print(np.max(C_middle), np.min(C_middle))
    # print(np.max(C_top), np.min(C_top))

    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')

    # Create mesh.
    X = np.arange(x_left, x_right, (x_right - x_left) / x_num)
    Y = np.arange(y_left, y_right, (y_right - y_left) / y_num)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros_like(X)

    norm = plt.Normalize(0, np.max(C_top))
    cm = plt.cm.get_cmap('Reds')
    # facecolors_top = cm(norm(C_top))
    # facecolors_mid = cm(norm(C_middle))
    # facecolors_bottom = cm(norm(C_bottom))

    facecolors_top = cm(C_top)
    facecolors_mid = cm(C_middle)
    # facecolors_bottom = cm(C_bottom)

    ax.plot_surface(X, Y, Z + z_top, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_top)
    ax.plot_surface(X, Y, Z + z_bottom/2, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_mid)
    # ax.plot_surface(X, Y, Z + z_bottom, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_bottom)

    # Add a colorbar to the plot
    mappable = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    mappable.set_array(C_PAHs)
    cbar = plt.colorbar(mappable, fraction=0.03, pad=0.1)
    cbar.set_label(label='PAHs concentration (μg/$L^3$)', size='18')
    cbar.ax.tick_params(labelsize=18)
    plt.gca().invert_zaxis()
    ax.set_xlabel("Lon", fontdict={'size': 18})
    ax.set_ylabel("Lat", fontdict={'size': 18})
    ax.set_zlabel("Depth", fontdict={'size': 18})
    ax.set_zlim(800, 0)
    elev = 25
    azim = -49
    ax.view_init(elev, azim)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    plt.show()


def plot_area(meshgrid, extent):

    total_bins = len(meshgrid)
    bins = ceil(total_bins ** (1/3))
    x_num, y_num, z_num = bins, bins, bins
    x_left, x_right, y_left, y_right, z_bottom, z_top = extent

    area = np.zeros((1, x_num, y_num))

    count = 0
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                area[0, i, j] = meshgrid[count]['A']
                count += 1

    data_max = np.max(area)
    data_min = np.min(area)
    print(np.max(area), np.min(area))

    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')

    # Create mesh.
    X = np.arange(x_left, x_right, (x_right - x_left) / x_num)
    Y = np.arange(y_left, y_right, (y_right - y_left) / y_num)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros_like(X)

    cm = plt.cm.get_cmap('Reds')
    facecolors_top = cm(area[0, :, :])

    ax.plot_surface(X, Y, Z + z_top, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_top)

    # Add a colorbar to the plot
    mappable = plt.cm.ScalarMappable(cmap=cm)
    mappable.set_array(area)
    cbar = plt.colorbar(mappable, fraction=0.03) # , norm=norm
    cbar.set_label(label='Exposed area ($m^2$)', size='18')

    plt.gca().invert_zaxis()
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_zlabel("Depth")

    ax.set_xlim()
    ax.set_ylim()
    ax.set_zlim(800, 0)

    elev = 25
    azim = -49
    ax.view_init(elev, azim)

    plt.show()


def plot_age(meshgrid, extent):

    total_bins = len(meshgrid)
    bins = ceil(total_bins ** (1/3))
    x_num, y_num, z_num = bins, bins, bins
    x_left, x_right, y_left, y_right, z_bottom, z_top = extent

    age = np.zeros((z_num, x_num, y_num))

    count = 0
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                age[k, i, j] = meshgrid[count]['age']
                count += 1

    age_bottom = age[0, :, :]
    age_middle = age[ceil(z_num/2), :, :]
    age_top = age[-1, :, :]

    data_max = np.max(age)
    data_min = np.min(age)
    print(np.max(age), np.min(age))

    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')

    # Create mesh.
    X = np.arange(x_left, x_right, (x_right - x_left) / x_num)
    Y = np.arange(y_left, y_right, (y_right - y_left) / y_num)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros_like(X)

    cm = plt.cm.get_cmap('Reds')
    facecolors_top = cm(age_top)
    facecolors_mid = cm(age_middle)
    facecolors_bottom = cm(age_bottom)

    ax.plot_surface(X, Y, Z + z_top, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_top)
    ax.plot_surface(X, Y, Z + z_bottom/2, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_mid)
    ax.plot_surface(X, Y, Z + z_bottom, rstride=1, cstride=1, alpha=0.5, facecolors=facecolors_bottom)

    # Add a colorbar to the plot
    mappable = plt.cm.ScalarMappable(cmap=cm)
    mappable.set_array(age)
    cbar = plt.colorbar(mappable, fraction=0.03) # , norm=norm
    cbar.set_label(label='Age ($h$)', size='18')

    plt.gca().invert_zaxis()
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_zlabel("Depth")

    ax.set_xlim()
    ax.set_ylim()
    ax.set_zlim(800, 0)

    elev = 25
    azim = -49
    ax.view_init(elev, azim)

    plt.show()


def plot_cuboid(ax, bound, bin_size=20, line_width=0.5):

    x_left, x_right, y_left, y_right, z_bottom, z_top = \
        bound[0], bound[1], bound[2], bound[3], bound[4], bound[5]
    ox, oy, oz = x_left + abs(x_left - x_right) / 2, y_left + abs(y_left - y_right) / 2, (z_bottom - z_top) / 2,
    l, w, h = abs(x_right - x_left), abs(y_right - y_left), abs(z_bottom - z_top)

    x = np.linspace(ox-l/2,ox+l/2, num=bin_size+1)
    y = np.linspace(oy-w/2,oy+w/2, num=bin_size+1)
    z = np.linspace(oz-h/2,oz+h/2, num=bin_size+1)

    x1, z1 = np.meshgrid(x, z)

    y11 = np.ones_like(x1)*(oy-w/2)
    y12 = np.ones_like(x1)*(oy+w/2)

    x2, y2 = np.meshgrid(x, y)

    z21 = np.ones_like(x2)*(oz-h/2)
    z22 = np.ones_like(x2)*(oz+h/2)

    y3, z3 = np.meshgrid(y, z)

    x31 = np.ones_like(y3)*(ox-l/2)
    x32 = np.ones_like(y3)*(ox+l/2)

    # outside surface
    ax.plot_wireframe(x1, y11, z1, color='b', rstride=1, cstride=1, alpha=0.6, linewidth=line_width)
    # inside surface
    ax.plot_wireframe(x1, y12, z1, color='b', rstride=1, cstride=1, alpha=0.6, linewidth=line_width)
    # bottom surface
    ax.plot_wireframe(x2, y2, z21, color='b', rstride=1, cstride=1, alpha=0.6, linewidth=line_width)
    # upper surface
    ax.plot_wireframe(x2, y2, z22, color='b', rstride=1, cstride=1, alpha=0.6, linewidth=line_width)
    # left surface
    ax.plot_wireframe(x31, y3, z3, color='b', rstride=1, cstride=1, alpha=0.6, linewidth=line_width)
    # right surface
    ax.plot_wireframe(x32, y3, z3, color='b', rstride=1, cstride=1, alpha=0.6, linewidth=line_width)

    plt.show()

