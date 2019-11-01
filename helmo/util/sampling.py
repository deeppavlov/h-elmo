import random

import numpy as np
from scipy import optimize


def point_from_circle(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def point_from_3d_sphere(r, theta, phi):
    xy = r * np.cos(theta)
    x, y = point_from_circle(xy, phi)
    return x, y, r * np.sin(theta)


def point_from_4d_sphere(r, psi, theta, phi):
    xyz = r * np.cos(psi)
    x, y, z = point_from_3d_sphere(xyz, theta, phi)
    return x, y, z, r * np.sin(psi)


def sample_theta():
    u = random.uniform(-1, 1)
    return np.arcsin(u)


def psi_equation(x, y0):
    return 0.5 * (x + 0.5 * np.sin(2*x)) - y0


def sample_psi():
    u = random.uniform(-np.pi / 4, np.pi / 4)
    sol = optimize.root_scalar(
        lambda x: psi_equation(x, u),
        method='bisect',
        bracket=[-np.pi / 2, np.pi / 2]
    )
    return sol.root


def sample_point_from_circle(r):
    return point_from_circle(r, random.uniform(0, 2*np.pi))


def sample_point_from_3d_sphere(r):
    phi = random.uniform(0, 2*np.pi)
    theta = sample_theta()
    return point_from_3d_sphere(r, theta, phi)


def sample_point_from_4d_sphere(r):
    phi = random.uniform(0, 2*np.pi)
    theta = sample_theta()
    psi = sample_psi()
    return point_from_4d_sphere(r, psi, theta, phi)


def move_to_3d_sum_plane(a, b):
    f1 = 0.5**0.5
    f2 = (1/6)**0.5
    x = f1*a + f2*b
    y = -f1*a + f2*b
    z = -2 * f2 * b
    return x, y, z


def move_to_4d_sum_hyperplane(a, b, c):
    f1 = 0.5**0.5
    f2 = 0.5
    x = f1*b + f2*c
    y = -f1*b + f2*c
    z = f1*a - f2*c
    u = -f1*a - f2*c
    return x, y, z, u


def sample_point_from_intersection_of_4d_sphere_and_sum_hyperplane(r):
    x, y, z = sample_point_from_3d_sphere(r)
    return move_to_4d_sum_hyperplane(x, y, z)


def sample_point_from_intersection_of_3d_sphere_and_sum_plane(r):
    x, y = sample_point_from_circle(r)
    return move_to_3d_sum_plane(x, y)


def sample_N_points_from_intersection_of_4d_sphere_and_sum_hyperplane(r, n):
    result = []
    for _ in range(n):
        result.append(sample_point_from_intersection_of_4d_sphere_and_sum_hyperplane(r))
    return result


def sample_N_points_from_intersection_of_3d_sphere_and_sum_plane(r, n):
    result = []
    for _ in range(n):
        result.append(sample_point_from_intersection_of_3d_sphere_and_sum_plane(r))
    return result


def sample_point_from_insides_of_sphere(r, nd, uniformly_by_radius):
    while True:
        coords = [random.uniform(0, r) for _ in range(nd)]
        dist = sum([c**2 for c in coords])**0.5
        if dist < r:
            if not uniformly_by_radius:
                return coords
            frac_dist = dist / r
            frac_dist **= nd - 1
            return [c * frac_dist for c in coords]


def sample_point_from_intersection_of_insides_of_4d_sphere_and_sum_hyperplane(r, uniformly_by_radius):
    x, y, z = sample_point_from_insides_of_sphere(r, 3, uniformly_by_radius)
    return move_to_4d_sum_hyperplane(x, y, z)


def sample_point_from_intersection_of_insides_of_3d_sphere_and_sum_plane(r, uniformly_by_radius):
    x, y = sample_point_from_insides_of_sphere(r, 2, uniformly_by_radius)
    return move_to_3d_sum_plane(x, y)


def sample_N_points_from_intersection_of_insides_of_4d_sphere_and_sum_hyperplane(r, n, uniformly_by_radius):
    result = []
    for _ in range(n):
        result.append(
            sample_point_from_intersection_of_insides_of_4d_sphere_and_sum_hyperplane(
                r,
                uniformly_by_radius,
            )
        )
    return result


def sample_N_points_from_intersection_of_insides_of_3d_sphere_and_sum_plane(r, n, uniformly_by_radius):
    result = []
    for _ in range(n):
        result.append(
            sample_point_from_intersection_of_insides_of_3d_sphere_and_sum_plane(
                r,
                uniformly_by_radius,
            )
        )
    return result


def sample_N_2d_points_with_zero_coords_sum(r, n):
    result = []
    for _ in range(n):
        shift = random.uniform(-r * 0.5**0.5, r * 0.5**0.5)
        result.append([-shift, shift])
    return result


def sample_shifted_points_inside_sphere_with_constant_sum(point, n, r, uniformly_by_radius, round_):
    nd = len(point)
    results = []
    if nd == 2:
        shifts = sample_N_2d_points_with_zero_coords_sum(r, n)
    elif nd == 3:
        shifts = sample_N_points_from_intersection_of_insides_of_3d_sphere_and_sum_plane(
            r, n, uniformly_by_radius)
    elif nd == 4:
        shifts = sample_N_points_from_intersection_of_insides_of_4d_sphere_and_sum_hyperplane(
            r, n, uniformly_by_radius)
    else:
        raise ValueError()
    for shift in shifts:
        new_point = [c+s for c, s in zip(point, shift)]
        if round_:
            new_point = [round(c) for c in new_point]
        results.append(new_point)
    return results


def compute_dr(vec1, vec2):
    return [b - a for a, b in zip(vec1, vec2)]


def L2_norm_vec(vec):
    s = 0
    for c in vec:
        s += c ** 2
    return s ** 0.5


def get_start_point(nd, dim_num, frac):
    center = [1/nd for _ in range(nd)]
    # Choosing vertex to which radius vector is directed.
    # Any vertex except for `dim_num` fits.
    dim_num = (dim_num + 1) % nd
    vertice = [0 for _ in range(nd)]
    vertice[dim_num] = 1
    radius = compute_dr(center, vertice)
    radius = [c*frac for c in radius]
    start_point = [c+d for c, d in zip(center, radius)]
    return start_point


def get_sum_prism_faces(nd, frac):
    # `faces` is a list of prism faces. Each face is a tuple of two vectors:
    # a normal to the face and a radius-vector to a point on the face.
    faces = [
        ([-1 for _ in range(nd)], [1/nd for _ in range(nd)]),
        ([1 for _ in range(nd)], [-1/nd for _ in range(nd)]),
    ]
    for dim_num in range(nd):
        normal = [-1 for _ in range(nd)]
        normal[dim_num] = nd - 1
        start_point = get_start_point(nd, dim_num, frac)
        faces.append((normal, start_point))
    return faces


def scalar_mul(vec1, vec2):
    s = 0
    for a, b in zip(vec1, vec2):
        s += a * b
    return s


def point_on_positive_side(point, plane):
    dr = compute_dr(plane[1], point)
    return scalar_mul(plane[0], dr) > 0


def point_in_polyhedron(point, faces):
    for face in faces:
        if not point_on_positive_side(point, face):
            return False
    return True


def sample_point_from_sum_prism(nd, frac):
    faces = get_sum_prism_faces(nd, frac)
    while True:
        point = [random.uniform(-1, 1) for _ in range(nd)]
        if point_in_polyhedron(point, faces):
            return point


def project_on_plane(point, plane):
    dr = compute_dr(point, plane[1])
    normal_length = L2_norm_vec(plane[0])
    dist = scalar_mul(dr, plane[0]) / normal_length
    vec = [c*dist/normal_length for c in plane[0]]
    return [a+b for a, b in zip(point, vec)]


def project_on_sum_plane(normed_point):
    nd = len(normed_point)
    start_point = [1/nd for _ in range(nd)]
    normal = [1 for _ in range(nd)]
    return project_on_plane(normed_point, (normal, start_point))


def sample_point_from_sum_triangle(s, nd, frac):
    normed_point = sample_point_from_sum_prism(nd, frac)
    normed_point = project_on_sum_plane(normed_point)
    return [c*s for c in normed_point]


def get_hidden_size(input_size, num_param):
    discriminant = 16*(input_size + 1)**2 + 16*num_param
    root = discriminant**0.5
    hidden_size = (-4*input_size - 4 + root) / 8
    return int(round(hidden_size))


def get_hidden_sizes_from_num_param(num_param, input_size):
    hidden_sizes = []
    for np in num_param:
        hs = get_hidden_size(input_size, np)
        hidden_sizes.append(hs)
        input_size = hs
    return hidden_sizes


def sample_hidden_sizes(total_num_param, frac, num_layers, input_size, n):
    """Sample sizes of hidden states which give specified number of params.
    The number of parameters for a layer is sampled from a uniform
    distribution.

    Two limitations are applied to parameter distribution. The first is that
    parameter point is laying on a hyperplane
      `total_num_param = num_layer_1 + num_layer_2 + ...`
    The second is that `num_layer_i > frac * 1 / num_layers * total_num_param`.

    Args:
        total_num_param (int): sum of numbers of parameters on layers of
            LSTM. A network parameters include neurons weights and biases.
        frac (float): the parameter defines a minimum number of parameters for
            one layer.
        num_layers (int): the number of LSTM layers.
        input_size (int): the size of the vector.
        n (int): number of samples.
    Returns:
        list of lists of int: hidden sizes of LSTM.
    """
    num_param = [
        sample_point_from_sum_triangle(total_num_param, num_layers, frac)
        for _ in range(n)
    ]
    hidden_sizes = []
    for np in num_param:
        hidden_sizes.append(get_hidden_sizes_from_num_param(np, input_size))
    return hidden_sizes
