import matplotlib.pyplot as plt
import numpy as np

from helmo.util.sampling import *


def test_point_from_circle():
    inputs = [(1, 0), (1, np.pi), (6, np.pi / 3), (3, 7 * np.pi / 6)]
    answers = [(1., 0.), (-1., 0.), (3., round(3 * 3 ** 0.5, 3)), (round(-1.5 * 3 ** 0.5, 3), -1.5)]
    for i, (inp, ans) in enumerate(zip(inputs, answers)):
        x, y = point_from_circle(*inp)
        x = round(x, 3)
        y = round(y, 3)
        assert (x, y) == ans, \
            "\ntest #{},\ninputs: {} {},\ncorrect answer: {},\ngiven answer: {}".format(
                i, *inp, ans, (x, y)
            )


def test_point_from_3d_sphere():
    inputs = [
        (1, 0, 0),
        (1, np.pi / 2, np.pi),
        (10, -np.pi / 4, np.pi / 4),
    ]
    answers = [
        (1.0, 0., 0.),
        (0., 0., 1.),
        (5., 5., -round(5. * 2 ** 0.5, 3))
    ]
    for i, (inp, ans) in enumerate(zip(inputs, answers)):
        x, y, z = point_from_3d_sphere(*inp)
        x = round(x, 3)
        y = round(y, 3)
        z = round(z, 3)
        assert (x, y, z) == ans, \
            "\ntest #{},\ninputs: {},\ncorrect answer: {},\ngiven answer: {}".format(
                i, inp, ans, (x, y, z)
            )


def test_point_from_4d_sphere():
    inputs = [
        (1, 0, 0, 0),
        (1, 0., np.pi / 2, np.pi),
        (10, -np.pi / 4, -np.pi / 4, np.pi / 4),
    ]
    answers = [
        (1.0, 0., 0., 0),
        (0., 0., 1., 0.),
        (
            round(2.5 * 2 ** 0.5, 3),
            round(2.5 * 2 ** 0.5, 3),
            -5.,
            -round(5. * 2 ** 0.5, 3),
        )
    ]
    for i, (inp, ans) in enumerate(zip(inputs, answers)):
        x, y, z, u = point_from_4d_sphere(*inp)
        x = round(x, 3)
        y = round(y, 3)
        z = round(z, 3)
        u = round(u, 3)
        assert (x, y, z, u) == ans, \
            "\ntest #{},\ninputs: {},\ncorrect answer: {},\ngiven answer: {}".format(
                i, inp, ans, (x, y, z, u)
            )


def test_move_to_3d_sum_plane():
    for i in range(100):
        a = random.uniform(-1000, 1000)
        b = random.uniform(-1000, 1000)
        x, y, z = move_to_3d_sum_plane(a, b)
        s = round(x + y + z, 3)
        assert s == 0., "\ntest #{},\ninputs: {},\nsum: {}".format(i, (a, b), s)


def test_move_to_4d_sum_hyperplane():
    for i in range(100):
        a = random.uniform(-1000, 1000)
        b = random.uniform(-1000, 1000)
        c = random.uniform(-1000, 1000)
        x, y, z, u = move_to_4d_sum_hyperplane(a, b, c)
        s = round(x + y + z + u, 3)
        assert s == 0., "\ntest #{},\ninputs: {},\nsum: {}".format(i, (a, b, c), s)


def test_sample_point_from_circle():
    for i in range(100):
        r = random.uniform(0.1, 10)
        x, y = sample_point_from_circle(r)
        r_given_square = round(x ** 2 + y ** 2, 3)
        correct = round(r ** 2, 3)
        assert correct == r_given_square, \
            "\ntest #{}\ninputs: {}\ngiven answer: {}\n" \
            "given square radius: {}\ncorrect square radius: {}".format(
                i, r, (x, y), r_given_square, correct
            )


def test_sample_point_from_3d_sphere():
    for i in range(100):
        r = random.uniform(0.1, 10)
        x, y, z = sample_point_from_3d_sphere(r)
        r_given_square = round(x ** 2 + y ** 2 + z ** 2, 3)
        correct = round(r ** 2, 3)
        assert correct == r_given_square, \
            "\ntest #{}\ninputs: {}\ngiven answer: {}\n" \
            "given square radius: {}\ncorrect square radius: {}".format(
                i, r, (x, y, z), r_given_square, correct
            )


def test_sample_point_from_4d_sphere():
    for i in range(100):
        r = random.uniform(0.1, 10)
        x, y, z, u = sample_point_from_4d_sphere(r)
        r_given_square = round(x ** 2 + y ** 2 + z ** 2 + u ** 2, 3)
        correct = round(r ** 2, 3)
        assert correct == r_given_square, \
            "\ntest #{}\ninputs: {}\ngiven answer: {}\n" \
            "given square radius: {}\ncorrect square radius: {}".format(
                i, r, (x, y, z, u), r_given_square, correct
            )


def test_sample_shifted_points_inside_sphere_with_constant_sum__radius():
    n = 1000
    for uniformly_by_radius in range(2):
        uniformly_by_radius = bool(uniformly_by_radius)
        for i in range(5):
            r = 100
            for nd in range(2, 5):
                point = [random.uniform(50, 200) for _ in range(nd)]
                for round_ in range(2):
                    round_ = bool(round_)
                    print(uniformly_by_radius, i, nd, round_)
                    points = sample_shifted_points_inside_sphere_with_constant_sum(
                        point, n, r, uniformly_by_radius, round_)
                    # print(point, points, sep='\n')
                    max_error = nd if round_ else 0
                    correct_sum = sum(point)
                    radiuses = [L2_norm_vec(compute_dr(p, point)) for p in points]
                    # print(radiuses[:10])
                    plt.hist(radiuses)
                    plt.show()


def test_sample_shifted_points_inside_sphere_with_constant_sum():
    for i in range(100):
        r = random.uniform(100, 200)
        nd = random.randint(2, 4)
        point = [random.uniform(50, 200) for _ in range(nd)]
        n = random.randint(1, 10)
        uniformly_by_radius = bool(random.randint(0, 1))
        round_ = bool(random.randint(0, 1))
        points = sample_shifted_points_inside_sphere_with_constant_sum(point, n, r, uniformly_by_radius, round_)
        max_error = nd if round_ else 0
        correct_sum = sum(point)
        for i, p in enumerate(points):
            s = sum(p)
            dr = [c - c_correct for c, c_correct in zip(p, point)]
            dr_mod = sum([c ** 2 for c in dr]) ** 0.5
            left = correct_sum - max_error
            right = correct_sum + max_error
            if not round_:
                left = round(left, 3)
                right = round(right, 3)
                s = round(s, 3)
            assert left <= s <= right, \
                "Sum is wrong\ntest #{}\nreturned sum: {}\ncorrect sum: {}".format(
                    i, s, correct_sum
                )
            assert dr_mod < r + max_error, \
                "Shift is to big\ntest #{}\nreturned point: {}\noriginal point: {}\n" \
                "shift length: {}\nmax shift: {}".format(
                    i, p, point, dr_mod, r + max_error
                )


if __name__ == '__main__':
    test_point_from_circle()
    test_point_from_3d_sphere()
    test_point_from_4d_sphere()
    test_move_to_3d_sum_plane()
    test_move_to_4d_sum_hyperplane()
    test_sample_point_from_circle()
    test_sample_point_from_3d_sphere()
    test_sample_point_from_4d_sphere()
    test_sample_shifted_points_inside_sphere_with_constant_sum()
    test_sample_shifted_points_inside_sphere_with_constant_sum__radius()