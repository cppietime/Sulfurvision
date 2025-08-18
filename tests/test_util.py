from sulfurvision import util


def test_catmull():
    p = [10, 20, -5, 1.5]
    for i, f in enumerate(p):
        value = util.catmull_rom(p, i - 1)
        assert (
            abs(value - f) <= 1e-9
        ), f"Actual {value} did not match expected {f} for index {i}"


def test_spline():
    # Test single element
    pairs_one = [(15, 0.5)]
    assert util.spline_step(pairs_one, 0) == 15
    assert util.spline_step(pairs_one, 0.5) == 15
    assert util.spline_step(pairs_one, 0) == 15

    # Test OOB with just two
    pairs_two = [(1, 0.25), (5, 0.75)]
    assert util.spline_step(pairs_two, 0) == 1
    assert util.spline_step(pairs_two, 0.25) == 1
    assert util.spline_step(pairs_two, 1) == 5
    assert util.spline_step(pairs_two, 0.75) == 5

    # Test linear interpolation
    assert util.spline_step(pairs_two, 0.5) == (1 + 5) / 2

    # Test OOB with three
    pairs_three = [(-1, 0), (5, 0.5), (3, 0.75)]
    assert util.spline_step(pairs_three, -1) == -1
    assert util.spline_step(pairs_three, 0) == -1
    assert util.spline_step(pairs_three, 0.75) == 3
    assert util.spline_step(pairs_three, 1) == 3

    # Test exact point
    assert util.spline_step(pairs_three, 0.5) == 5

    # Test lerp
    assert util.spline_step(pairs_three, 0.25) == (-1 + 5) / 2
    assert util.spline_step(pairs_three, 0.625) == (5 + 3) / 2

    # Testing full case
    pairs_many = [(2, 0), (0, 0.5), (5, 0.75), (1, 1), (-4, 3), (5, 3.5)]
    assert util.spline_step(pairs_many, -1) == 2
    assert util.spline_step(pairs_many, 0) == 2
    assert util.spline_step(pairs_many, 0.5) == 0
    assert util.spline_step(pairs_many, 0.75) == 5
    assert util.spline_step(pairs_many, 1) == 1
    assert util.spline_step(pairs_many, 3) == -4
    assert util.spline_step(pairs_many, 3.5) == 5
    assert util.spline_step(pairs_many, 4) == 5


def main():
    test_catmull()
    test_spline()


if __name__ == "__main__":
    main()
