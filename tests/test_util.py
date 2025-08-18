from sulfurvision import util

def test_catmull():
    p = [10, 20, -5, 1.5]
    for i, f in enumerate(p):
        value = util.catmull_rom(p, i - 1)
        assert abs(value - f) <= 1e-9, f'Actual {value} did not match expected {f} for index {i}'

def main():
    test_catmull()

if __name__ == '__main__':
    main()
