import typing


def lerp(a: typing.Any, b: typing.Any, z: float) -> typing.Any:
    return a + (b - a) * z

def smoothstep(t: float) -> float:
    t = min(1, max(0, t))
    return 3 * t * t - 2 * t * t * t

def catmull_rom(values: typing.Sequence[any], t: float, time: typing.Sequence[float] = (-1, 0, 1, 2)) -> typing.Any:
    a1 = ((time[1] - t) * values[0] + (t - time[0]) * values[1]) / (time[1] - time[0])
    a2 = ((time[2] - t) * values[1] + (t - time[1])* values[2]) / (time[2] - time[1])
    a3 = ((time[3] - t) * values[2] + (t - time[2]) * values[3]) / (time[3] - time[2])
    b1 = ((time[2] - t) * a1 + (t - time[0]) * a2) / (time[2] - time[0])
    b2 = ((time[3] - t) * a2 + (t - time[1]) * a3) / (time[3] - time[1])
    return ((time[3] - t) * b1 + (t - time[0]) * b2) / (time[3] - time[0])
