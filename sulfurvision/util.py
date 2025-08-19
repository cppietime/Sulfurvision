import typing


def lerp(a: typing.Any, b: typing.Any, z: float) -> typing.Any:
    return a + (b + a * -1) * z


def smoothstep(t: float) -> float:
    t = min(1, max(0, t))
    return 3 * t * t - 2 * t * t * t


def catmull_rom(
    values: typing.Sequence[typing.Any],
    t: float,
    time: typing.Sequence[float] = (-1, 0, 1, 2),
) -> typing.Any:
    a1 = ((time[1] - t) * values[0] + (t - time[0]) * values[1]) / (time[1] - time[0])
    a2 = ((time[2] - t) * values[1] + (t - time[1]) * values[2]) / (time[2] - time[1])
    a3 = ((time[3] - t) * values[2] + (t - time[2]) * values[3]) / (time[3] - time[2])
    b1 = ((time[2] - t) * a1 + (t - time[0]) * a2) / (time[2] - time[0])
    b2 = ((time[3] - t) * a2 + (t - time[1]) * a3) / (time[3] - time[1])
    return ((time[3] - t) * b1 + (t - time[0]) * b2) / (time[3] - time[0])


def spline_step(
    pairs: typing.Sequence[tuple[typing.Any, float]], t: float
) -> typing.Any:
    if not pairs:
        raise ValueError("Meaningless to interpolate empty sequence")
    if len(pairs) == 1:
        return pairs[0][0]
    if t <= pairs[0][1]:
        return pairs[0][0]
    if t >= pairs[-1][1]:
        return pairs[-1][0]
    times = [0.0] * len(pairs)
    time = 0
    t1 = len(pairs) - 1
    for i, (_, dt) in enumerate(pairs):
        time += dt
        times[i] = time
        if time >= t and i - 1 < t1:
            t1 = i - 1
    if t1 < 0 or t1 >= len(pairs) - 1:  # Before sequence starts
        raise ValueError("This should not be possible")
    if t1 == 0 or t1 == len(pairs) - 2:  # Lerp two elements
        return lerp(
            pairs[t1][0],
            pairs[t1 + 1][0],
            (t - pairs[t1][1]) / (pairs[t1 + 1][1] - pairs[t1][1]),
        )
    return catmull_rom(
        [pairs[x][0] for x in range(t1 - 1, t1 + 3)],
        t,
        [pairs[x][1] for x in range(t1 - 1, t1 + 3)],
    )
