import typing


def lerp(a: typing.Any, b: typing.Any, z: float) -> float:
    return a + (b - a) * z
