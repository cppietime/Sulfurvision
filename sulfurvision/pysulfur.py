#!/bin/python3
"""
Purely Pythonic implementation of fractal generation
"""

import dataclasses
import typing

import numpy as np

def xorshift32(seed: int) -> int:
    seed ^= (seed << 13) & 0xffffffff
    seed ^= (seed >> 17) & 0xffffffff
    seed ^= (seed << 5) & 0xffffffff
    return seed

def lcg32(seed: int) -> int:
    mul = 1664525
    inc = 1013904223
    return (seed * mul + inc) & 0xffffffff

_randfunc = xorshift32

AffineTransform = tuple[float, float, float, float, float, float]
ParamsList = list[float]
Coord = tuple[float, float]
VariationFunc = typing.Callable[[Coord, AffineTransform, ParamsList], Coord]

@dataclasses.dataclass
class Variation:
    function: VariationFunc
    num_params: int
    params_base: int = dataclasses.field(init=False, default=0)
    
    variations: typing.ClassVar[list['Variation']] = []
    param_counter: typing.ClassVar[int] = 0
    
    def __post_init__(self):
        self.params_base = Variation.param_counter
        Variation.param_counter += self.num_params
        Variation.variations.append(self)
    
    def __call__(self, coord: Coord, affine: AffineTransform, params: ParamsList) -> Coord:
        return self.function(coord, affine, params[self.params_base : self.params_base + self.num_params])

variation_linear = Variation(lambda xy, affine, params: xy, 0)

def affine_transform(coord: Coord, affine: AffineTransform) -> Coord:
    return (coord[0] * affine[0] + coord[1] * affine[1] + affine[2], coord[0] * affine[3] + coord[1] * affine[4] + affine[5])

@dataclasses.dataclass
class State:
    coord: Coord
    seed: int
    color: float = 0

@dataclasses.dataclass
class Transform:
    weights: list[float]
    params: ParamsList
    affine: AffineTransform
    probability: float
    color: float
    
    def __call__(self, state: State) -> State:
        x, y = 0., 0.
        for i, weight in enumerate(self.weights):
            variation = Variation.variations[i]
            dx, dy = variation(affine_transform(state.coord, self.affine), self.affine, self.params)
            x += dx * weight
            y += dy * weight
        return State((x,  y), state.seed, (state.color + self.color) / 2)

@dataclasses.dataclass
class Flame:
    transforms: list[Transform]
    total_weight: float = dataclasses.field(init=False, default=0)
    
    def __post_init__(self):
        self.total_weight = sum(map(lambda x: x.probability, self.transforms))
    
    def iterate(self, state: State) -> State:
        state.seed = _randfunc(state.seed)
        weight = state.seed / 0x100000000 * self.total_weight
        for i, transform in enumerate(self.transforms):
            if weight >= transform.probability:
                weight -= transform.probability
                continue
            return transform(state)
        else:
            raise 'Error calculating weights'
    
    def plot(self, size: Coord, seeds: list[int], iters: int, skip: int = 20) -> np.ndarray:
        states: list[State] = []
        for seed in seeds:
            seed = _randfunc(seed)
            x = int(seed / 0x100000000) * 2 - 1
            seed = _randfunc(seed)
            y = int(seed / 0x100000000) * 2 - 1
            states.append(State((x, y), seed))
        grid = np.zeros(size)
        for itr in range(iters):
            for i, state in enumerate(states):
                state = self.iterate(state)
                states[i] = state
                if itr >= skip:
                    x, y = int((state.coord[0] + 1) / 2 * size[0]), int((state.coord[1] + 1) / 2 * size[1])
                    if 0 <= x < size[0] and 0 <= y < size[1]:
                        grid[(x, y)] = (grid[(x, y)] + state.color) / 2
        return grid
