#!/bin/python3
"""
Purely Pythonic implementation of fractal generation
"""

import dataclasses
import typing

import numpy as np

from sulfurvision import prng, types, util, variations


def affine_transform(coord: types.Coord, affine: types.AffineTransform) -> types.Coord:
    """Apply an affine transform to a 2D coordinate and return the result."""
    return (
        coord[0] * affine[0] + coord[1] * affine[1] + affine[2],
        coord[0] * affine[3] + coord[1] * affine[4] + affine[5],
    )


@dataclasses.dataclass
class Event:
    """A logged event of a particle having a certain color at a certain coordinate."""

    coord: types.Coord
    color: float


@dataclasses.dataclass
class State:
    """The state of one particle at one epoch. Essentially an event plus a PRNG seed."""

    coord: types.Coord
    seed: int
    color: float = 0

    def log_event(self) -> Event:
        return Event(self.coord, self.color)


@dataclasses.dataclass
class Transform:
    """A transform is a weighted list of variations plus metadata.
    Applying a transform applies the weighted sum of all variations,
    but only one transform is chosen psueod-randomly per iteration.
    """

    weights: list[float]
    # Params of all variations
    params: types.ParamsList
    # Transform applied before each variation
    affine: types.AffineTransform
    # Probability of this transform being chosen
    probability: float
    # Color applied when this transform is chosen
    color: float
    # LERP factor for color mixing
    color_speed: float = 0.5

    def __call__(self, state: State) -> State:
        x, y = 0.0, 0.0
        seed = state.seed
        transformed = affine_transform(state.coord, self.affine)
        for i, weight in enumerate(self.weights):
            if abs(weight) < 1e-9:
                continue
            variation = variations.Variation.variations[i]
            coord, seed = variation(transformed, self.affine, self.params, seed)
            dx, dy = coord
            x += dx * weight
            y += dy * weight
        return State((x, y), seed, self.__mix_color(state.color))

    def __mix_color(self, color: float) -> float:
        return util.lerp(color, self.color, self.color_speed)


@dataclasses.dataclass
class Flame:
    """All parameters of generating a flame fractal:
    All available transformations, a color palette, and an affine transform.
    """

    transforms: list[Transform]
    palette: types.Colorizer
    camera: types.AffineTransform = types.IdentityAffine
    total_weight: float = dataclasses.field(init=False, default=0)

    def __post_init__(self):
        self.total_weight = sum(map(lambda x: x.probability, self.transforms))

    def iterate(self, state: State) -> State:
        """Apply one iteration of the chaos game to one particle state."""
        state.seed, weight = prng.rand_uniform(state.seed, self.total_weight)
        for i, transform in enumerate(self.transforms):
            if weight >= transform.probability:
                weight -= transform.probability
                continue
            return transform(state)
        else:
            raise Exception("Error calculating weights")

    def iterate_step(self, states: list[State], grid: types.ImageGrid | None) -> None:
        """Performs one iteration on each state in a list, and returns the list of logged events.
        Modifies states to the new states, but does not modify any State objects passed within the list.
        """
        for i, state in enumerate(states):
            new_state = self.iterate(state)
            states[i] = new_state
            coord = affine_transform(new_state.coord, self.camera)
            if not (0 <= coord[0] < 1 and 0 <= coord[1] < 1) or grid is None:
                continue
            x, y = int(coord[0] * grid.shape[0]), int(coord[1] * grid.shape[1])
            color = self.palette(new_state.color)
            grid[x, y] += color

    def iterate_steps(
        self, initial_states: list[State], grid: types.ImageGrid, epochs: int
    ) -> None:
        for _ in range(epochs):
            self.iterate_step(initial_states, grid)

    def plot(
        self,
        size: tuple[int, int, int],
        seeds_in: list[int] | list[State] | tuple[int, int],
        iters: int,
        skip: int = 20,
    ) -> types.ImageGrid:
        """Generate an image array for this fractal."""
        # TODO supersampling
        # Populate starting states
        states: list[State] = []
        if isinstance(seeds_in, list):
            if not seeds_in:
                raise Exception("seeds_in cannot be empty list")
            if isinstance(seeds_in[0], int):
                for seed in seeds_in:
                    if not isinstance(seed, int):
                        raise Exception("All members of seeds_in must be int or State")
                    seed, cx = prng.rand_uniform(seed)
                    seed, cy = prng.rand_uniform(seed)
                    states.append(State((cx, cy), seed))
            elif isinstance(seeds_in[0], State):
                for seed in seeds_in:
                    if not isinstance(seed, State):
                        raise Exception("All members of seeds_in must be int or State")
                    states.append(seed)
        elif isinstance(seeds_in, tuple):
            n_seeds, seed_base = seeds_in
            for i in range(n_seeds):
                seed = seed_base + i
                seed, cx = prng.rand_uniform(seed)
                seed, cy = prng.rand_uniform(seed)
                states.append(State((cx, cy), seed))
        grid = np.zeros(size)
        self.iterate_steps(states, None, skip)
        no_skip = iters - skip
        self.iterate_steps(states, grid, no_skip)
        return grid
