from dataclasses import dataclass
from typing import List
import random
import math


@dataclass
class Dataset:
    """A Dataset is just a collection of data points (X) and their answers (y).

    - X: A list of points, where each point is made up of two numbers (like a pair of coordinates).
    - y: A list of "correct answers" (either 0 or 1) for each point in X.
    """

    X: List[List[float]]  # A list of points (like coordinates on a map).
    y: List[int]  # The answer for each point: either 0 or 1.


def make_pts(N: int) -> List[List[float]]:
    """Makes N random points, where each point is just two random numbers between 0 and 1.

    Think of it like generating random dots on a piece of paper.
    """
    return [[random.random(), random.random()] for _ in range(N)]


def simple(N: int = 50) -> Dataset:
    """Creates a simple dataset of N random points where:
    - Each point is a random dot on a 2D plane.
    - The answer (y) is 1 if the first number of the point is less than 0.5, otherwise it's 0.

    This is a very basic dataset that separates points based on their position.
    """
    X = make_pts(N)
    y = [1 if x[0] < 0.5 else 0 for x in X]
    return Dataset(X, y)


def diag(N: int = 50) -> Dataset:
    """Creates a dataset of N random points where:
    - Each point is a random dot on a 2D plane.
    - The answer (y) is 1 if the sum of the two numbers in the point is less than 0.5, otherwise it's 0.

    This creates a diagonal-like split where points that are "closer" together get a label of 1.
    """
    X = make_pts(N)
    y = [1 if x[0] + x[1] < 0.5 else 0 for x in X]
    return Dataset(X, y)


def split(N: int = 50) -> Dataset:
    """Creates a dataset where the points are random, but:
    - If the first number (x) of the point is either less than 0.2 or greater than 0.8, the answer is 1.
    - Otherwise, the answer is 0.

    This creates two separate "groups" of points.
    """
    X = make_pts(N)
    y = [1 if x[0] < 0.2 or x[0] > 0.8 else 0 for x in X]
    return Dataset(X, y)


def xor(N: int = 50) -> Dataset:
    """Creates a dataset of N random points where:
    - The answer (y) is 1 if the point follows an XOR pattern:
      - If the first number is less than 0.5 and the second is greater than 0.5, or vice versa.
      - Otherwise, the answer is 0.

    This is a tricky dataset that can't be solved with a straight line, so it's used to test more complex algorithms.
    """
    X = make_pts(N)
    y = [
        1 if ((x[0] < 0.5 and x[1] > 0.5) or (x[0] > 0.5 and x[1] < 0.5)) else 0
        for x in X
    ]
    return Dataset(X, y)


def circle(N: int = 50) -> Dataset:
    """Creates a dataset where:
    - Each point is a random dot.
    - The answer (y) is 1 if the point is outside a small circle in the middle, and 0 if it's inside the circle.

    This makes a set of points either inside or outside a circle.
    """
    X = make_pts(N)
    y = [1 if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 > 0.1 else 0 for x in X]
    return Dataset(X, y)


def spiral(N: int = 50) -> Dataset:
    """Creates a dataset with two spirals:
    - The first spiral gets a label of 0.
    - The second spiral gets a label of 1.

    Each spiral is made up of points that follow a curved pattern in 2D space.
    This is a more complicated dataset that's used to test algorithms that can handle curvy shapes.
    """

    def spiral_x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def spiral_y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = []
    for i in range(5, 5 + N // 2):
        t = 10.0 * (float(i) / (N // 2))
        X.append([spiral_x(t) + 0.5, spiral_y(t) + 0.5])

    for i in range(5, 5 + N // 2):
        t = 10.0 * (float(i) / (N // 2))
        X.append([spiral_y(-t) + 0.5, spiral_x(-t) + 0.5])

    labels = [0] * (N // 2) + [1] * (N // 2)
    return Dataset(X, labels)


# A dictionary that stores all the functions for generating different types of datasets.
datasets = {
    "simple": simple,  # A simple dataset where the label depends on the first number of the point.
    "split": split,  # A dataset that splits points based on the x-coordinate.
    "xor": xor,  # A dataset that follows an XOR pattern, often used for testing.
    "diag": diag,  # A dataset where the label depends on the sum of the two numbers in the point.
    "circle": circle,  # A dataset that classifies points inside or outside a circle.
    "spiral": spiral,  # A dataset made up of two spirals.
}
