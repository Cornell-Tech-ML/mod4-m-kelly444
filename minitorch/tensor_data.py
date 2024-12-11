from __future__ import annotations
import random
from typing import Iterable, Optional, Sequence, Tuple, Union
import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias
from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Raised when trying to access an invalid or out-of-bounds index."""

    pass


# Type aliases to make the code easier to read
Storage: TypeAlias = npt.NDArray[
    np.float64
]  # NumPy array holding floating point numbers (tensor data)
OutIndex: TypeAlias = npt.NDArray[
    np.int32
]  # NumPy array to store calculated indices (coordinates)
Index: TypeAlias = npt.NDArray[
    np.int32
]  # NumPy array representing an index (position) in the tensor
Shape: TypeAlias = npt.NDArray[
    np.int32
]  # NumPy array representing the tensor's size in each dimension
Strides: TypeAlias = npt.NDArray[
    np.int32
]  # NumPy array for calculating memory steps for each dimension
UserIndex: TypeAlias = Sequence[
    int
]  # A list of integers representing an index (coordinate)
UserShape: TypeAlias = Sequence[
    int
]  # A list of integers representing the size of each dimension
UserStrides: TypeAlias = Sequence[
    int
]  # A list of integers for the memory steps in each dimension


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multi-dimensional index to a single position in the flattened storage array.

    Parameters
    ----------
    index : Index
        The multi-dimensional index to convert into a position
    strides : Strides
        The strides representing memory steps for each dimension

    Returns
    -------
    int
        The position in the flattened array corresponding to the index

    """
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Converts a flat index (ordinal) to a multi-dimensional index (coordinates).

    Parameters
    ----------
    ordinal : int
        The flat index to convert into coordinates
    shape : Shape
        The shape of the tensor defining valid index ranges
    out_index : OutIndex
        The array where the calculated coordinates will be stored

    Returns
    -------
    None
        The multi-dimensional index is stored in out_index

    """
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Adjusts an index to match the shape of a smaller tensor for broadcasting.

    Parameters
    ----------
    big_index : Index
        The index in the larger tensor to be adjusted
    big_shape : Shape
        The shape of the larger tensor
    shape : Shape
        The shape of the smaller tensor
    out_index : OutIndex
        The array where the adjusted index will be stored

    Returns
    -------
    None
        The adjusted index is stored in out_index

    """
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Calculates a new shape that both tensors can broadcast to (compatible shape).

    Parameters
    ----------
    shape1 : UserShape
        The shape of the first tensor
    shape2 : UserShape
        The shape of the second tensor

    Returns
    -------
    UserShape
        The new shape that both tensors can be broadcasted to

    Raises
    ------
    IndexingError
        If the shapes cannot be broadcasted together

    """
    a, b = shape1, shape2
    m = max(len(a), len(b))  # Get the largest number of dimensions
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))

    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Cannot broadcast {a} and {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Cannot broadcast {a} and {b}")

    return tuple(reversed(c_rev))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Calculates the memory strides (steps) for each dimension based on the tensor's shape.

    Parameters
    ----------
    shape : UserShape
        The shape of the tensor to calculate strides for

    Returns
    -------
    UserStrides
        The calculated memory steps for each dimension

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(
            s * offset
        )  # Multiply by the size of each dimension to calculate memory step
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """A multi-dimensional tensor that stores data and provides methods for accessing and modifying it.

    Attributes
    ----------
    _storage (Storage): The raw data stored in the tensor (usually a NumPy array).
    _strides (Strides): Memory steps for each dimension to navigate the tensor in memory.
    _shape (Shape): The dimensions (size) of the tensor.
    strides (UserStrides): User-friendly strides, representing memory steps.
    shape (UserShape): The shape of the tensor.
    dims (int): The number of dimensions (axes) in the tensor.

    """

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initializes a tensor with storage (data), shape (size), and strides (memory steps).

        Parameters
        ----------
        storage : Union[Sequence[float], Storage]
            The raw data to be stored in the tensor
        shape : UserShape
            The dimensions/size of the tensor
        strides : Optional[UserStrides], optional
            The memory steps for each dimension, by default None

        Raises
        ------
        IndexingError
            If the length of strides doesn't match the shape dimensions

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be a tuple"
        assert isinstance(shape, tuple), "Shape must be a tuple"

        if len(strides) != len(shape):
            raise IndexingError(
                "Strides and shape must have the same number of dimensions."
            )

        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(list(shape)))
        self.shape = shape

        assert (
            len(self._storage) == self.size
        ), f"Storage size doesn't match the shape: {len(self._storage)} vs {self.size}"

    def to_cuda_(self) -> None:  # pragma: no cover
        """Moves tensor data to the GPU (if it's not already there).

        Returns
        -------
        None

        """
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Checks if the tensor’s memory is laid out without gaps (contiguous).

        Returns
        -------
        bool: True if memory is contiguous, False otherwise.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcasts two shapes to a compatible common shape.

        Parameters
        ----------
        shape_a : UserShape
            The shape of the first tensor to broadcast
        shape_b : UserShape
            The shape of the second tensor to broadcast

        Returns
        -------
        UserShape
            The compatible broadcasted shape

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Converts a multi-dimensional index to a position in the tensor's flattened storage array.

        Parameters
        ----------
        index : Union[int, UserIndex]
            The index to convert, either as an integer or sequence of coordinates

        Returns
        -------
        int
            The position in the flattened array

        Raises
        ------
        IndexingError
            If the index is invalid or out of bounds

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        if aindex.shape[0] != len(self.shape):
            raise IndexingError(
                "Index must match the number of dimensions in the tensor."
            )

        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError("Index out of bounds for the given shape.")
            if ind < 0:
                raise IndexingError("Negative indexing not supported.")

        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Yields all possible indices (coordinates) for the tensor.

        Returns
        -------
        Iterable[UserIndex]: An iterable of all possible indices (coordinates).

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Randomly selects an index (coordinate) in the tensor.

        Returns
        -------
        UserIndex: A randomly chosen index.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieves the value stored at a specific index in the tensor.

        Parameters
        ----------
        key : UserIndex
            The index coordinates where to retrieve the value

        Returns
        -------
        float
            The value at the specified index

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Sets the value in the tensor at a specific index.

        Parameters
        ----------
        key : UserIndex
            The index coordinates where to store the value
        val : float
            The value to store at the specified index

        Returns
        -------
        None

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns a tuple of the tensor's raw storage, shape, and strides.

        Returns
        -------
        Tuple[Storage, Shape, Strides]: A tuple containing raw data, shape, and strides.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Reorders the dimensions of the tensor (like rearranging axes).

        Parameters
        ----------
        *order : int
            Variable number of integers specifying the new order of dimensions

        Returns
        -------
        TensorData
            A new tensor with rearranged dimensions

        Raises
        ------
        AssertionError
            If the order doesn't match the number of dimensions

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must specify one position for each dimension. Order: {order}"

        return TensorData(
            self._storage,
            tuple([self.shape[o] for o in order]),
            tuple([self._strides[o] for o in order]),
        )

    def to_string(self) -> str:
        """Converts the tensor to a string representation for printing.

        Returns
        -------
        str: A string showing the tensor as a grid of values.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
