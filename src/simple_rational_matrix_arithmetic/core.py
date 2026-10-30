"""Core rational matrix implementation.

The central design choice is to store every entry as a
:class:`fractions.Fraction`.  ``Fraction`` automatically normalises to
lowest terms and keeps the sign on the numerator, which means equality and
hashing behave predictably even when matrices are constructed from a mix of
``int``, ``str``, ``float`` and ``Fraction`` inputs.

A float such as ``0.1`` converts to ``Fraction(3602879701896397,
36028797018963968)`` -- the exact binary value the float represents -- rather
than ``Fraction(1, 10)``.  This is deliberate: if the caller already lost
precision by writing a float literal, silently guessing a nearby decimal
would hide the real input.  Callers who want ``1/10`` should pass the string
``"1/10"`` or ``Fraction(1, 10)``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, List, Sequence, Union

__all__ = ["Matrix", "eye"]

# A scalar entry may be supplied as int, str (e.g. "1/2"), float, or an
# existing Fraction.  Everything is funnelled through Fraction(...).
Entry = Union[int, float, str, Fraction]


def _to_fraction(value: Entry) -> Fraction:
    """Coerce *value* to a :class:`Fraction`.

    Repeated coercion is cheap: ``Fraction(fraction)`` returns the same
    object, so calling this on already-converted data is effectively free.
    """
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


class Matrix:
    """An immutable matrix of exact rational entries.

    Construct from a non-empty rectangular sequence of sequences::

        Matrix([[1, 2], [3, 4]])
        Matrix([[Fraction(1, 2), 0], [1, 1]])

    The internal storage is copied, so mutating the outer list after
    construction has no effect on the matrix.
    """

    __slots__ = ("_rows", "_nrows", "_ncols")

    def __init__(self, data: Sequence[Sequence[Entry]]) -> None:
        if len(data) == 0:
            raise ValueError("Matrix requires at least one row")
        if len(data[0]) == 0:
            raise ValueError("Matrix requires at least one column")

        ncols = len(data[0])
        rows: List[List[Fraction]] = []
        for row in data:
            if len(row) != ncols:
                raise ValueError(
                    f"Jaggy matrix: row lengths {ncols} and {len(row)} differ"
                )
            rows.append([_to_fraction(v) for v in row])

        # Store a private copy so callers cannot reach in and mutate.
        self._rows = rows
        self._nrows = len(rows)
        self._ncols = ncols

    # -- accessors ----------------------------------------------------------

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    def shape(self) -> "tuple[int, int]":
        """Return ``(rows, cols)``."""
        return (self._nrows, self._ncols)

    def __getitem__(self, idx: "tuple[int, int]") -> Fraction:
        i, j = idx
        return self._rows[i][j]

    def to_list(self) -> "List[List[Fraction]]":
        """Return a fresh list-of-lists of :class:`Fraction` entries.

        A copy is returned each time so the caller cannot mutate the
        matrix through the returned object.
        """
        return [list(row) for row in self._rows]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self._nrows != other._nrows or self._ncols != other._ncols:
            return False
        return self._rows == other._rows

    def __repr__(self) -> str:
        return f"Matrix({self._rows!r})"

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other: "Matrix") -> "Matrix":
        if self._nrows != other._nrows or self._ncols != other._ncols:
            raise ValueError(
                f"Cannot add matrices of shapes {self.shape()} and {other.shape()}"
            )
        return Matrix(
            [
                [self._rows[i][j] + other._rows[i][j] for j in range(self._ncols)]
                for i in range(self._nrows)
            ]
        )

    def __mul__(self, other: Union["Matrix", Entry]) -> "Matrix":
        """Right multiplication by a scalar or another :class:`Matrix`.

        Scalar multiplication keeps the shape; matrix multiplication uses
        the textbook O(n^3) algorithm, which is fine for the small sizes
        this library is intended for.
        """
        if isinstance(other, Matrix):
            if self._ncols != other._nrows:
                raise ValueError(
                    f"Cannot multiply matrices of shapes {self.shape()} "
                    f"and {other.shape()}: inner dimensions differ"
                )
            result: List[List[Fraction]] = [
                [sum(self._rows[i][k] * other._rows[k][j]
                     for k in range(self._ncols))
                 for j in range(other._ncols)]
                for i in range(self._nrows)
            ]
            return Matrix(result)
        # Scalar path.
        s = _to_fraction(other)
        return Matrix([[s * v for v in row] for row in self._rows])

    def __rmul__(self, scalar: Entry) -> "Matrix":
        return self.__mul__(scalar)

    # -- linear algebra -----------------------------------------------------

    def minor(self, row: int, col: int) -> "Matrix":
        """Return the matrix with *row* and *col* removed.

        Used by both :meth:`det` and :meth:`inverse`.  Raises ``IndexError``
        for out-of-range indices.
        """
        if not (0 <= row < self._nrows):
            raise IndexError(f"row {row} out of range for {self._nrows}-row matrix")
        if not (0 <= col < self._ncols):
            raise IndexError(f"col {col} out of range for {self._ncols}-col matrix")
        return Matrix(
            [
                [self._rows[i][j] for j in range(self._ncols) if j != col]
                for i in range(self._nrows) if i != row
            ]
        )

    def det(self) -> Fraction:
        """Exact determinant via cofactor expansion.

        Only square matrices have a determinant.  The algorithm is
        recursive cofactor expansion along the first row, which is exact but
        exponential -- again, fine for small matrices and the whole point
        here is exactness, not speed.
        """
        if self._nrows != self._ncols:
            raise ValueError(
                f"Determinant undefined for non-square matrix {self.shape()}"
            )
        n = self._nrows
        if n == 1:
            return self._rows[0][0]
        # 2x2 handled inline for clarity and to short-circuit the recursion.
        if n == 2:
            (a, b), (c, d) = self._rows
            return a * d - b * c
        total = Fraction(0)
        for j in range(n):
            total += (-1) ** j * self._rows[0][j] * self.minor(0, j).det()
        return total

    def inverse(self) -> "Matrix":
        """Exact inverse via the adjugate / cofactor method.

        Raises :class:`ValueError` if the matrix is singular (determinant
        zero) or non-square.  Each entry of the result is computed exactly,
        so the product ``m * m.inverse()`` is exactly the identity for any
        invertible ``m``.
        """
        if self._nrows != self._ncols:
            raise ValueError(
                f"Inverse undefined for non-square matrix {self.shape()}"
            )
        d = self.det()
        if d == 0:
            raise ValueError("Matrix is singular; no inverse exists")
        n = self._nrows
        if n == 1:
            return Matrix([[Fraction(1) / self._rows[0][0]]])
        # Cofactor matrix, then transpose to get the adjugate, then scale by
        # 1/det.  Cofactor(i, j) = (-1)^(i+j) * minor(i, j).det().
        cofactors = [
            [((-1) ** (i + j)) * self.minor(i, j).det() for j in range(n)]
            for i in range(n)
        ]
        adjugate = [[cofactors[j][i] for j in range(n)] for i in range(n)]
        inv_d = Fraction(1) / d
        return Matrix([[inv_d * v for v in row] for row in adjugate])


def eye(n: int) -> Matrix:
    """Return the *n* x *n* identity matrix.

    A dedicated constructor is provided because building it via
    ``Matrix([[1 if i == j else 0 ...]])`` is noisy and easy to get wrong.
    """
    if n < 1:
        raise ValueError(f"eye requires n >= 1, got {n}")
    return Matrix([[Fraction(1) if i == j else Fraction(0)
                    for j in range(n)] for i in range(n)])
