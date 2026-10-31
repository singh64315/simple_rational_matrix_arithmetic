# simple_rational_matrix_arithmetic

Exact matrix addition, multiplication, determinant, and inverse over rational numbers, using Python's `fractions.Fraction` for every entry so there is no floating-point error.

## Usage

```python
from fractions import Fraction
from simple_rational_matrix_arithmetic import Matrix, eye

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

a + b            # Matrix([[6, 8], [10, 12]])
a * b            # Matrix([[19, 22], [43, 50]])
a * 3            # Matrix([[3, 6], [9, 12]])
3 * a            # Matrix([[3, 6], [9, 12]]) -- __rmul__
a.det()          # Fraction(-2, 1)
a.inverse()      # Matrix([[-2, 1], [3/2, -1/2]])

# Exact with fractions -- no rounding, ever
m = Matrix([[Fraction(1, 2), Fraction(1, 3)],
             [Fraction(1, 4), Fraction(1, 5)]])
m.det()          # Fraction(1, 60)
m * m.inverse()  # exactly the identity

eye(3)           # 3x3 identity matrix
```

## Why

Floating-point matrices accumulate error: a determinant that should be zero comes out as `1e-16`, and an inverse multiplied by its original is "close enough" rather than exact. For small matrices where inputs are genuinely rational (exact decimals, fractions, counts), `fractions.Fraction` gives bit-for-bit correct results with no tolerances to tune.

The trade-off is speed. Determinant and inverse use recursive cofactor expansion, which is exponential in the matrix size. This library is aimed at small matrices -- 4x4, 5x5 -- where correctness matters more than throughput.

## Edge case: floats are exact, not guessed

Passing a Python `float` stores the exact binary value that float represents, not a nearby "nice" fraction. `Matrix([[0.1]])` holds `Fraction(3602879701896397, 36028797018963968)`, not `Fraction(1, 10)`. If you want `1/10`, pass the string `"1/10"` or `Fraction(1, 10)`:

```python
Matrix([["1/10", "1/3"], ["1/7", "2/5"]])  # exact as written
```

This avoids silently guessing a decimal approximation for a value that already lost precision when it was written as a float literal.

## Exports

- `Matrix` -- constructed from `Matrix([[...], [...]])` with int, str, float, or Fraction entries. Methods: `shape()`, `to_list()`, `det()`, `inverse()`, `minor(row, col)`. Operators: `+`, `*` (scalar or matrix), `==`.
- `eye(n)` -- returns the `n x n` identity matrix.
