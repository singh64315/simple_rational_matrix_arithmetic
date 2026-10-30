import unittest
from fractions import Fraction

from simple_rational_matrix_arithmetic import Matrix, eye


class TestConstruction(unittest.TestCase):
    def test_int_entries_become_fractions(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m[0, 0], Fraction(1))
        self.assertEqual(m[1, 1], Fraction(4))
        self.assertIsInstance(m[0, 0], Fraction)

    def test_mixed_entry_types(self):
        m = Matrix([[Fraction(1, 2), "3/4"], [0.5, 2]])
        self.assertEqual(m[0, 0], Fraction(1, 2))
        self.assertEqual(m[0, 1], Fraction(3, 4))
        self.assertEqual(m[1, 0], Fraction(1, 2))
        self.assertEqual(m[1, 1], Fraction(2))

    def test_float_is_exact_binary_value(self):
        # 0.1 as a float is NOT 1/10; it is the exact binary value the float
        # represents.  The library preserves that rather than guessing.
        m = Matrix([[0.1]])
        self.assertEqual(m[0, 0], Fraction(0.1))
        self.assertNotEqual(m[0, 0], Fraction(1, 10))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Matrix([])

    def test_empty_row_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[]])

    def test_jaggy_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

    def test_shape_and_to_list(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.shape(), (2, 3))
        self.assertEqual(m.nrows, 2)
        self.assertEqual(m.ncols, 3)
        ll = m.to_list()
        self.assertEqual(ll, [[1, 2, 3], [4, 5, 6]])
        # to_list returns a copy
        ll[0][0] = 99
        self.assertEqual(m[0, 0], 1)

    def test_eq(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2], [3, 4]])
        c = Matrix([[1, 2], [3, 5]])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, "not a matrix")


class TestArithmetic(unittest.TestCase):
    def test_add(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        c = a + b
        self.assertEqual(c.to_list(), [[6, 8], [10, 12]])

    def test_add_fraction_results(self):
        a = Matrix([[Fraction(1, 2)]])
        b = Matrix([[Fraction(1, 3)]])
        c = a + b
        self.assertEqual(c[0, 0], Fraction(5, 6))

    def test_add_shape_mismatch(self):
        a = Matrix([[1, 2]])
        b = Matrix([[1], [2]])
        with self.assertRaises(ValueError):
            a + b

    def test_scalar_mul_left(self):
        m = Matrix([[1, 2], [3, 4]])
        r = m * 3
        self.assertEqual(r.to_list(), [[3, 6], [9, 12]])

    def test_scalar_mul_right(self):
        m = Matrix([[1, 2], [3, 4]])
        r = Fraction(1, 2) * m
        self.assertEqual(r.to_list(), [[Fraction(1, 2), 1], [Fraction(3, 2), 2]])

    def test_scalar_mul_fraction(self):
        m = Matrix([[1, 2]])
        r = m * Fraction(1, 3)
        self.assertEqual(r[0, 0], Fraction(1, 3))
        self.assertEqual(r[0, 1], Fraction(2, 3))

    def test_matrix_mul_identity(self):
        m = Matrix([[1, 2], [3, 4]])
        i = eye(2)
        self.assertEqual((m * i).to_list(), m.to_list())

    def test_matrix_mul_general(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        r = a * b
        self.assertEqual(r.shape(), (2, 2))
        self.assertEqual(r[0, 0], 1 * 7 + 2 * 9 + 3 * 11)  # 58
        self.assertEqual(r[0, 1], 1 * 8 + 2 * 10 + 3 * 12)  # 64
        self.assertEqual(r[1, 0], 4 * 7 + 5 * 9 + 6 * 11)  # 139
        self.assertEqual(r[1, 1], 4 * 8 + 5 * 10 + 6 * 12)  # 154

    def test_matrix_mul_fraction_exact(self):
        a = Matrix([[Fraction(1, 2), Fraction(1, 3)]])
        b = Matrix([[Fraction(1, 4)], [Fraction(1, 6)]])
        r = a * b
        self.assertEqual(r.shape(), (1, 1))
        self.assertEqual(r[0, 0], Fraction(1, 2) * Fraction(1, 4)
                         + Fraction(1, 3) * Fraction(1, 6))

    def test_matrix_mul_shape_mismatch(self):
        a = Matrix([[1, 2, 3]])
        b = Matrix([[1, 2]])
        with self.assertRaises(ValueError):
            a * b


class TestLinearAlgebra(unittest.TestCase):
    def test_det_1x1(self):
        self.assertEqual(Matrix([[5]]).det(), 5)

    def test_det_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.det(), -2)

    def test_det_3x3(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        self.assertEqual(m.det(), -3)

    def test_det_fraction(self):
        m = Matrix([[Fraction(1, 2), Fraction(1, 3)],
                    [Fraction(1, 4), Fraction(1, 6)]])
        # (1/2)(1/6) - (1/3)(1/4) = 1/12 - 1/12 = 0
        self.assertEqual(m.det(), 0)

    def test_det_singular(self):
        m = Matrix([[1, 2], [2, 4]])
        self.assertEqual(m.det(), 0)

    def test_det_non_square_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2, 3], [4, 5, 6]]).det()

    def test_inverse_2x2(self):
        m = Matrix([[4, 7], [2, 6]])
        inv = m.inverse()
        expected = Matrix([[Fraction(3, 5), Fraction(-7, 10)],
                           [Fraction(-1, 5), Fraction(2, 5)]])
        self.assertEqual(inv, expected)
        # Round-trip: m * inv == identity (exact)
        prod = m * inv
        self.assertEqual(prod.to_list(),
                         [[1, 0], [0, 1]])

    def test_inverse_3x3(self):
        m = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        inv = m.inverse()
        prod = m * inv
        # Exact identity
        for i in range(3):
            for j in range(3):
                expected = 1 if i == j else 0
                self.assertEqual(prod[i, j], expected)

    def test_inverse_1x1(self):
        m = Matrix([[5]])
        inv = m.inverse()
        self.assertEqual(inv[0, 0], Fraction(1, 5))
        self.assertEqual((m * inv)[0, 0], 1)

    def test_inverse_fraction_entries(self):
        m = Matrix([[Fraction(1, 2), Fraction(1, 3)],
                    [Fraction(1, 4), Fraction(1, 5)]])
        # det = 1/10 - 1/12 = 6/60 - 5/60 = 1/60, non-zero so invertible
        d = m.det()
        self.assertEqual(d, Fraction(1, 60))
        inv = m.inverse()
        prod = m * inv
        self.assertEqual(prod[0, 0], 1)
        self.assertEqual(prod[0, 1], 0)
        self.assertEqual(prod[1, 0], 0)
        self.assertEqual(prod[1, 1], 1)

    def test_inverse_singular_raises(self):
        m = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            m.inverse()

    def test_inverse_non_square_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2, 3], [4, 5, 6]]).inverse()


class TestEye(unittest.TestCase):
    def test_eye_3(self):
        i = eye(3)
        self.assertEqual(i.shape(), (3, 3))
        for r in range(3):
            for c in range(3):
                self.assertEqual(i[r, c], 1 if r == c else 0)

    def test_eye_1(self):
        i = eye(1)
        self.assertEqual(i[0, 0], 1)

    def test_eye_zero_raises(self):
        with self.assertRaises(ValueError):
            eye(0)

    def test_eye_is_identity_for_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m * eye(2), m)


if __name__ == "__main__":
    unittest.main()
