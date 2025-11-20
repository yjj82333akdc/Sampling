import numpy as np
from scipy.special import legendre


# legendre polynomial on [0,1]
# need to add a change of basis
class polynomial:
    def __init__(self, max_degree=20) -> None:
        self.max_degree = max_degree
        self.coefficients = []

        for i in range(max_degree + 1):
            self.coefficients.append(np.array(legendre(i)) * np.sqrt(2 * i + 1))

    def compute_single_x_single_degree(self, degree, powers_x):
        return np.dot(self.coefficients[degree], powers_x[-degree - 1:])

    def compute_single_x_all_degree(self, cur_shape, x):
        powers_x = np.ones(cur_shape)
        new_x = 2 * x - 1
        for i in range(cur_shape - 2, -1, -1):
            powers_x[i] = powers_x[i + 1] * new_x

        return np.array([self.compute_single_x_single_degree(degree, powers_x) for degree in range(cur_shape)])

    def compute_integral_single_x_all_degree(self, cur_shape, x):
        """
        Analytic integrals I_n(x) = ∫_0^x ψ_n(t) dt, n = 0,...,cur_shape-1,
        where ψ_n(t) = sqrt(2n+1) * P_n(2t-1).
        """
        if x <= 0.0:
            return np.zeros(cur_shape, dtype=float)
        if x >= 1.0:
            x = 1.0

        new_x = 2.0 * x - 1.0  # s = 2x - 1

        # P_k(new_x) for k=0..cur_shape using Legendre recurrence
        P = np.empty(cur_shape + 1, dtype=float)
        P[0] = 1.0
        if cur_shape >= 1:
            P[1] = new_x
            for k in range(1, cur_shape):
                # (k+1) P_{k+1}(z) = (2k+1) z P_k(z) - k P_{k-1}(z)
                P[k + 1] = ((2 * k + 1) * new_x * P[k] - k * P[k - 1]) / (k + 1)

        integrals = np.empty(cur_shape, dtype=float)
        # n = 0: ψ_0(t) = 1 ⇒ ∫_0^x ψ_0(t) dt = x
        if cur_shape > 0:
            integrals[0] = x
        # n ≥ 1: use ∫ P_n(s) ds formula + change of variable
        for n in range(1, cur_shape):
            integrals[n] = (np.sqrt(2 * n + 1) /
                            (2.0 * (2 * n + 1))) * (P[n + 1] - P[n - 1])
        return integrals


class generate_basis_vector_given_coordinates:
    def __init__(self, dim):
        self.basis = polynomial()
        self.dim = dim

    # basis_val[d][m] is the m-th basis evaluated at x[d]

    def compute_basis_val_new_coordinates(self, x_input, tensor_shape, new_coordinates):
        basis_val = [self.basis.compute_single_x_all_degree(tensor_shape[d], x_input[new_coordinates[d]]) for d in
                     range(self.dim)]

        return basis_val


######################################################

class new_basis:
    def __init__(self, M, U) -> None:
        self.M = M
        # U mixes the original Legendre-based basis values
        self.U = np.asarray(U, dtype=float)
        self.coefficients = []

        # coefficients of ψ_i(x) = sqrt(2i+1) P_i(2x-1) in monomials of z = 2x-1
        for i in range(M):
            self.coefficients.append(
                np.concatenate(
                    (np.zeros(M - i - 1),
                     np.array(legendre(i)) * np.sqrt(2 * i + 1))
                )
            )

        # helper for analytic integrals of the original basis
        self._poly = polynomial(max_degree=M - 1)

        # coefficients of the new basis in monomial basis for fast evaluation
        self.new_coefficients = np.matmul(self.U, self.coefficients)
        self.new_length = len(self.U)

    def compute_single_x_all_new_basis(self, x):
        """Evaluate all new basis functions φ_j(x) at a single x∈[0,1]."""
        powers_x = np.ones(self.M)
        new_x = 2 * x - 1
        for i in range(self.M - 2, -1, -1):
            powers_x[i] = powers_x[i + 1] * new_x

        return np.array(
            [np.dot(self.new_coefficients[index], powers_x)
             for index in range(self.new_length)]
        )

    def compute_integral_single_x_new_basis(self, x):
        """
        Compute integrals ∫_0^x φ_j(t) dt for all new basis functions φ_j.

        If φ = U ψ, where ψ is the original Legendre-based basis, then
        ∫_0^x φ dt = U (∫_0^x ψ dt).
        """
        base_integrals = self._poly.compute_integral_single_x_all_degree(self.M, x)
        return self.U @ base_integrals



class generate_new_basis_vector:
    def __init__(self, dim, M, P_x_basis):
        self.basis = []
        for dd in range(dim):
            self.basis.append(new_basis(M, P_x_basis[dd]))
        self.dim = dim

    # basis_val[d][m] is the m-th basis evaluated at x[d]
    def compute_basis_val(self, x_input):
        basis_val = [self.basis[d].compute_single_x_all_new_basis(x_input[d]) for d in range(self.dim)]

        return basis_val

    def compute_basis_val_given_coordinates(self, x_input, new_coordinates):
        basis_val = [self.basis[d].compute_single_x_all_new_basis(x_input[new_coordinates[d]]) for d in range(self.dim)]

        return basis_val
