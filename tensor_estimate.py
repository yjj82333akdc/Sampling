from polynomial import generate_basis_vector_given_coordinates, generate_new_basis_vector
import numpy as np
import time

class compute_rank_one:
    # we compute rank 1 tensor given basis_val
    # the rank one tensor is
    # [p_0(x[0]), p_1(x[0]) , ...  ]\otimes [p_0(x[1]),p_1(x[1]),... ]\otimes [p_0(x[d-1]), p_1(x[d-1]),... ]
    def __init__(self, dim):
        self.dim = dim

    def compute(self, basis_val):
        temp_tensor = 1

        for dd in range(self.dim - 1, -1, -1):
            temp_tensor = np.tensordot(np.reshape(basis_val[dd], (len(basis_val[dd]), 1)), [temp_tensor], axes=(1, 0))

        # for dd in range(dim):
        #    temp_tensor= np.tensordot([temp_tensor], [basis_val[dd]],axes=[[0],[0]])
        return temp_tensor


class SVD_adaptive_thresholding:
    def __init__(self):
        pass

    def compute(self, A_temp):
        P, D = np.linalg.svd(A_temp, full_matrices=True, hermitian=False)[:2]

        # thresholding the rank
        # should be the maximum ratio!!
        # we keep at least one singular function

        cur_rank = 1
        cur_square_sum = D[0] * D[0]
        # print(D)
        # for rank in range(1,len(D)):
        for rank in range(1, len(D)):
            temp = D[rank] * D[rank]
            if temp / cur_square_sum < 1 / 100:
                break
            else:
                cur_rank = rank + 1
                cur_square_sum += temp
        # we keep everything  D[1,...,rank-1]
        # print chosen_rank to ensure that the threshold we choose is good

        ##################################### this is essential  for debugging
        ###omitted for now to run large experiments
        # print(D,cur_rank)
        # cur_rank=min(cur_rank,3)
        # print(D)
        print('selected rank =', cur_rank)
        P_transpose = P.transpose()

        P_basis = P_transpose[: cur_rank]
        # print(P_basis)

        return P_basis


class tensor:
    def __init__(self, dim, index):
        self.dim = dim
        self.generate_basis_vector_given_coordinates = generate_basis_vector_given_coordinates(dim)
        self.new_coordinate = [ii for ii in range(self.dim)]
        self.new_coordinate[0] = index
        self.new_coordinate[index] = 0
        self.compute_rank_one = compute_rank_one(dim)

    def compute_tensor(self, x_train, tensor_shape, new_coordinates):
        A_temp = np.zeros(tensor_shape, dtype=float)

        for i in range(len(x_train)):
            basis_val = self.generate_basis_vector_given_coordinates.compute_basis_val_new_coordinates(x_train[i],
                                                                                                       tensor_shape,
                                                                                                       new_coordinates)

            A_temp = A_temp + self.compute_rank_one.compute(basis_val)

        A_temp = A_temp / len(x_train)
        return A_temp

    def compute_range_basis(self, x_train, tensor_shape):
        A_temp = self.compute_tensor(x_train, tensor_shape, self.new_coordinate)
        # print(A_temp)

        A_temp = A_temp.reshape(tensor_shape[0], -1)
        return SVD_adaptive_thresholding().compute(A_temp)


from transform_domain import domain


class vrs_prediction:
    def __init__(self, tensor_shape, dim, M, X_train):

        self.compute_rank_one = compute_rank_one(dim)
        self.dim = dim
        self.new_domain = domain(dim, X_train)
        self.X_train_transform = self.new_domain.compute_data(X_train)
        self.X_train_transform = np.clip(self.X_train_transform, 0, 1)
        N_train = len(X_train)

        # compute projection basis matrix using iteration function
        self.cur_basis = []
        self.cur_shape = []
        for dd in range(dim):
            print('coordinate =', dd)

            self.cur_basis.append(tensor(dim, dd).compute_range_basis(self.X_train_transform, tensor_shape))
            self.cur_shape.append(len(self.cur_basis[-1]))

        self.generate_new_basis_vector = generate_new_basis_vector(dim, M, self.cur_basis)

        self.A_predict = np.zeros(self.cur_shape, dtype=float)
        for i in range(N_train):
            basis_val = self.generate_new_basis_vector.compute_basis_val(self.X_train_transform[i])

            self.A_predict = self.A_predict + self.compute_rank_one.compute(basis_val)

        self.A_predict = self.A_predict / N_train

        # precompute contracted cores for all prefixes k = 0..dim
        self._precompute_contracted_cores()

    def _precompute_contracted_cores(self):
        """
        Precompute B_k = A_predict x_{j>=k} c_j^T for k = 0..D.

        For each k, we contract A_predict along modes j = k, ..., D-1
        with the first column of cur_basis[j].  B_k is then reused
        in compute_density and F / F_inverse.
        """
        D = self.dim
        self._core_prefix = [None] * (D + 1)

        for k in range(D + 1):
            B = self.A_predict
            if k < D:
                for j in reversed(range(k, D)):
                    vec = self.cur_basis[j][:, 0]
                    B = np.tensordot(B, vec, axes=([j], [0]))
            self._core_prefix[k] = B

    def predict_one_x(self, X_test):
        X_test_transform = self.new_domain.compute_data(X_test)
        X_test_transform = np.clip(X_test_transform, 0, 1)
        basis_val = self.generate_new_basis_vector.compute_basis_val(X_test_transform)

        ans = np.tensordot(basis_val[0], self.A_predict, axes=(0, 0))
        for dd in range(1, self.dim):
            ans = np.tensordot(basis_val[dd], ans, axes=(0, 0))
        ans = self.new_domain.transform_density_val(ans)
        return ans

    def predict(self, X_new):
        y_lr = np.array([self.predict_one_x(xx) for xx in X_new])
        return y_lr


    def contract_core(self, k):
        """
        Return precomputed contracted core B_k.
        """
        return self._core_prefix[k]

        # contract modes j = D down to k+1
        for j in reversed(range(k, D)):
            vec = self.cur_basis[j][:, 0]
            B = np.tensordot(B, vec, axes=([j], [0]))

        return B

    def compute_density(self, X):
        '''
        X length k list, [x_1, x_2 ... x_{k-1}] have been generated as fix values,
        to compute the joint density p(x_1,x_2,...,x_k)
        '''
        if 0 < len(X) < self.dim:
            X_enlarged = X + [0.0] * (self.dim - len(X))
        elif len(X) == 0:
            return 1
        else:
            X_enlarged = X
        basis_val_k = self.generate_new_basis_vector.compute_basis_val(X_enlarged)[:len(X)]

        B = self.contract_core(len(X))
        p = np.tensordot(basis_val_k[0], B, axes=(0, 0))
        for dd in range(1, len(X)):
            p = np.tensordot(basis_val_k[dd], p, axes=(0, 0))
        return p


    def conditional_density(self, X_, x):
        '''
        X_ length k-1 list, [x_1, x_2 ... x_{k-1}] have been generated as fix values,
        to compute the conditional density p(x_k | x_1,x_2,...,x_{k-1})
        '''
        joint_density = self.compute_density(X_ + x)
        marginal_density = self.compute_density(X_)
        if marginal_density <= 0.0:
            return 0.0
        else:
            return joint_density / marginal_density


    def _conditional_info(self, X_):
        """
        For a fixed prefix X_ = (x_1,...,x_{k-1}), compute:

        - marginal_density = p(X_)
        - coeff_vec: coefficients of the basis for x_k in the expansion
        - basis_k:   new_basis object for dimension k

        This is reused many times in F_inverse.
        """
        k_minus_1 = len(X_)
        D = self.dim
        if k_minus_1 > D:
            raise ValueError("Prefix length exceeds dimension")

        k = k_minus_1 + 1

        # no previous coordinates: p(empty) := 1, coeff_vec is just B_1
        if k_minus_1 == 0:
            marginal_density = 1.0
            coeff_vec = self.contract_core(k)  # shape (r_0,)
        else:
            # enlarge X_ for basis evaluation (same as compute_density)
            if 0 < k_minus_1 < D:
                X_enlarged = X_ + [0.0] * (D - k_minus_1)
            else:
                X_enlarged = X_

            basis_prev = self.generate_new_basis_vector.compute_basis_val(X_enlarged)[:k_minus_1]

            # marginal density p(X_)
            Bm = self.contract_core(k_minus_1)      # core for modes 0..k-2
            marginal_density = np.tensordot(basis_prev[0], Bm, axes=(0, 0))
            for dd in range(1, k_minus_1):
                marginal_density = np.tensordot(basis_prev[dd], marginal_density, axes=(0, 0))

            # coefficients for the k-th coordinate
            Bk = self.contract_core(k)              # core for modes 0..k-1
            coeff_vec = np.tensordot(basis_prev[0], Bk, axes=(0, 0))
            for dd in range(1, k_minus_1):
                coeff_vec = np.tensordot(basis_prev[dd], coeff_vec, axes=(0, 0))

        basis_k = self.generate_new_basis_vector.basis[k_minus_1]
        return float(marginal_density), coeff_vec, basis_k

    def F(self, X_, x, cache=None):
        """
        Analytic CDF F(X_, x) = ∫_0^x p(x_k | X_) dx_k using basis integrals.

        Optionally accepts a precomputed cache = (marginal_density, coeff_vec, basis_k)
        from _conditional_info to avoid recomputing contractions.
        """
        # trivial bounds
        if x <= 0.0:
            return 0.0
        elif x >= 1.0:
            return 1.0

        if cache is None:
            marginal_density, coeff_vec, basis_k = self._conditional_info(X_)
        else:
            marginal_density, coeff_vec, basis_k = cache

        if marginal_density <= 0.0:
            return 0.0

        # integrate basis of k-th coordinate and take dot product
        basis_integrals_k = basis_k.compute_integral_single_x_new_basis(x)
        joint_integral = float(np.tensordot(basis_integrals_k, coeff_vec, axes=(0, 0)))

        F_val = joint_integral / marginal_density
        if F_val < 0.0:
            F_val = 0.0
        elif F_val > 1.0:
            F_val = 1.0
        return float(F_val)

    def F_inverse(self, X_, y, low=0.0, high=1.0, tol=1e-6, max_iter=50):
        """
        Find x_k in [0,1] such that F(X_, x_k) = y using bisection.

        Uses cached conditional info for this X_ so that each F-evaluation
        is O(r_k) instead of recomputing all tensor contractions.
        """
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))
        y = float(y)

        if y <= 0.0:
            return 0.0
        if y >= 1.0:
            return 1.0

        cache = self._conditional_info(X_)
        marginal_density, coeff_vec, basis_k = cache
        if marginal_density <= 0.0:
            # degenerate; fall back to uniform
            return y

        def F_cached(x):
            if x <= 0.0:
                return 0.0
            if x >= 1.0:
                return 1.0
            basis_integrals_k = basis_k.compute_integral_single_x_new_basis(x)
            joint_integral = float(np.tensordot(basis_integrals_k, coeff_vec, axes=(0, 0)))
            val = joint_integral / marginal_density
            return float(np.clip(val, 0.0, 1.0))

        f_low = F_cached(low)
        f_high = F_cached(high)

        if abs(f_low - y) <= tol:
            return low
        if abs(f_high - y) <= tol:
            return high

        # allow small slack
        if y < f_low - tol:
            y = f_low
        elif y > f_high + tol:
            y = f_high
        else:
            y = float(np.clip(y, f_low, f_high))

        a, b = low, high
        fa, fb = f_low, f_high
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = F_cached(m)
            if abs(fm - y) <= tol or (b - a) <= 1e-12:
                return float(m)
            if fm < y:
                a, fa = m, fm
            else:
                b, fb = m, fm

        return float(0.5 * (a + b))


    def sampling_one_x(self):
        X_ = []
        for i in range(self.dim):
            u = np.random.uniform(0.0, 1.0)
            x_i = self.F_inverse(X_, u)
            X_.append(x_i)
        return np.array(X_, dtype=float)

    def sampling_N(self, N):
        if N <= 0:
            raise ValueError("N must be positive.")
        Z = np.empty((N, self.dim), dtype=float)
        for n in range(N):
            Z[n, :] = self.sampling_one_x()
        return Z

    def sampling_N_ori_domain(self, N):
        start = time.perf_counter()
        Z_samples = self.sampling_N(N)
        X_samples = self.new_domain.inverse_compute_data(Z_samples)
        elapsed = time.perf_counter() - start
        return X_samples, elapsed