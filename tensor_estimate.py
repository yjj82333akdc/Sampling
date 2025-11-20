from polynomial import generate_basis_vector_given_coordinates, generate_new_basis_vector
import numpy as np
from scipy.integrate import quad

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
        Compute A_predict x_{j=k+1}^D c_j^T:
        - contracts self.A_predict along modes (k+1)..(D) with vectors
          c_j = self.cur_basis[j][:,0].
        - returns the resulting tensor with modes 0..k remaining.
        """
        B = self.A_predict
        D = self.dim

        # if k is the last mode index or beyond, nothing to contract
        if k >= D:
            return B

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

    def F(self, X_, x):
        """Analytic CDF F(X_, x) = ∫_0^x p(x_k | X_) dx_k using basis integrals."""
        # trivial bounds
        if x <= 0.0:
            return 0.0
        elif x >= 1.0:
            return 1.0

        k_minus_1 = len(X_)      # number of already-fixed coordinates
        k = k_minus_1 + 1
        if k > self.dim:
            raise ValueError("Length of X_ + 1 exceeds dimension in F().")

        # denominator: marginal density p(X_)
        marginal_density = self.compute_density(X_)
        if marginal_density <= 0.0:
            return 0.0

        # 1) contract A_predict over modes j >= k, as in compute_density for length-k input
        B = self.contract_core(k)    # shape: (r_0, ..., r_{k-1})

        # 2) contract over the first k-1 coordinates with basis values at X_
        if k_minus_1 > 0:
            if 0 < k_minus_1 < self.dim:
                X_enlarged = X_ + [0.0] * (self.dim - k_minus_1)
            else:
                X_enlarged = X_
            basis_prev = self.generate_new_basis_vector.compute_basis_val(X_enlarged)[:k_minus_1]

            coeff_vec = np.tensordot(basis_prev[0], B, axes=(0, 0))
            for dd in range(1, k_minus_1):
                coeff_vec = np.tensordot(basis_prev[dd], coeff_vec, axes=(0, 0))
        else:
            # no previous coordinates: B is already the coefficient vector in dim 0
            coeff_vec = B

        # coeff_vec now has shape (r_{k-1},) and represents the coefficients of the
        # new basis φ_j for the k-th coordinate.
        # 3) Integrate φ_j from 0 to x analytically and take the dot product.
        basis_integrals_k = self.generate_new_basis_vector.basis[k_minus_1] \
            .compute_integral_single_x_new_basis(x)

        joint_integral = float(np.tensordot(basis_integrals_k, coeff_vec, axes=(0, 0)))

        F_val = joint_integral / marginal_density
        # clamp small numerical noise
        if F_val < 0.0:
            F_val = 0.0
        elif F_val > 1.0:
            F_val = 1.0
        return float(F_val)



    def F_inverse(self, X_, y, low=0.0, high=1.0, tol=1e-6, max_iter=50):
        """
        Find x_k in [0,1] such that F(X_, x_k) = y using bisection.
        Robust to tiny numerical mismatch at the endpoints by clamping y into [f_low, f_high].
        """
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))
        y = float(y)

        if y <= 0.0:
            return 0.0

        f_low = float(self.F(X_, low))
        f_high = float(self.F(X_, high))

        if abs(f_low - y) <= tol:
            return low
        if abs(f_high - y) <= tol:
            return high

        # allow small numerical slack: clamp y into [f_low, f_high] if within tol
        if y < f_low - tol:
            y = f_low
        elif y > f_high + tol:
            y = f_high
        else:
            # if y sits slightly outside due to roundoff, clamp to endpoints
            y = float(np.clip(y, f_low, f_high))

        a, b = low, high
        fa, fb = f_low, f_high
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = float(self.F(X_, m))
            if abs(fm - y) <= tol:
                return float(m)
            if fm < y:
                a, fa = m, fm
            else:
                b, fb = m, fm
            if (b - a) <= 1e-12:
                break

        return float(0.5 * (a + b))

    def sampling_one_x(self):
        X_ = []
        for i in range(self.dim):
            x_i = self.F_inverse(X_, np.random.uniform(0.0, 1.0))
            X_.append(x_i)
        return np.array(X_)

    def sampling_N(self, N):
        if N <= 0:
            raise ValueError("N must be positive.")
        rows = [np.asarray(self.sampling_one_x(), dtype=float) for _ in range(N)]
        return np.vstack(rows)  # shape: (N, dim)

    def sampling_N_ori_domain(self, N):
        Z_samples = self.sampling_N(N)
        X_samples = self.new_domain.inverse_compute_data(Z_samples)
        return X_samples