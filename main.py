import numpy as np
from gaussian_mixture import gaussian_mixture
from tensor_estimate import vrs_prediction

from utils import metric
from kde import kernel_density
from comparison_algo import mcmc_sample_density

dim = 6
N_train = 10000
N_test = 10000
N_samples = 10000
MM = 10
lr_rec = 0
kde_rec = 0

if N_train < 2 ** dim * MM:
    print('insufficient data')
    LL = 1
else:
    LL = 2

print(MM, LL)
tensor_shape = [LL for _ in range(dim)]

tensor_shape[0] = MM

#########################################

distribution = gaussian_mixture(dim, [5, -5], [0.1, 0.1])

X_train = distribution.generate(N_train)
X_test = distribution.generate(N_test)

#############density transform
vrs_model = vrs_prediction(tensor_shape, dim, MM, X_train)
y_lr = vrs_model.predict(X_test)
y_true = np.array([distribution.density_value(xx) for xx in X_test])
lr_rec += np.linalg.norm(y_lr - y_true, 2) ** 2 / np.linalg.norm(y_true, 2) ** 2
print('lr transform prediction error = ', lr_rec )



#vrs_Sampling:
samples, time_elapsed = vrs_model.sampling_N_ori_domain(N_samples)
print("------------------below is VRS sampling result -------------------")
metric(X_test, samples, dim, N_samples, time_elapsed)
print("------------------above is VRS sampling result -------------------")



#KDE + MCMC:

y_kde = kernel_density().compute(dim, X_train, X_test)
err_kde = np.linalg.norm(y_kde - y_true, 2) ** 2 / np.linalg.norm(y_true, 2) ** 2

kde_rec += err_kde

print('kde prediction error = ', kde_rec )

kde_model = kernel_density(dim=dim, data=X_train)

def density_fn_kde(x):
    """
    Density function wrapper for MCMC, using the fitted KDE.
    x: 1D array of shape (dim,)
    """
    # evaluate expects shape (M, dim), so wrap in [x]
    return float(kde_model.evaluate(np.asarray(x, dtype=float).reshape(1, -1))[0])

samples_MCMC, time_elapsed_MCMC = mcmc_sample_density(
    density_fn=density_fn_kde,
    dim=dim,
    N_samples=N_samples,
    x0=X_train.mean(axis=0),
    proposal_std=0.8,
    burn_in=2000,
    thinning=2,
    random_state=42,
)
print("------------------below is MCMC sampling result -------------------")
metric(X_test, samples_MCMC, dim, N_samples, time_elapsed_MCMC)
print("------------------above is MCMC sampling result -------------------")
