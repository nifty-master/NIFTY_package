import jax
import jax.numpy as jnp
from jax import random, vmap
import numpyro 
from numpyro.infer import MCMC, NUTS, init_to_value
from matplotlib import pyplot as plt  
import numpyro.distributions as dist
from numpyro.distributions import constraints
import time
import numpy as np
import scipy.stats as stats
from jax.scipy.linalg import eigh
from jax.scipy.linalg import svd 
print(jax.devices()) 
def soft_threshold_lambda(Lambda, tau): 
    # Compute column 2-norms
    col_norms = jnp.linalg.norm(Lambda, axis=0) 
    scale_factors = jnp.where(col_norms >= tau, (col_norms - tau) / col_norms, 0)
    Lambda_transformed = Lambda * scale_factors 
    return Lambda_transformed, np.sum(col_norms > tau)
def diffusion_map(y, eps, q, C):
    n = y.shape[0]  # Number of data points
    
    # Step 1: Compute pairwise squared Euclidean distance using broadcasting
    diff = y[:, None, :] - y[None, :, :]  # Shape: (n, n, m)
    distances = jnp.sum(diff ** 2, axis=2)  # Shape: (n, n), squared Euclidean distances

    # Step 2: Apply Gaussian kernel and apply thresholding based on eps * C^2
    Kernel = jnp.where(distances < eps * C**2, jnp.exp(-distances / eps), 0.0)

    # Step 3: Row normalization of Kernel matrix
    rowsum_A = jnp.sum(Kernel, axis=1)
    A = Kernel / rowsum_A[:, None]  # Normalize each row

    # Step 4: Compute the Laplacian matrix
    D1 = jnp.diag(1 / jnp.sum(A, axis=0))  # D1 is the inverse of column sums
    L = D1 @ A - jnp.eye(n)  # Normalize the Laplacian matrix
    
    # Step 5: Perform SVD to extract eigenvectors and eigenvalues
    svdL = jnp.linalg.svd(L)
    y_embed = svdL[0][:, -(q+1):-1]  
    return y_embed
def diffusion_dimension(y, eps, ratio, qmax = 10):
    n, p = y.shape  # n: number of data points, p: number of features
    
    # Step 1: Compute pairwise squared Euclidean distance using broadcasting
    diff = y[:, None, :] - y[None, :, :]  # Shape: (n, n, p)
    distances = jnp.linalg.norm(diff, axis=2)  # Shape: (n, n), squared Euclidean distances

    # Step 2: Apply Gaussian kernel and apply thresholding based on eps * C^2
    mask = distances < eps
    diff_masked = diff * mask[:, :, None]  # Apply mask to the difference matrix
    local_cov =  jnp.einsum('ijk,ijl->ikl', diff_masked, diff_masked) 
    eigvals = np.linalg.eigh(local_cov)[0]  # Shape (n, d), sorted in ascending order

    # Step 6: Extract the largest q eigenvalues and take the average over all k
    lam_mean = np.mean(eigvals[:, -qmax:], axis=0)
    q = len(lam_mean) 
    # Step 5: Find the dimension where the eigenvalue ratio falls below the threshold
    for i in range(q - 1):
        if lam_mean[q-i-2] / lam_mean[q-i-1] < ratio:
            return (i+1)
    
    # If no dimension is found where the ratio condition is met, return the full dimension
    return len(lam_mean)
class CoReUnif(dist.Distribution):
    arg_constraints = {"alpha": constraints.positive}
    support = constraints.unit_interval  # Enforces u in [0, 1]
    
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
        super().__init__(batch_shape=(n,), event_shape=())

    def sample(self, key, sample_shape=()):
        # Uniform sampling over [0, 1]
        return jax.random.uniform(key, shape=sample_shape + (self.n,))

    def log_prob(self, u):
        # Check if u is within [0, 1]
        if self._validate_args:
            self._validate_sample(u)
        
        # Compute ranks (1-based indexing in ranks)
        #ranks = jnp.argsort(jnp.argsort(u)) + 1
        
        temp = u.argsort(axis=1)  # argsort along each row
        ranks = jnp.empty_like(temp)
        ranks = ranks.at[jnp.arange(u.shape[0])[:, None], temp].set(jnp.arange(u.shape[1]))

        ranks_normalized = ranks / self.n

        # Compute penalty term
        penalty = jnp.sum((u - ranks_normalized) ** 2) 
        # Return log probability
        return -self.alpha * (penalty)
    

def compute_upiece(u, W, L):
    """
    Translate the R function to Python.
    Args:
        u: (n, K) array of uniform random variables
        W: (H, K) weight matrix (stack of diagonal matrices, H = dK)
        L: Number of bins
    Returns:
        weighted_u_piece: (L, n, H) array
    """
    # Getting the dimensions
    n, K = u.shape
    H = W.shape[0]  # H = dK
    
    # Expand u to match L bins
    u_expand = jnp.repeat(u[None, :, :], L, axis=0)  # Shape: (L, n, K)
    
    # Create threshold arrays
    thresholds_upper = jnp.linspace(1 / L, 1, L).reshape(L, 1, 1)  # Shape: (L, 1, 1)
    thresholds_lower = jnp.linspace(0, (L - 1) / L, L).reshape(L, 1, 1)  # Shape: (L, 1, 1)
    
    # Compute u_1, u_star, and u_2
    u_1 = (u_expand >= thresholds_upper).astype(float)  # Shape: (L, n, K)
    u_star = (u_expand >= thresholds_lower).astype(float)  # Shape: (L, n, K)
    u_2 = (u_expand < thresholds_upper).astype(float)  # Shape: (L, n, K)
    
    # Compute u_piece
    u_piece = u_1 / L + (u_star * u_2) * (u_expand % (1 / L))  # Shape: (L, n, K)
    
    # Expand W for alignment
    W_expanded = W[None, None, :, :]  # Shape: (1, 1, H, K)
    W_expanded = jnp.tile(W_expanded, (L, n, 1, 1))  # Shape: (L, n, H, K)
    
    # Reshape u_piece for alignment with W
    u_piece_expanded = u_piece[:, :, None, :]  # Shape: (L, n, 1, K)
    
    # Weighted sum: sum over the last axis (K)
    weighted_u_piece = jnp.sum(u_piece_expanded * W_expanded, axis=-1)  # Shape: (L, n, H)
    
    return weighted_u_piece

def compute_eta(alpha, weighted_u, L, H, N):
    # Replicate alpha across the N axis (axis 2) to match the dimensions of weighted_u
    alpha_expanded = jnp.repeat(alpha[:, None, :], N, axis=1)  # Shape: (L, N, H)

    # Element-wise multiplication of alpha_expanded with weighted_u
    weighted_u_alpha = alpha_expanded * weighted_u  # Shape: (L, n, H)

    # Sum along axes (0, 2) to reduce dimensions (L, n, H) --> (H, N)
    eta = jnp.sum(weighted_u_alpha, axis=0)  # Shape: (n, H) 
    return eta
def model(y, ystar, W, K, H, L, lam, sig_Gamma = .1, sig_slopes = 1., a_sigma = 10,
          b_sigma = .5, sparsity = False, CoRe = True, thresh=1, sig2star=.1):
    ##lam is the constraint relaxation parameter assigned on u. 
    n, p = y.shape    
    if sparsity == True:
        Gamma = numpyro.sample('Gamma', dist.Laplace(jnp.zeros((p, H)), sig_Gamma))  
        #tau = numpyro.sample('tau', dist.Exponential(.0001))
        tau = numpyro.sample('tau', dist.TruncatedNormal(low=sig_Gamma*jnp.sqrt(p)*thresh, scale=100.))
        Lambda, nfactor = soft_threshold_lambda(Gamma, tau) 
        numpyro.deterministic('Lambda', Lambda)  # Save eta in the posterior samples  
        numpyro.deterministic("nfactors", nfactor)
    else:
        Gamma = numpyro.sample('Gamma', dist.Normal(jnp.zeros((p, H)), sig_Gamma))  
        Lambda = Gamma
        numpyro.deterministic('Lambda', Lambda)  
    if CoRe == True:
        CoReUnif_k = dist.Independent(CoReUnif(lam, n), reinterpreted_batch_ndims=1)
        u = numpyro.sample("u", CoReUnif_k, sample_shape=(K,))   
    else: 
        u = numpyro.sample('u', dist.Uniform(0, 1).expand([K, n])) 
    weighted_u = compute_upiece(u.T, W, L) 
    slopes = numpyro.sample('slopes', dist.Normal(loc=jnp.zeros((L, H)), scale=sig_slopes )) 
    slopestar = numpyro.sample('slopestar', dist.Normal(loc=jnp.zeros((L, K)), scale=sig_slopes)) 
    sigma2 = .01#numpyro.sample('sigma2', dist.InverseGamma(a_sigma, b_sigma)) 


    
    eta = compute_eta(slopes, weighted_u, L, H, n)  
    numpyro.deterministic('eta', eta)  # Save eta in the posterior samples  
    etastar = compute_eta(slopestar, compute_upiece(u.T, jnp.eye(K), L), L, K, n)  
    unew = numpyro.sample('unew', dist.Uniform(0, 1).expand([K, n])) 
    fitted = eta @ Lambda.T 
    numpyro.deterministic('fitted', fitted) 
    numpyro.sample('y', dist.Normal(fitted, jnp.sqrt(sigma2)), obs=y) 
    weighted_u = compute_upiece(u.T, W, L) 
    etanew = compute_eta(slopes, weighted_u, L, H, n)  
    generated = etanew@Lambda.T + numpyro.sample('noise', dist.Normal(jnp.zeros(fitted.shape), jnp.sqrt(sigma2)))
    numpyro.deterministic("generated", generated)
    numpyro.sample('ystar', dist.Normal(etastar, sig2star), obs=ystar) 
    
def run_mcmc(y, L, map_per_unit, burn = 1000, nmcmc= 1000, thinning = 1, CoRe = True, sparsity = True, lam = 1e2,
             randseed = 0, bandwidth = .1, ratio_diff = .3, eps_diff = 100., C=10, thresh =1., sig_Gamma = 1., sig2star = .1):

    n, p = y.shape #L is the number of pieces in the splines
    key = random.PRNGKey(randseed)
    key, subkey1, subkey2, subkey3 = random.split(key, 4) 
  
    X = y  
    y = X#(X - X.mean(axis=0)) / np.sqrt(np.max(X)) 
    K = diffusion_dimension(y, eps = bandwidth, ratio = ratio_diff) 
    I_K = jnp.eye(K)  # K x K identity matrix
    W =  jnp.kron(jnp.ones((map_per_unit, 1)), I_K) 
    H = K * map_per_unit  
    ystar = diffusion_map(y, eps_diff, K, C)
    ystar = (ystar - ystar.mean(axis=0)) / ystar.std(0)
    normalized_ystar = (ystar - ystar.min(axis =0)) / (ystar.max(axis =0)-ystar.min(axis =0))*(1-1e-4)+1e-4
    dat = { 
        'K': K,
        'L': L,
        'y': jnp.array(y),
        'ystar': ystar,
        'W': W,
        'lam': lam,
        'H': H,
        'CoRe': CoRe,
        'sparsity': sparsity,
        'thresh': thresh,
        'sig_Gamma': sig_Gamma,
        'sig2star': sig2star
    } 
    #ax.profiler.start_trace("/tmp/jax_profiler")
    init_params = { 
        'u': (.99*normalized_ystar.T +.001) * jnp.ones((K,n)),    
        'slopes': jnp.ones((L, H)),
        'slopestar': jnp.ones((L, K)),
        'Lambda': random.normal(subkey1, shape = (p, K)),
        'sigma2': jnp.ones(1)*.1,
    }
    # Run MCMC with NumPyro
    nuts_kernel = NUTS(model, init_strategy=init_to_value(values=init_params))
    mcmc = MCMC(nuts_kernel, num_warmup=burn, num_samples=nmcmc, num_chains=1, thinning=thinning)
    rng_key = random.PRNGKey(randseed)
    mcmc.run(rng_key, **dat)
    mcmc.print_summary()

    # Extract the posterior samples
    samples = mcmc.get_samples()
    return(samples)