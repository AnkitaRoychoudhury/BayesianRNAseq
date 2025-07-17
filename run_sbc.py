import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

def compute_rank(flat_samples, true_value):
    """
    Compute SBC rank: number of posterior samples < true value.

    Args:
        flat_samples: flattened array of parameter values from posterior
        true_val: true value of parameter from the data simulation

    Returns:
        rank of parameter value
        
    """
    #flat_samples = trace.posterior[param].values.reshape(-1)
    rank = np.sum(flat_samples < true_value)
    return rank

def model(s_spike, G, s_seq_obs, x_seq_obs, mu_b_sigma=1, sigma_b_sigma=0.5, 
          X_mrna_mu=np.log(200000), X_mrna_sigma=1):

    """
    Defines the model for numpyro. With spike-ins and mRNA samples.

    Args:
        s_spike: numpy array of spike-in inputs
        G: (int) number of genes
        s_seq_obs: observed value of spike-ins
        x_seq_obs: observed value of mRNA counts
        mu_b_sigma: sigma value for mu_b 
        sigma_b_sigma: sigma value for sigma_b
        X_mrna_mu: mu value for X_mrna
        X_mrna_sigma: sigma value for X_mrna
    
    """
    K = len(s_spike)
    
    mu_b = numpyro.sample("mu_b", dist.Normal(0, mu_b_sigma))
    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(sigma_b_sigma))
    
    b_s = numpyro.sample("b_s", dist.LogNormal(mu_b, sigma_b).expand([K]))
    b_x = numpyro.sample("b_x", dist.LogNormal(mu_b, sigma_b).expand([G]))

    numpyro.sample("s_seq_obs", dist.Poisson(s_spike * b_s), obs=s_seq_obs)

    X_mrna = numpyro.sample("X_mrna", dist.LogNormal(X_mrna_mu, X_mrna_sigma))
    alpha = numpyro.sample("alpha", dist.Dirichlet(np.ones(G)))
    x_mrna = X_mrna * alpha

    numpyro.sample("x_seq_obs", dist.Poisson(b_x * x_mrna), obs=x_seq_obs)



def run_SBC(iters, G=5, s_spike=np.array([2, 20, 200, 2000, 20000, 200000]),
            mu_b_sigma=1, sigma_b_sigma=0.5, 
            X_mrna_mu=np.log(200000), X_mrna_sigma=1, display_outputs = True):

    """
    Function to run simulation-based calibration and return key values for plotting

    Args:
        iters: (int) number of SBC runs
        G: (int) number of genes
        s_spike: numpy array of spike-in inputs
        mu_b_sigma: sigma value for mu_b 
        sigma_b_sigma: sigma value for sigma_b
        X_mrna_mu: mu value for X_mrna
        X_mrna_sigma: sigma value for X_mrna

    Returns:
        true_param_dict: (dict) true values for parameters from simulation
        trace_param_dict: (dict) mean values of parameters from posterior
        rank_param_dict: (dict) ranks of parameters
        sd_param_dict: (dict) standard deviation of parameters from posterior
        
    """
    
    
    K = len(s_spike)

    # store true values
    mu_b_true_all, sigma_b_true_all, b_s_true_all, b_x_true_all = [], [], [], []
    X_mrna_true_all, alpha_true_all, x_mrna_true_all = [], [], []
    s_seq_true_all, x_seq_true_all = [], []

    # store posterior means
    mu_b_all, sigma_b_all, b_s_all, b_x_all = [], [], [], []
    X_mrna_all, alpha_all, x_mrna_all = [], [], []
    s_seq_all, x_seq_all = [], []

    # store ranks
    mu_b_rank, sigma_b_rank, X_mrna_rank = [], [], []
    b_s_rank = [[] for _ in range(K)]
    b_x_rank = [[] for _ in range(G)]
    alpha_rank = [[] for _ in range(G)]
    x_mrna_rank = [[] for _ in range(G)]
    s_seq_rank = [[] for _ in range(K)]
    x_seq_rank = [[] for _ in range(G)]

    # posterior sds
    mu_b_sds, sigma_b_sds, X_mrna_sds = [], [], []
    b_s_sds, b_x_sds, alpha_sds, x_mrna_sds = [], [], [], []
    s_seq_sds, x_seq_sds = [], []

    for i in tqdm(range(iters)):
        # sample true data
        mu_b_true = np.random.normal(0, mu_b_sigma)
        sigma_b_true = np.abs(np.random.normal(0, sigma_b_sigma))
        b_s_true = np.random.lognormal(mean=mu_b_true, sigma=sigma_b_true, size=K)
        b_x_true = np.random.lognormal(mean=mu_b_true, sigma=sigma_b_true, size=G)
        X_mrna_true = np.random.lognormal(mean=X_mrna_mu, sigma=X_mrna_sigma)
        alpha_true = np.random.dirichlet(alpha=np.ones(G))
        x_mrna_true = X_mrna_true * alpha_true
        s_seq_true = np.random.poisson(lam=b_s_true * s_spike, size=K)
        x_seq_true = np.random.poisson(lam=b_x_true * x_mrna_true, size=G)

        # record truths
        mu_b_true_all.append(mu_b_true)
        sigma_b_true_all.append(sigma_b_true)
        b_s_true_all.append(b_s_true)
        b_x_true_all.append(b_x_true)
        X_mrna_true_all.append(X_mrna_true)
        alpha_true_all.append(alpha_true)
        x_mrna_true_all.append(x_mrna_true)
        s_seq_true_all.append(s_seq_true)
        x_seq_true_all.append(x_seq_true)

        # run mcmc
        kernel = NUTS(model, init_strategy=init_to_median())
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, progress_bar=display_outputs)
        mcmc.run(jax.random.PRNGKey(i), s_spike=s_spike, G = G, s_seq_obs=s_seq_true, x_seq_obs=x_seq_true,
                 mu_b_sigma=mu_b_sigma, sigma_b_sigma=sigma_b_sigma,
                 X_mrna_mu=X_mrna_mu, X_mrna_sigma=X_mrna_sigma)
        samples = mcmc.get_samples()

        # posterior means and stds
        mu_b_all.append(np.mean(samples["mu_b"]))
        sigma_b_all.append(np.mean(samples["sigma_b"]))
        b_s_all.append(np.mean(samples["b_s"], axis=0))
        b_x_all.append(np.mean(samples["b_x"], axis=0))
        X_mrna_all.append(np.mean(samples["X_mrna"]))
        alpha_all.append(np.mean(samples["alpha"], axis=0))
        x_mrna_all.append(np.mean(samples["X_mrna"][:, None] * samples["alpha"], axis=0))

        mu_b_sds.append(np.std(samples["mu_b"]))
        sigma_b_sds.append(np.std(samples["sigma_b"]))
        b_s_sds.append(np.std(samples["b_s"], axis=0))
        b_x_sds.append(np.std(samples["b_x"], axis=0))
        X_mrna_sds.append(np.std(samples["X_mrna"]))
        alpha_sds.append(np.std(samples["alpha"], axis=0))
        x_mrna_sds.append(np.std(samples["X_mrna"][:, None] * samples["alpha"], axis=0))

        # posterior predictive
        predictive = Predictive(model, posterior_samples=samples)
        ppc_samples = predictive(jax.random.PRNGKey(10000 + i), s_spike=s_spike, G = G, s_seq_obs=None, x_seq_obs=None,
                                 mu_b_sigma=mu_b_sigma, sigma_b_sigma=sigma_b_sigma,
                                 X_mrna_mu=X_mrna_mu, X_mrna_sigma=X_mrna_sigma)
        s_seq_pp = ppc_samples["s_seq_obs"]
        x_seq_pp = ppc_samples["x_seq_obs"]

        s_seq_all.append(np.mean(s_seq_pp, axis=0))
        x_seq_all.append(np.mean(x_seq_pp, axis=0))

        s_seq_sds.append(np.std(s_seq_pp, axis=0))
        x_seq_sds.append(np.std(x_seq_pp, axis=0))

        # compute ranks
        mu_b_rank.append(compute_rank(samples["mu_b"].reshape(-1), mu_b_true))
        sigma_b_rank.append(compute_rank(samples["sigma_b"].reshape(-1), sigma_b_true))
        X_mrna_rank.append(compute_rank(samples["X_mrna"].reshape(-1), X_mrna_true))

        for k in range(K):
            b_s_rank[k].append(compute_rank(samples["b_s"][:, k], b_s_true[k]))
            s_seq_rank[k].append(compute_rank(s_seq_pp[:, k], s_seq_true[k]))

        for g in range(G):
            b_x_rank[g].append(compute_rank(samples["b_x"][:, g], b_x_true[g]))
            alpha_rank[g].append(compute_rank(samples["alpha"][:, g], alpha_true[g]))
            x_mrna_rank[g].append(compute_rank((samples["X_mrna"][:, None] * samples["alpha"])[:, g], x_mrna_true[g]))
            x_seq_rank[g].append(compute_rank(x_seq_pp[:, g], x_seq_true[g]))

    # pack into dictionaries
    true_param_dict = {'mu_b': mu_b_true_all, 'sigma_b': sigma_b_true_all, 'b_s': b_s_true_all, 
                       'b_x': b_x_true_all, 'X_mrna': X_mrna_true_all, 'alpha': alpha_true_all, 
                       'x_mrna': x_mrna_true_all, 's_seq': s_seq_true_all, 'x_seq': x_seq_true_all}
    
    trace_param_dict = {'mu_b': mu_b_all, 'sigma_b': sigma_b_all, 'b_s': b_s_all, 'b_x': b_x_all,
                        'X_mrna': X_mrna_all, 'alpha': alpha_all, 'x_mrna': x_mrna_all,
                        's_seq': s_seq_all, 'x_seq': x_seq_all}
    rank_param_dict = {'mu_b': mu_b_rank, 'sigma_b': sigma_b_rank, 'b_s': b_s_rank, 'b_x': b_x_rank,
                       'X_mrna': X_mrna_rank, 'alpha': alpha_rank, 'x_mrna': x_mrna_rank}
                       #'s_seq': s_seq_rank, 'x_seq': x_seq_rank}
    sd_param_dict = {'mu_b': mu_b_sds, 'sigma_b': sigma_b_sds, 'b_s': b_s_sds, 'b_x': b_x_sds,
                     'X_mrna': X_mrna_sds, 'alpha': alpha_sds, 'x_mrna': x_mrna_sds,
                     's_seq': s_seq_sds, 'x_seq': x_seq_sds}
    
    return true_param_dict, trace_param_dict, rank_param_dict, sd_param_dict
