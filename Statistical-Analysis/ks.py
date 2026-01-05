import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gamma, kstest

# Load transition time data
time_list = np.loadtxt("transition-time.dat")
min_time = min(time_list)
max_time = max(time_list)
n_bins = 20

# Function for fitting an exponential CDF
def func(x, a):
    return 1 - np.exp(-a * x)

# ECDF function
def get_ecdf(time_list, min_time, max_time, n_bins):
    time_domain = np.logspace(np.log10(min_time), np.log10(max_time), n_bins)
    N = len(time_list)
    a, b = np.histogram(time_list, time_domain)
    ECDF = np.cumsum(a) / N
    if len(time_domain[:-1]) == len(ECDF):
        return time_domain[:-1], ECDF
    else:
        return "error"

# Compute CDFs and fit parameters
def compute_cdfs(time_list):
    min_t = np.min(time_list) / 10
    max_t = np.max(time_list) * 10
    nbins = 500
    x, y = get_ecdf(time_list, min_t, max_t, nbins)
    test_mean = np.mean(time_list)
    test_sigma = np.std(time_list, ddof=1)
    test_m = np.median(time_list)
    
    # Fit the gamma distribution
    guess = 1.0 / test_mean
    pars, cov = curve_fit(f=func, xdata=x, ydata=y, p0=[guess], maxfev=1000000)
    stdevs = np.sqrt(np.diag(cov))
    
    tau = np.power(pars[0], -1)
    tcdf = gamma.cdf(x, 1, scale=tau)
    
    cdfs = [x, y, tcdf]
    data = {
        "tau": tau,
        "mean": test_mean,
        "median": test_m,
        "std_dev": test_sigma,
        "tau_std_dev": stdevs[0]
    }
    
    return cdfs, data

# Perform KS test
def perform_ks_test(time_list, tau):
    D, p_value = kstest(time_list, 'gamma', args=(1, 0, tau))
    return D, p_value

# Main function call
cdfs, data = compute_cdfs(time_list)
tau = data["tau"]
D, p_value = perform_ks_test(time_list, tau)

# Print KS test result
print(f"Kolmogorov-Smirnov test D-statistic: {D}")
print(f"Kolmogorov-Smirnov test p-value: {p_value}")

# Plot the results
plt.plot(cdfs[0], cdfs[1], label='Empirical CDF', color='blue')
plt.plot(cdfs[0], cdfs[2], label='Fitted Gamma CDF', color='red', linestyle='--')
plt.xscale('log')
plt.xlabel('Transition Time')
plt.ylabel('CDF')
plt.title('Alanine dipeptide in vacuum')
plt.legend()
plt.grid(True)
plt.savefig('cdf_plot.png')
plt.show()

