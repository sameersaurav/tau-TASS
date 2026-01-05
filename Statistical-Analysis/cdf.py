import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
import os
from scipy.optimize import curve_fit
from scipy.stats import gamma
from scipy import stats
import matplotlib.font_manager as fm
from scipy.stats import gamma, kstest
from matplotlib import rc
# Enable LaTeX text rendering
#rc('text', usetex=True)
#rc('font', family='serif')

time_list = np.loadtxt("transition-time.dat")
min_time=min(time_list)
#print("min_time",min_time)

max_time=max(time_list)
#print("max_time",max_time)
n_bins=20

def func(x, a):
    return 1 - np.exp(-a * x)

def get_ecdf(time_list, min_time, max_time, n_bins):
    time_domain = np.logspace(np.log10(min_time), np.log10(max_time), n_bins)
    N = len(time_list)
    a, b = np.histogram(time_list, time_domain)
    ECDF = np.cumsum(a) / N
    if len(time_domain[:-1]) == len(ECDF):
         return time_domain[:-1], ECDF
    else:
        return "error"
x, y = get_ecdf(time_list, min_time, max_time, n_bins)

def compute_cdfs(time_list):
    min_t = np.min(time_list)/10
    max_t = np.max(time_list)*10
    nbins = 500
    x, y = get_ecdf(time_list, min_t, max_t, nbins)
    test_mean = np.mean(time_list)
    test_sigma = np.std(time_list, ddof=1)
    test_m = np.median(time_list)
    means = test_mean
    medians = test_m
    uncertainty = test_sigma / np.sqrt(len(time_list))
    stdvs = test_sigma
    mean_sigma_ratio = test_mean / test_sigma
    log2_median_ratio = np.log(2) * test_mean / test_m
    guess = 1.0 / test_mean
    # print(np.mean(time_list),min_t,max_t)
    pars, cov = curve_fit(f=func, xdata=x, ydata=y, p0=[guess], maxfev=1000000)
    stdevs = np.sqrt(np.diag(cov))
    f_stdvs = stdevs[0]
    tau = np.power(pars[0], -1)
    print("tau",tau)
    tau_mean_ratio = (tau / test_mean)
    tcdf = gamma.cdf(x, 1, scale=tau)
    cdfs = [x, y, tcdf]
    data = [tau, stdvs, f_stdvs, means, medians,
            uncertainty, mean_sigma_ratio, log2_median_ratio, tau_mean_ratio]
    return cdfs, data
cdfs, data = compute_cdfs(time_list)
#D, p_value = kstest(time_list, 'gamma', args=(1,0,tau))

from matplotlib.font_manager import FontProperties

# Create a FontProperties object with desired settings
font_prop = FontProperties(family='serif', size=24)

#plt.plot(cdfs[0], cdfs[1], label='ECDF')
#plt.plot(cdfs[0], cdfs[2], label='TCDF')
plt.plot(cdfs[0], cdfs[2], linewidth=4)
plt.xscale('log')
plt.xlabel('Transition Time',fontsize=24)
plt.ylabel('Transition Probability',fontsize=24)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
#plt.legend(prop=font_prop)
plt.grid(True)
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                left=False, right=False, labelbottom=False, labelleft=False)

plt.savefig('schematic_cdf.png',dpi=600)
plt.show()

