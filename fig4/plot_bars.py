import numpy as np
import matplotlib.pyplot as plt
import matplotlib

optimal = np.loadtxt("../../bars/logs/cresults.txt", delimiter='\t')
deneve = np.loadtxt("../../bars_somatic/logs/cresults.txt", delimiter='\t')
n_boot = 10000
cil = 0.95
corrs = np.arange(0.0,1.05,0.05)

##Data
# Get data optimal
loss_optimal = []
corr_optimal = []
for corr in corrs:
    def criterion(x): 
        return np.abs(corr - x) < 0.02
    inds = np.vectorize(criterion)(optimal[:,0])
    if sum(inds) > 0:
        temp = optimal[inds, 2]
        loss_optimal.append(temp)
        corr_optimal.append(corr)

# Get data deneve
loss_deneve = []
corr_deneve = []
for corr in corrs:
    def criterion(x): 
        return np.abs(corr - x) < 0.01
    inds = np.vectorize(criterion)(deneve[:,0])
    if sum(inds) > 0:
        temp = deneve[inds, 2]
        loss_deneve.append(temp)
        corr_deneve.append(corr)



def conf(x):
    mn = np.median(x)
    samples = np.random.choice(x, size=(len(x), n_boot), replace=True)
    medians = np.sort(np.apply_along_axis(np.median, 0, samples))
    ind1 = int((1 - cil)/2*n_boot)
    ind2 = int((1 - (1 - cil)/2)*n_boot)
    return (mn - medians[ind1], medians[ind2] - mn)

##Plots
## The naming makes a difference for the order and color!
# Plots loss
mn = np.apply_along_axis(np.median, 1, loss_optimal)
sd = np.apply_along_axis(conf, 1, loss_optimal)
sx1 = "DB"
fig, ax = plt.subplots(figsize=(0.5*5,0.5*3))
print(sd)
p1 = plt.errorbar(corr_optimal, mn, yerr = sd.T, label=sx1, capsize=2, elinewidth=0.7, color='darkolivegreen')
plt.xlabel("correlation (p)")
plt.ylabel("decoder loss")

mn = np.apply_along_axis(np.median, 1, loss_deneve)
sd = np.apply_along_axis(conf, 1, loss_deneve)
sx2 = "SB"
print(sd)
p2 = plt.errorbar(corr_deneve, mn, yerr = sd.T, label=sx2, capsize=2, elinewidth=0.7, color='orange')

# get handles
handles, labels = ax.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
# use them in the legend
#from matplotlib.font_manager import FontProperties
#fontP = FontProperties()
#fontP.set_size('small') 
#plt.legend(reversed(handles), [sx2,sx1], loc='upper left',numpoints=1)

"""
plt.yscale("log")
ax.set_yticks([0.0075,0.01,0.0125],minor=False)
ax.set_yticks([],minor=True)
y_formatter = matplotlib.ticker.ScalarFormatter()
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_minor_formatter(y_formatter)
plt.ticklabel_format(axis='y',style='plain')"""

plt.tight_layout()
plt.savefig("../../plots/bars_comparison.svg")

