import numpy as np
import matplotlib.pyplot as plt
from DynamicModeDecomposition.DMD import standard_dynamic_mode_decomposition, resized_data, mode_err_cmp, recon_err_cmp
from DynamicModeDecomposition.plotter import eigen_plot, data_plot
import pickle

def dmd_corr_comp(phim, kmodes, shft):
    nt, nmds = np.shape(phim)
    corr_mat = np.zeros((nmds, nmds), dtype=np.complex128)

    kmnrms = np.linalg.norm(kmodes, ord=2, axis=0)
    kmnrlzd = kmodes @ np.diag(1./kmnrms)
    tmat = phim - np.tile(np.mean(phim, 0), (nt, 1))
    vrncs = np.sqrt(np.mean(tmat*np.conj(tmat), 0))
    phimnrmlzd = tmat @ np.diag(1./vrncs)
    for jj in range(nmds):
        for kk in range(nmds):
            kval = np.sum(kmnrlzd[:, jj]*np.conj(kmnrlzd[:, kk]))
            shftvec = np.conj(np.roll(phimnrmlzd[:, kk], shft))

            pval = np.mean(phimnrmlzd[shft:, jj]*shftvec[shft:])
            corr_mat[jj, kk] = kval * pval

    return corr_mat

print('Beginning')
data = pickle.load(open('../Data/Poission/Poisson_Thij/poisson_smooth_data_twitter_020809.p', 'rb'))

print('Data is loaded')
data = data.T
print(data.shape)
scaled_data, std_dev, mean = resized_data(data)

print(scaled_data.shape)

print('Data is now normalized')

evcs, evls, phim, average = standard_dynamic_mode_decomposition(scaled_data, 4)
print('Finished DMD on scaled data')

eigen_plot(evcs, evls, phim.T, file_name='../PostThesisWork/ModeCorr/Thij0208/scaled_data_eigs_thij0208', show=False)
data_plot(scaled_data, name='../PostThesisWork/ModeCorr/Thij0208/scaled_data_thij0208', show=False, save=True)

mat = dmd_corr_comp(phim, evcs, 3)
fig1 = plt.figure(figsize=(7,5))
corr = plt.pcolor(np.abs(mat), cmap='jet')
fig1.colorbar(corr)
plt.savefig('../PostThesisWork/ModeCorr/Thij0208/scaled_data_corr_thij0208.png')

recon_error = recon_err_cmp(scaled_data, evls, evcs, phim)
mode_error = mode_err_cmp(evls, phim)
plt.close()
plt.close()
plt.close()

with open('ModeCorr/Thij0208/scaled_thij0208_dmd_mode_error.p', 'wb') as fp:
    pickle.dump(mode_error, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('ModeCorr/Thij0208/scaled_thij0208_rec_error.p', 'wb') as fp:
    pickle.dump(recon_error, fp, protocol=pickle.HIGHEST_PROTOCOL)

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
fig, ax = plt.subplots(1, figsize=(12,8), constrained_layout=True)
yy = np.log10(mode_error)
print(np.max(yy))
ax.text(x=6 + 0.4, y=-3.5, s = str('The associated one-step reconstruction error is %1.5f' %recon_error), fontsize=14)
ax.scatter(range(14), yy, color='c', s=400)
ax.set_title('Mode Errors and Recon Error', fontsize=20)
ax.set_xlabel('Modes', fontsize=15)
ax.set_ylabel('Mode Errors', fontsize=15)
plt.savefig('../PostThesisWork/ModeCorr/Thij0208/errors.png')
