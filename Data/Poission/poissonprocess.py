import numpy as np
import pickle
import matplotlib.pyplot as plt

def poisson_process(x):
    nt = x.shape[1]
    atimes = np.zeros(nt, dtype=np.float64)
    lam = 10.
    atimes[1:] = .5*(1./lam + np.random.exponential(lam, nt-1))
    dt = np.min(atimes[1:])
    cnts_ext=[np.array(x[:, 0])]
    for jj in range(1,nt):
        cnt = 0
        while cnt*dt < atimes[jj]:
            cnts_ext = np.append(cnts_ext, [x[:, jj-1]], axis=0)
            cnt +=1
            print(cnt)
    return cnts_ext

file_name = '../twitter_sim/features_matrix_sample_080809.p'
features_matrix = (pickle.load(open(file_name, 'rb')))



# print(features_matrix_pre_process.shape)
# features_matrix = features_matrix_pre_process[1:161,:,:].reshape(4,4,10,14,300)
poisson_data = poisson_process(features_matrix)
#
with open('Poisson_Thij/poisson_data_twitter_080809.p', 'wb') as file_name:
     pickle.dump(poisson_data, file_name, protocol=pickle.HIGHEST_PROTOCOL)

data = pickle.load(open('Poisson_Thij/poisson_data_twitter_080809.p', 'rb'))
print(data.shape)

associated_strings = ['C3', 'C4', 'C5', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13']
colors = ['r','b','g','c','y','k', 'orange', 'm', 'salmon','sienna', 'gray','plum','orangered','navy']
linestyles = ['--',':','-','--',':','-','--',':','-','--',':','-','--',':']

fig = plt.figure(figsize=(15,8))

for k in range(14):
    plt.plot(data[:,k], alpha = 1, color = colors[k])
plt.legend(associated_strings)
plt.savefig('twitter_counts_080809.png')