import matplotlib.pyplot as plt
import numpy as np
import time

time_string = time.strftime("%Y%m%d-%h%m%s")

def eigen_plot(eigenvectors, eigenvalues, phi_modes, show=False):
    fig = plt.figure(figsize=(6,10))
    plt.subplot(311)
    plt.scatter(eigenvalues.real, eigenvalues.imag)
    plt.title('Eigenvalues')

    plt.subplot(312)
    plt.pcolor(np.array(np.log10(np.abs(eigenvectors))))
    plt.title('Estimated Koopman Modes')

    plt.subplot(313)
    plt.pcolor(np.array(np.log10(np.abs(phi_modes))))
    plt.title('Estimated Phi Modes')

    if show:
        plt.show()

    plt.savefig('DMD_results/'+'eigen'+time_string+'.png')
    return None

def data_plot(data, show=False):
    associated_strings = ['C3', 'C4', 'C5', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13']
    rows, columns = data.shape
    if rows > columns:
        print('Is the matrix properly transposed? Ensure time on x-axis')
        pass

    fig = plt.figure(figsize=(10,5))
    for row in range(rows):
        plt.plot(data[row,:])
    plt.legend(associated_strings)

    if show:
        plt.show()

    plt.savefig('DMD_results/'+'data'+time_string+'.png')
