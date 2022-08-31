"""
A demo that illustrates how to:
    + Construct the Discrete Karhunen-Loeve expansion of a random field.
    + Sample from it.

Author:
    Ilias Bilionis
   

Date:
    3/24/2014
    
Update:
    7/14/2022

"""

import os
import numpy as np
#import scipy.weave as weave        weave depreciated in python 3.xx -- Numba or Cython is used as the alternative
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import kle
#from numba import njit
#from setuptools import setup
#from Cython.Build import cythonize


#@njit(parallel=True)
def compute_covariance_matrix(X, s, ell):
    """
    Computes the covariance matrix at ``X``. This simply computes a
    squared exponential covariance and is here for illustration only.
    We will be using covariances from this package:
    `GPy <https://github.com/SheffieldML/GPy>`_.

    :param X:   The evaluation points. It has to be a 2D numpy array of
                dimensions ``num_points x input_dim``.
    :type X:    :class:`numpy.ndarray``
    :param s:   The signal strength of the field. It must be positive.
    :type s:    float
    :param ell: A list of lengthscales. One for each input dimension. The must
                all be positive.
    :type ell:  :class:`numpy.ndarray`
    """
    assert X.ndim == 2
    assert s > 0
    assert ell.ndim == 1
    assert X.shape[1] == ell.shape[0]
    C=np.zeros((X.shape[0], X.shape[0]))
    # We implement the function in C, otherwise it is very slow...
    #code = \
    """
    double dx;
    for(int i=0; i<NX[0]; i++)
    for(int j=0; j<NX[0]; j++)
        for(int k=0; k<NX[1]; k++) {
            dx = (X2(i, k) - X2(j, k)) / ELL1(k);
            C2(i, j) += dx * dx;
        }
    """
    #breakpoint()
    #weave.inline(code, ['X', 'ell', 'C'])
    for i in range(0,X.shape[0],1):
        for j in range(0,X.shape[0],1):
            for k in range(0,X.shape[1],1):
                dx=(X[i,k]-X[j,k])/ell[k]
                C[i,j]+=dx*dx
    return s ** 2 * np.exp(-0.5 * C)

def write_output_KLE(k=None,idx=None):
    # Create a folder where the permeability fields are to be exported
    folder_name='KLE_Samples'
    active_dir_path=os.path.dirname(__file__)
    folder_path=os.path.join(active_dir_path,folder_name)
    
    # Check if folder exists
    import shutil
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)
    
    for i in range(len(k)):
        file_path=os.path.join(folder_path,'PERM_KLE_SAMPLE_'+str(idx[i])+'.GRDECL')
        with open(file_path,'w') as open_file:
            # Flatten the array
            # a=np.reshape(k[i],(-1))
            open_file.write('PERMX'+'\n')
            for jk in range(k[i].shape[1]):
                a=k[i][jk,:]
                open_file.write((' '.join(['%10.12f ']*a.size)+'\n') % tuple(a))
            open_file.write('/')

if __name__ == '__main__':
    # Number of samples
    no_samples=100
    # Seed value
    rseed=50
    # Mean of the permeability field (assumed a log-normal distribution)
    mu_x=3 
    # Standard deviation of the permeability field                                
    sd_x=2   
    # Standard deviation of the log of variables (log of the variables is a normal distribution)                        
    sd_lnx=np.sqrt(np.log(((sd_x/mu_x)**2)+1)) 
    # Mean of the log of variable 
    mu_lnx=np.log(mu_x)-0.5*sd_lnx**2       
    
    corr_len_frac=0.2
    plot_group=True

    # Number of input points in x dimension
    n_x = 29
    # Number of inputs points in y dimension
    n_y = 29
    # Size of x dimension
    L_x = 2900.
    # Size of y dimension
    L_y = 2900.
    # Length scales
    ell = np.array([(corr_len_frac*L_x), (corr_len_frac*L_y)])
    # The signal strength of the field
    s = sd_lnx
    # The percentage of energy of the field you want to keep
    energy = 0.98
    # The points of evaluation of the random field
    x = np.linspace(0, L_x, n_x)
    y = np.linspace(0, L_y, n_y)
    XX, YY = np.meshgrid(x, y)
    X = np.hstack([XX.flatten()[:, None], YY.flatten()[:, None]])
    # Construct the covariance matrix
    C = compute_covariance_matrix(X, s, ell)
    # Compute the eigenvalues and eigenvectors of the field
    eig_values, eig_vectors = np.linalg.eigh(C)
    # Let's sort the eigenvalues and keep only the largest ones
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    # The energy of the field up to a particular eigenvalue:
    energy_up_to = np.cumsum(eig_values) / np.sum(eig_values)
    # The number of eigenvalues giving the desired energy
    i_max = np.arange(energy_up_to.shape[0])[energy_up_to >= energy][0]
    # Plot this
    print('Ploting energy of the field.')
    plt.figure()
    plt.plot(energy_up_to, 'b', linewidth=2)
    plt.plot(np.ones(energy_up_to.shape[0]), 'r', linewidth=2)
    plt.plot(np.hstack([np.arange(i_max), [i_max] * 50]),
             np.hstack([np.ones(i_max) * energy, np.linspace(0, energy_up_to[i_max], 50)[::-1]]),
             'g', linewidth=2)
    plt.ylim([0, 1.1])
    plt.title('Field Energy', fontsize=16)
    plt.xlabel('Eigenvalue Number', fontsize=16)
    plt.ylabel('Energy', fontsize=16)
    plt.legend(['Truncated expansion energy', 'Full Energy', '98% energy'],
               loc='best')
    print('Close figure to continue...')
    plt.show()
    # Now let's plot a few eigenvectors
    for i in range(3):
        plt.figure()
        c = plt.contourf(XX, YY, eig_vectors[:, i].reshape((n_x, n_y)))
        plt.colorbar(c)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.title('Eigenvector %d' % (i + 1), fontsize=16)
    print('Close all figures to continue...')
    plt.show()
    # Now, let's construct the D-KLE of the field and sample it.
    d_kle = kle.DiscreteKarhunenLoeveExpansion(X, eig_vectors[:,:i_max+1],
                                               eig_values[:i_max+1],seed=rseed)
    print('Some info about the expansion:')
    print(str(d_kle))
    
    k_train=[]
    k_sen=[]
    sen_idx=[]
    train_test_samp=10
    # Let's plot a few samples
    a=4; b=5; step=int(no_samples/(a*b))
    fig1=plt.figure('KLE_samples')
    fig2=plt.figure('KLE_historgram')
    fig1, ax1 = plt.subplots(a,b); fig2, ax2=plt.subplots(a,b)
    fig1.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.4, wspace=0.4)
    fig2.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.4, wspace=0.4)
    
    for i in range(no_samples):
        z=d_kle.sample().reshape((n_x, n_y))
        # Scale to the mean and standard deviation
        ln_x=z*1+mu_lnx
        kx=np.exp(ln_x)
        
        # Generate the random sample and rescale according to the mean and and standard deviation
        if not plot_group:
            plt.figure()
            c = plt.contourf(XX, YY, kx)
            plt.colorbar(c)
            plt.xlabel('x', fontsize=16)
            plt.ylabel('y', fontsize=16)
            plt.title('Sample %d' % i, fontsize=16)
        else:
            fig1.suptitle(f'Generated KLE Permeability Samples {no_samples} realizations',fontsize=8,y=1.02,weight='bold')
            if (i%step)==0:
                ij=int(i/step/b)
                ik=int(i/step%b)
                img1=ax1[ij,ik].contourf(XX, YY, kx)
                ax1[ij,ik].set_xlabel('x', fontsize=2)
                ax1[ij,ik].set_ylabel('y', fontsize=2)
                ax1[ij,ik].set_title('Sample %d' % i, fontsize=7,y=0.95)
                ax1[ij,ik].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                ax1[ij,ik].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                
                #Histogram
                img2=ax2[ij,ik].hist(kx,facecolor='g')
                ax2[ij,ik].set_xlabel('k', fontsize=5)
                ax2[ij,ik].set_ylabel('Frequency', fontsize=5)
                ax2[ij,ik].set_title('Hist_Sample %d' % i, fontsize=7,y=0.95)
                ax2[ij,ik].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                ax2[ij,ik].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
            
        # Append the sampled field to the list
        if (i%train_test_samp)!=0:
            k_train.append(kx)
        else:
            k_sen.append(kx)
            sen_idx.append(i)
    if plot_group:
        cbar=fig1.colorbar(img1,ax=ax1.flat,aspect=36)
        #ticks=np.round(np.linspace(-1.2,8.0,10),2)
        #cbar.set_ticks(ticks)
    plt.show()
    
    write_output_KLE(k=k_sen,idx=sen_idx)   


def KLE_samples(no_samples=100,rseed=50,mu=2.5,sd=1.0,grid_dim=(29,29,1),gridblock_size=(100,100,80),corr_len_fac=0.2,energy_fac=0.98,train_test_samp=10,write_out=True):            
    # Standard deviation of the log of variables (log of the variables is a normal distribution)                        
    sd_lnx=np.sqrt(np.log(((sd/mu)**2)+1)) 
    # Mean of the log of variable 
    mu_lnx=np.log(mu)-0.5*sd_lnx**2       
    
    # Number of input points in x dimension
    n_x = grid_dim[0]
    # Number of inputs points in y dimension
    n_y = grid_dim[1]
    # Size of x dimension
    L_x = gridblock_size[0]*n_x
    # Size of y dimension
    L_y = gridblock_size[1]*n_y
    # Length scales
    ell = np.array([(corr_len_fac*L_x), (corr_len_fac*L_y)])
    # The signal strength of the field
    s = sd_lnx
    # The percentage of energy of the field you want to keep
    energy = energy_fac
    # The points of evaluation of the random field
    x = np.linspace(0, L_x, n_x)
    y = np.linspace(0, L_y, n_y)
    XX, YY = np.meshgrid(x, y)
    X = np.hstack([XX.flatten()[:, None], YY.flatten()[:, None]])
    # Construct the covariance matrix
    C = compute_covariance_matrix(X, s, ell)
    # Compute the eigenvalues and eigenvectors of the field
    eig_values, eig_vectors = np.linalg.eigh(C)
    # Let's sort the eigenvalues and keep only the largest ones
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    # The energy of the field up to a particular eigenvalue:
    energy_up_to = np.cumsum(eig_values) / np.sum(eig_values)
    # The number of eigenvalues giving the desired energy
    i_max = np.arange(energy_up_to.shape[0])[energy_up_to >= energy][0]
    
    # Now, let's construct the D-KLE of the field and sample it.
    d_kle = kle.DiscreteKarhunenLoeveExpansion(X, eig_vectors[:,:i_max+1],
                                               eig_values[:i_max+1],seed=rseed)
    
    k_train=[]
    k_sen=[]
    sen_idx=[]
    for i in range(no_samples):
        z=d_kle.sample().reshape((n_x, n_y))
        # Scale to the mean and standard deviation
        ln_x=z*1+mu_lnx
        kx=np.exp(ln_x)
        
        # Append the sampled field to the list
        if (i%train_test_samp)!=0:
            k_train.append(kx)
        else:
            k_sen.append(kx)
            sen_idx.append(i)
    
    if write_out:
        write_output_KLE(k=k_sen,idx=sen_idx)
  
    return k_train,k_sen