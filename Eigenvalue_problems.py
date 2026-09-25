import numpy as np
import matplotlib.pyplot as plt
from Matrix import householder, QR_decomposition, QR_linear_system

#power method
def power_method (mat, epsilon, n_iteration) :
    mat = mat.copy()
    N = len(mat)
    y = np.random.rand(N)
    lamb = 0
    lamb_prime = 1
    i = 0
    lamb_list = []

    while(i < n_iteration and abs(lamb-lamb_prime) > epsilon):
        y = mat @ y
        y /= np.sqrt(np.vdot(y, y))
        lamb_prime = lamb
        lamb = np.vdot(y, mat @ y) / np.vdot(y, y)
        lamb_list.append(lamb) 
        i += 1
    
    lamb_list = np.array(lamb_list)

    return lamb, y, lamb_list

#inverse power method
def inverse_power_method (mat, epsilon, n_iteraction) :
    mat = mat.copy()
    N = len(mat)
    y = np.random.rand(N)
    lamb = 0
    lamb_prime = 1
    i = 0
    lamb_list = []

    while(i < n_iteraction and abs(lamb-lamb_prime) > epsilon):
        y = QR_linear_system(mat, y)
        y /= np.sqrt(np.vdot(y, y))
        lamb_prime = lamb
        lamb = np.vdot(y, mat @ y) / np.vdot(y, y)
        lamb_list.append(lamb) 
        i += 1

    lamb_list = np.array(lamb_list)

    return lamb, y, lamb_list

#shifted power method
def shifted_power_method (mat, alpha, epsilon, n_iteration) :
    N = len(mat)
    y = np.random.rand(N)
    y /= np.sqrt(np.vdot(y, y))
    lamb = y.conj() @ mat @ y
    lamb_prime = 1
    i = 0
    lamb_list = []

    while(i < n_iteration and abs(lamb-lamb_prime) > epsilon):
        y = QR_linear_system(mat - alpha * np.identity(N), y)
        y /= np.sqrt(np.vdot(y, y))
        lamb_prime = lamb
        lamb = np.vdot(y, mat @ y) / np.vdot(y, y)
        lamb_list.append(lamb)
        i += 1

    lamb_list = np.array(lamb_list)

    return lamb, y, lamb_list

#power method con deflazione
def power_method_deflation (mat, epsilon, n_iteration) :
    N = len(mat)
    mat = mat.copy()

    eigenvalues = np.zeros(N, dtype=complex)
    eigenvectors = np.zeros((N, N), dtype=complex)

    for i in range (N) :
        lamb, y, lamb_list = power_method(mat, epsilon, n_iteration)
        eigenvalues [i] = lamb
        eigenvectors [:, i] = y
        mat = mat - lamb * np.outer(y, y.conj())
        
    return eigenvalues, eigenvectors


def QR_eigenvalues (mat, epsilon, n_iteration) :
    mat = mat.copy()
    dtype = np.result_type(mat)
    N = len(mat)

    Qk = np.identity(N, dtype=dtype)
    eigenvalues = np.zeros(N, dtype=dtype)
    eigenvalues_list = []

    for i in range (n_iteration) :

        Q, R = QR_decomposition(mat)
        mat = R @ Q 
        Qk = Qk @ Q
        
        lower_mat = np.tril(mat, k=-1)

        eigenvalues_list.append(np.diagonal(mat))

        if (np.max(np.abs(lower_mat)) < epsilon) :
            '''print("converged after", i, "iterations")'''
            break
                

    eigenvalues_list = np.array(eigenvalues_list) 
    eigenvalues = np.diagonal(mat)
    
    return eigenvalues, eigenvalues_list, Qk
         