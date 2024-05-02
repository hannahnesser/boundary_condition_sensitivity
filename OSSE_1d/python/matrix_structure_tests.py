
import numpy as np
np.set_printoptions(precision=4, linewidth=300, suppress=True)


# We will solve for the gain matrix for a range of state vector sizes
mn = 2

# Initialize the symbols we will use: tau is the lifetime (1/k),
# sa is the prior error variance (assumed constant), and 
# so is the observational error variance (also assumed constant).
# We also defne y = so/(sa*tau^2)
tau = 0.2#*30 # 0.2 lifetime * 30 ppb/day
so = 15**2*np.ones(mn)

# Initialize the Jacobian
k = tau*np.ones((mn, mn))
k = np.tril(k)

# define a 10 ppb BC perturbation
delta_c = 10*np.ones((2, 1))

# Calculate the gain matrix
def gain(kk, sasa, soso):
    sasa_inv = np.diag(1/sasa)
    soso_inv = np.diag(1/soso)
    gain = np.linalg.inv(sasa_inv + kk.T @ soso_inv @ kk) @ kk.T @ soso_inv
    return gain

# Case 1: Standard
sa_1 = 0.5**2*np.ones(mn)
k_1 = k*30 # multiply by 30 ppb/day
g_1 = gain(k_1, sa_1, so)
print('Standard')
print(g_1)
print(g_1 @ delta_c)
print('-'*70)

# Case 2: BC correction
sa_2 = np.append(sa_1, 0.01**2)
k_2 = np.concatenate([k_1, np.ones((mn, 1))*1900], axis=1)
g_2 = gain(k_2, sa_2, so)
print('BC correction')
print(g_2)
print(g_2 @ delta_c)
print('-'*70)

# Case 3: Buffer grid cell
sa_3 = np.array([5**2, 0.5**2])
k_3 = k_1.copy()
g_3 = gain(k_3, sa_3, so)
print('Buffer grid cell')
print(g_3)
print(g_3 @ delta_c)
print('-'*70)

# Case 4: Combine 2 and 3
sa_4 = np.array([5**2, 0.5**2, 0.01**2])
k_4 = k_2.copy()
g_4 = gain(k_4, sa_4, so)
print('BC correction + buffer grid cell')
print(g_4)
print(g_4 @ delta_c)
print('-'*70)

