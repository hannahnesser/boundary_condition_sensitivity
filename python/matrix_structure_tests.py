from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy import symbols, pprint
from sympy import *

#%%
# We will solve for the gain matrix for a range of state vector sizes
mn = 1

# Initialize the symbols we will use: tau is the lifetime (1/k),
# sa is the prior error variance (assumed constant), and 
# so is the observational error variance (also assumed constant).
# We also defne y = so/(sa*tau^2)
# tau, sa, sbc, p, so, y, xa, xbc, deltay = symbols('t sa sbc p so y xa xbc c')
U, D, L, xd, x, saup, sai, sa, soup, soi, so, c, n, kk, tau, p = symbols('U Lup L xup xa saup sai a soup soi o c n k t p')

# Initialize the Jacobian
k = kk*ones(mn + 1)
k = k.lower_triangular()
# # k = tau*xa*k
# k[0, 0] *= n #*xd
# k[1, 0] *= n #*xd
# k[1, 1] = kk*L/U #*x
k = k[1, :]
k[1] = 1

# for i in range(mn):
#     k[i, -1] = xbc
# k = k[:(mn - 1), :]
# # # print(k)
# # # k = 0.2*k
k
print('-'*70)

# Initialize the error matrices
so_mat = so**2*eye(mn)
# so_mat[0, 0] *= 1/n
# so_mat[1, 1] = soi**2 
sa_mat = sa**2*eye(mn + 1) # Assume sa is absolute
# sa_mat[1, 1] = sai**2
# so[0, 0] *= L/D
# so = 100*eye(mn)
# sa = 0.25*eye(mn)
# sa_mat[0, 0] *= n**2
# sa[1, 1] *= x**2
sa_mat[1, 1] = c**2
# sa[0, 0] = sbc**2
sa_mat
print('-'*70)
so_mat
print('-'*70)

# and initialize matrix structures without the proper scaling
# L = ones(mn)
# L = L.lower_triangular()
# I = eye(mn)

# x = ones(mn - 1, 1)
deltac = c*ones(mn, 1)
# deltac[0] = L1*c/L0
# deltac[1] = L2*c/L0

deltac
print('-'*70)

# Calculate the product
# g = ((1/tau)*(L.T*L + y**2*I).inv()*L.T)*x
shat = (sa_mat.inv() + k.T @ so_mat.inv() @ k).inv()
shat_det = (sa_mat.inv() + k.T @ so_mat.inv() @ k).det()
# print(shat_det)
# m4 s2/kg2   + ppb2 m4 s2/kg2 / ppb2 ==> kg2/m4/s2
g_deltac = (shat @ k.T @ so_mat.inv()) @ deltac
# (kg2/m4/s2) (ppb m2 s/kg) (/ppb2) (ppb) ==> kg/m2/s
# pprint(simplify((shat @ k.T @ so.inv())))
# pprint(simplify(g_deltac*shat_det/c/kk*so**(2*mn)*sa**(2*mn - 2)*x**(2*mn - 2)))
# pprint(simplify(sa_mat.inv() + k.T @ so_mat.inv() @ k))
print('-'*70)
opt2 = simplify(g_deltac[0])

#%%
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy import symbols, pprint
from sympy import *
import numpy as np
#%%
n = 4
# tri = np.tril(np.repeat(np.arange(1, n + 1)[::-1][:, None], n, axis=1))
tri = np.tril(np.ones((n, n)))
base = tri.T @ tri
# shat = tri + tri.T
# shat[np.arange(n), np.arange(n)] = shat[np.arange(n), np.arange(n)]/2
r, sa, so, k = symbols('R a o k')
shat = base + r*eye(n)
shat_det = shat.det()
shat_inv = shat.inv()
kk = ones(n).upper_triangular()
g = shat_inv @ kk # the whole thing should be multiplied by R k sa^2 / so^2
a = g @ kk # The whole thing is now multiplied by 1 (The Rs cancel out)
simplify(simplify(Trace(a*shat_det)))
# deltac = simplify(g @ ones(n, 1))
# # print(n)
# simplify(deltac*shat_det)
# shat, simplify(shat_inv*shat_det)
#%%