from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy import symbols, pprint
from sympy import *

# We will solve for the gain matrix for a range of state vector sizes
mn = 2

# Initialize the symbols we will use: tau is the lifetime (1/k),
# sa is the prior error variance (assumed constant), and 
# so is the observational error variance (also assumed constant).
# We also defne y = so/(sa*tau^2)
# tau, sa, sbc, p, so, y, xa, xbc, deltay = symbols('t sa sbc p so y xa xbc c')
U, D, L, xd, x, sa, so, c, n = symbols('U D L xd x sa so c n')

# Initialize the Jacobian
k = ones(mn)
k = k.lower_triangular()
# k = tau*xa*k
k[0, 0] = D*xd/U
k[1, 0] = D*xd/U
k[1, 1] = L*x/U
k = k[1, :]

# for i in range(mn):
#     k[i, -1] = xbc
# k = k[:(mn - 1), :]
# # # print(k)
# # # k = 0.2*k
pprint(k)
print('-'*70)

# Initialize the error matrices
so = so**2*eye(1)
sa = sa**2*eye(mn)
# so[0, 0] *= L/D
# so = 100*eye(mn)
# sa = 0.25*eye(mn)
# sa[0, 0] *= p
# sa[0, 0] = sbc**2
pprint(sa)
print('-'*70)
pprint(so)
print('-'*70)

# and initialize matrix structures without the proper scaling
# L = ones(mn)
# L = L.lower_triangular()
# I = eye(mn)

# x = ones(mn - 1, 1)
deltac = c*ones(1, 1)
# deltac[0] = L1*c/L0
# deltac[1] = L2*c/L0

pprint(deltac)
print('-'*70)

# Calculate the product
# g = ((1/tau)*(L.T*L + y**2*I).inv()*L.T)*x
g_deltac = ((sa.inv() + k.T * so.inv() * k).inv() * k.T * so.inv()) * deltac
pprint(simplify(g_deltac[1]))
