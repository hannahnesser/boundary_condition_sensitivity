from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy import symbols, pprint
from sympy import *

# We will solve for the gain matrix for a range of state vector sizes
mn = 3

# Initialize the symbols we will use: tau is the lifetime (1/k),
# sa is the prior error variance (assumed constant), and 
# so is the observational error variance (also assumed constant).
# We also defne y = so/(sa*tau^2)
tau, sa, sbc, p, so, y, xa, xbc, deltay = symbols('t sa sbc p so y xa xbc c')

# Initialize the Jacobian
k = ones(mn)
k = k.lower_triangular()
k = tau*xa*k

for i in range(mn):
    k[i, -1] = xbc
k = k[:(mn - 1), :]
# # print(k)
# # k = 0.2*k

# Initialize the error matrices
so = so**2*eye(mn - 1)
sa = sa**2*eye(mn)
# so = 100*eye(mn)
# sa = 0.25*eye(mn)
# sa[0, 0] *= p
sa[-1, -1] = sbc**2

# and initialize matrix structures without the proper scaling
# L = ones(mn)
# L = L.lower_triangular()
# I = eye(mn)

# x = ones(mn - 1, 1)
deltay = deltay*ones(mn - 1, 1)

# Calculate the product
# g = ((1/tau)*(L.T*L + y**2*I).inv()*L.T)*x
g = ((sa.inv() + k.T * so.inv() * k).inv() * k.T * so.inv()) * deltay
pprint(simplify(g))
