import numpy as np
import sys
sys.path.append('.')
import forward_model as fm

## Standard inversion settings
# Random state
random_state  = 372 #872
# rs = np.random.RandomState(random_state)

# Inversion dimensions
nstate        = 20
nobs_per_cell = 15
nobs = nstate*nobs_per_cell

# Courant number
Cmax          = 1

# Grid cell length (km)
L             = 25

# Wind speed (km/day)
U             = 5*24

# Times (day)
init_t        = 150/24
total_t       = 300/24

# True quantities
BC_t          = 1900 # Boundary condition (ppb)
x_abs_t       = 30 # Emissions (ppb/day)

# Prior errors (ppb/day)
sa            = 0.75
sa_BC         = 0.05

# Observational errors (ppb)
so            = 1

# Plotting materials
xp            = np.arange(1, nstate + 1)