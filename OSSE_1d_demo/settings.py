## Standard inversion settings
# Random state
random_state  = 572

# Inversion dimensions
nstate        = 20
nobs_per_cell = 50
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
BCt           = 1900 # Boundary condition (ppb)
xt_abs        = 30 # Emissions (ppb/day)

# Prior errors (ppb/day)
sa            = 0.5

# Observational errors (ppb)
so            = 10