tau = 0.2
sigma_o = 10
sigma_b = 5
sigma_a = 0.5

def g_2d(tau, sigma_o, sigma_a, sigma_b):
    term1 = sigma_o**2/(sigma_a**2*tau**2)
    term2 = sigma_o**2/(sigma_b**2*tau**2)
    coeff = 1/(tau*((term2 + 2)*(term1 + 1) - 1))
    return 10*coeff*(2*term1 + 1), 10*coeff*term2

print(g_2d(tau, sigma_o, sigma_a, sigma_b))

print(g_2d(tau, sigma_o, sigma_a, sigma_a))