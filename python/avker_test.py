import numpy as np
import pandas as pd
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
import inversion as inv
import plot
import plot_settings as ps
ps.SCALE = ps.PRES_SCALE
ps.BASE_WIDTH = ps.PRES_WIDTH
ps.BASE_HEIGHT = ps.PRES_HEIGHT
import format_plots as fp

rcParams['text.usetex'] = True
np.set_printoptions(precision=5, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = '../plots'
plot_dir = f'{plot_dir}/n{s.nstate}_m{s.nobs}'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Define sensitivity test parameters
## -------------------------------------------------------------------------##
# Default is opt_BC = False
test = inv.Inversion(gamma=1, nstate=5, so=30, U=10*24)

def inv_col(B):
    return np.linalg.inv(B.T @ B) @ B.T

def inv_row(B):
    return B.T @ np.linalg.inv(B @ B.T)

def AA_red(A, B, Bstar):
    return B @ A @ Bstar

print(test.xa_abs)
print(test.xa_abs.sum())
print('-'*70)
print(test.xhat*test.xa_abs)
print((test.xhat*test.xa_abs).sum())
print('-'*70)
print(test.a)

# Test 1: x_red = W x
# Test 1a: relative
Wrel = np.ones(test.nstate).reshape((1, -1))
Wrel_star = inv_row(Wrel)
xhat_red_1a = Wrel @ (test.xhat * test.xa_abs)
a_red_1a = AA_red(test.a, Wrel, Wrel_star)
print('-'*70)
print('TEST 1A')
print(xhat_red_1a)
print('-'*50)
print(a_red_1a)

# Test 1b: absolute
Wabs = test.xa_abs.reshape((1, -1))
Wabs_star = inv_row(Wabs)
xhat_red_1b = Wabs @ test.xhat
a_red_1b = AA_red(test.a, Wabs, Wabs_star)
print('-'*70)
print('TEST 1B')
print(xhat_red_1b)
print('-'*50)
print(a_red_1b)

# Test 1c: weighted
Wwei = test.xa_abs.reshape((1, -1))/test.xa_abs.sum()
Wwei_star = inv_row(Wwei)
xhat_red_1c = Wwei @ test.xhat
a_red_1c = AA_red(test.a, Wwei, Wwei_star)
print('-'*70)
print('TEST 1C')
print(xhat_red_1c)
print('-'*50)
print(a_red_1c)

# Test 2: x = W x_red (Calisesi 2005)
W = (test.xa_abs/(test.xa_abs).sum()).reshape((-1, 1))
Wstar = inv_col(W)
xhat_red_2 = Wstar @ (test.xhat * test.xa_abs)
a_red_2 = AA_red(test.a, Wstar, W)
print('-'*70)
print('TEST 2')
# print(W @ (test.xhat * test.xa_abs).sum().reshape((1, 1)))
print(xhat_red_2)
print('-'*50)
print(a_red_2)

print('-'*50)
print(a_red_2 - a_red_1b)

