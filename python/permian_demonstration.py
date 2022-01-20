

local = False

if local:
    ...
else:
    data_dir = '/n/seasasfs02/dvaron/GEOSChem/files_for_hannah'
    code_dir = ''


import sys
sys.path.append('.')
import gcpy as gc
