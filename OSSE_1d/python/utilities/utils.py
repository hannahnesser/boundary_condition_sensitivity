import os
from pathlib import Path
import yaml

def setup(config_name='config.yaml'):
    # Get project directory
    project_dir = Path(os.path.abspath(__file__)).parent.parent.parent

    # Read in config file
    with open(f'{project_dir}/{config_name}', 'r') as file:
        config = yaml.safe_load(file)

    # Calculate the number of observations
    config['nobs'] = config['nstate']*config['nobs_per_cell']

    return project_dir, config