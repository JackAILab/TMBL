import numpy as np
from random import random
from config import get_config
from data_loader import get_loader
from solver import Solver
# from solver_original import Solver
import torch
import warnings


warnings.filterwarnings("ignore")  # ignore the warning

if __name__ == '__main__':
    # Setting random seed
    random_name = str(random())
    random_seed = 55654
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Solver is a wrapper for model traiing and testing
    # solver = Solver
    solver = Solver(train_config,  train_data_loader, dev_data_loader, test_data_loader, is_train=True) 

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()






















