import os
import torch
import pandas as pd
import numpy as np
import datetime
from focalloss import *

print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")


class args:

    def __init__(self,  hidden_dim = 50, validation_ratio = 0.2,
                  tr_batch_size = 128, val_batch_size = 128, num_epochs= 1000,
                  lr = 0.0005, weight_decay= 1e-4 ):






        # Train-test split args

         self.validation_ratio = validation_ratio

        # NN architecture args
         self.hidden_dim =  hidden_dim

        # NN training args

         self.tr_batch_size =  tr_batch_size
         self.val_batch_size = val_batch_size
         self.num_epochs = num_epochs
         self.lr = lr

    # NN backpropagation args

         self.lr = lr
         self.weight_decay = weight_decay







