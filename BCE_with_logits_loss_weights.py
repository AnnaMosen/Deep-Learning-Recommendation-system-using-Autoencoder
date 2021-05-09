import random
import torch
import numpy as np
from handy_function import *


class BCE_with_logits_loss_weights():


        def __init__(self, y, batch_size ):


           self.y = y
           self.criterion = torch.nn.BCEWithLogitsLoss()
           self.batch_size = batch_size
           weights_number = y.shape[1]
           self.weights = torch.from_numpy(np.zeros(weights_number))

           self.create_criterion()

        def create_weights(self):

            # rows = self.y.shape[0]
            # columns = self.y.shape[1]
            #
            # for j in range(columns):
            #     ones_counter = sum(1 for i in range(rows) if self.y[i][j] == 1)
            #     if ones_counter > 0:
            #         self.weights[j] = self.batch_size / ones_counter
            #     ones_counter = 0

            if self.y.is_cuda:
                self.y = self.y.cpu()
            y_np = self.y.numpy()

            ones_columns_counter = y_np.sum(axis=0)

            weights =  np.true_divide(self.batch_size, ones_columns_counter,  out=np.zeros(self.y.shape[1]), where=ones_columns_counter!=0)
            self.weights = torch.from_numpy(weights)




        def create_criterion(self) :
          self.create_weights()
          self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights)



        def get_criterion(self):
            return self.criterion





