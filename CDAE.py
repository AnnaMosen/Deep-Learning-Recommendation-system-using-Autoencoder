import torch
from torch import nn
import torch.nn.functional as F


class CDAE(torch.nn.Module):


   def __init__(self, args, num_users, num_items, device= 0):

       super(CDAE, self).__init__()
       self.hidden_dim = args.hidden_dim
       self.num_users = num_users
       self.num_items = num_items

       self.encoder = nn.Linear(self.num_items, self.hidden_dim)
       self.decoder = nn.Linear(self.hidden_dim, self.num_items)



   def forward(self, rating_matrix):

       enc = self.encoder(rating_matrix)
       enc =  F.tanh(enc)
       dec = self.decoder(enc)

       return dec
