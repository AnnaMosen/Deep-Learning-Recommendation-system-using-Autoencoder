import torch
from numpy import where
from numpy.random import choice
import numpy as np
from predictions_metrics import   corrupt_user_vector
import pandas as pd
from scipy.special import softmax

def sample_popularity_zero_index( user_item_vector, item_probabilities ):

    if  user_item_vector.is_cuda:
        user_item_vector=  user_item_vector.cpu()

    zero_indices = where(user_item_vector == 0)[0]
    zeros_item_prob =[item_probabilities.get(key+1) for key in zero_indices]
    #zeros_item_prob = [x if x != None else 0 for x in zeros_item_prob ]
    zeros_item_prob = np.array (zeros_item_prob)
    user_zeros_item_prob= softmax(zeros_item_prob)

    from random import choices
    chosen_zero_item_index = choices(zero_indices ,  user_zeros_item_prob)[0]


    return  chosen_zero_item_index





def random_one_label (x):

    if x.is_cuda:
        x = x.cpu()

    x= x.detach().numpy()

    random_one_index =  random_index_of_value(x,1)

    return  random_one_index



def corrupt_val_batch_pop_prob(val_batch,  item_probabilities):

    batch_size = val_batch.shape[0]
    zero_ind_arr = []
    one_ind_arr = []


    for user_index in range(batch_size):
        single_user_items_vector =  val_batch[user_index]
        random_one_index =  random_one_label ( single_user_items_vector)

        popularity_zero_index = sample_popularity_zero_index( val_batch[user_index], item_probabilities )


        zero_ind_arr.append(  popularity_zero_index)
        one_ind_arr.append(random_one_index)


        corrupted_val_batch= corrupt_user_vector( val_batch, random_one_index, user_index)



    return(corrupted_val_batch, zero_ind_arr, one_ind_arr)


def random_index_of_value(x, value):
    indices = where(x == value)[0]

    return choice(indices)
