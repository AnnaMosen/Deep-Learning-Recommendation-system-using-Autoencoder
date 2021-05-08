import numpy as np
import torch
from numpy import where
from numpy.random import choice




def random_index_of_value(x, value):

    indices = where(x == value)[0]

    return choice(indices)


def random_zero_and_one_label (x):

    if x.is_cuda:
        x = x.cpu()

    x= x.detach().numpy()

    random_zero_index = random_index_of_value(x,0)
    random_one_index =  random_index_of_value(x,1)

    return random_zero_index, random_one_index


def corrupt_user_vector( val_batch, random_one_index, user_index):

   val_batch[user_index][random_one_index] = 0

   return val_batch



def corrupt_val_batch(val_batch):

    batch_size = val_batch.shape[0]
    zero_ind_arr = []
    one_ind_arr = []

    for user_index in range(batch_size):
        single_user_items_vector =  val_batch[user_index]
        random_zero_index, random_one_index =  random_zero_and_one_label ( single_user_items_vector)

        zero_ind_arr.append(random_zero_index)
        one_ind_arr.append(random_one_index)

        corrupted_val_batch= corrupt_user_vector( val_batch, random_one_index, user_index)



    return(corrupted_val_batch, zero_ind_arr, one_ind_arr)


def calculate_correct_masked_ones_ratio(val_batch, zero_ind_arr, one_ind_arr):

    sigmoid_x_batch_hat = torch.sigmoid(val_batch)
    predicted_masked_one_num = 0

    for user_index in range(len(sigmoid_x_batch_hat)):


        user_zero_rand_item_index =   zero_ind_arr[user_index]
        user_one_rand_item_index  =   one_ind_arr[user_index]

        if   sigmoid_x_batch_hat[user_index][user_zero_rand_item_index] <   sigmoid_x_batch_hat[user_index][user_one_rand_item_index]:
            predicted_masked_one_num += 1

    predicted_masked_one_ratio =   predicted_masked_one_num / len(val_batch)

    return predicted_masked_one_ratio