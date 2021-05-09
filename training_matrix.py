import os
# os.chdir("C:/Users/amit_/PycharmProjects/pythonProject/deep_final_project/US")
# print(os.getcwd())
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from args import *
import torch

print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")


class training_users_items_matrix:

    def __init__(self, raw_data_csv_path, args, rating_matrix_csv_path=None):

        self.raw_data = pd.read_csv(raw_data_csv_path)
        self.validation_ratio = args.validation_ratio
        self.traning_batch_size = args.tr_batch_size
        self.val_bacth_size = args.val_batch_size

        self.temp_data = self.raw_data.copy()
        self.temp_data = self.temp_data.sort_values("UserID")

        self.unique_users_id = self.temp_data.iloc[:, 0].unique()


        self.item_probabilities = (self.raw_data['ItemID'].value_counts() / self.raw_data.shape[0]).to_dict()
        self.item_probabilities[1750] = 0.0



        self.training_data_weights = [0,0]

        if rating_matrix_csv_path is None:
            self.max_item_id = max(self.temp_data.iloc[:, 1])
            self.max_user_id = max(self.temp_data.iloc[:, 0])
            self.user_item_train_matrix = pd.DataFrame(np.zeros((self.max_user_id, self.max_item_id)))
            self.create_training_matrix()
        else:
            self.user_item_train_matrix = pd.read_csv(rating_matrix_csv_path)
            self.max_item_id = self.user_item_train_matrix.shape[1]
            self.max_user_id = self.user_item_train_matrix.shape[0]

        self.user_item_val_matrix = pd.DataFrame(np.zeros((self.max_user_id, self.max_item_id)))


        if self.validation_ratio > 0:
            self.split_train_val_data()
        else:
            print("train set wasn't splitted to validation set")

        self.create_dataloader()

        self.get_minority_class_ratio()


    def create_training_matrix(self):

        for id in self.unique_users_id:
            self.create_user_vector(id)

    def create_user_vector(self, id):

        delete_indcies = []
        user_vector = np.zeros(self.max_item_id)
        find_id = False

        for i in range(len(self.temp_data)):
            if self.temp_data.loc[i, "UserID"] == id:
                find_id = True
                item_id = self.temp_data.loc[i, "ItemID"]
                user_vector[item_id - 1] = 1
                delete_indcies.append(i)
            else:
                if find_id:
                    break

        self.temp_data.drop(self.temp_data.index[delete_indcies], inplace=True)
        self.temp_data = self.temp_data.reset_index(drop=True)

        self.user_item_train_matrix.loc[id - 1] = user_vector

    def split_train_val_data(self):

        from sklearn.model_selection import train_test_split
        self.user_item_train_matrix, self.user_item_val_matrix = train_test_split(self.user_item_train_matrix, test_size=self.validation_ratio, random_state=1)


    def create_dataloader(self):

        user_item_train_matrix_tensor = torch.Tensor(np.array(self.user_item_train_matrix))
        self.train_dataloader = DataLoader(user_item_train_matrix_tensor, batch_size=self.traning_batch_size)

        if self.validation_ratio > 0:
            user_item_val_matrix_tensor = torch.Tensor(np.array(self.user_item_val_matrix))
            self.val_dataloader = DataLoader(user_item_val_matrix_tensor, batch_size=self.val_bacth_size)

    def get_minority_class_ratio(self):
        classes_counts =   np.unique(self.user_item_train_matrix.values, return_counts=True)
        minority_class =   classes_counts[1][1]
        majority_class =   classes_counts[1][0]

        minoirty_class_weight = majority_class/   minority_class
        majority_class_weight = majority_class / majority_class

        self.training_data_weights = [majority_class_weight, minoirty_class_weight ]


































