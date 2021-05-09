import pandas as pd
from tqdm import tqdm
from handy_function import print_current_time
import tensorflow as tf
from BCE_with_logits_loss_weights import *
from predictions_metrics import *
from popularity_metric import *




class training_net():


        def __init__(self, net, train_dataloader, val_dataloader, args,  optimizer, device, num_items,  item_probabilities
):

            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.epochs = args.num_epochs
            self.optimizer = optimizer
            self.net = net
            self.device = device
            self.num_items = num_items
            self. item_probabilities =  item_probabilities
            self.use_validation = True
            if args.validation_ratio <= 0:
                self.use_validation = False

            self.train_loss= [None] * self.epochs
            self.val_loss= [None] * self.epochs
            self.val_correct_unifrom_masked_ones_ratio = [None] * self.epochs
            self.val_correct_pop_masked_ones_ratio = [None] * self.epochs
            self.epoch_before_eraly_stop = 0

            self.train_net()






        def get_val_metrics(self):

            epoch_total_val_loss =    predicted_uniform_masked_one_ratio=    predicted_pop_masked_one_ratio =0.0
            val_dataloader_length = len(self.val_dataloader)

            self.net.eval()

            for x_val in tqdm(self.val_dataloader):

                x_val = x_val.to(self.device)

                with torch.no_grad():

                    corrupted_uniform_val_batch, zero_unifrom_ind_arr, one_unifrom_ind_arr =  corrupt_val_batch(x_val)
                    x_uniform_corrupted_val_hat = self.net(corrupted_uniform_val_batch)
                    predicted_uniform_masked_one_ratio += calculate_correct_masked_ones_ratio( x_uniform_corrupted_val_hat,  zero_unifrom_ind_arr, one_unifrom_ind_arr)

                    corrupted_pop_val_batch, zeros_pop_ind_arr, ones_pop_ind_arr = corrupt_val_batch_pop_prob(x_val, self.item_probabilities)
                    x_pop_corrupted_val_hat = self.net(corrupted_pop_val_batch)
                    predicted_pop_masked_one_ratio += calculate_correct_masked_ones_ratio( x_pop_corrupted_val_hat, zeros_pop_ind_arr, ones_pop_ind_arr)

                    CF = BCE_with_logits_loss_weights( y= x_val,  batch_size= len(self.train_dataloader))
                    criterion_func = CF.get_criterion().to(self.device)
                    x_val_hat = self.net(x_val)
                    loss = criterion_func(x_val_hat,  x_val)
                    epoch_total_val_loss += loss

            epoch_val_uniform_corret_masked_ones_ratio = 0.0
            epoch_val_uniform_corret_masked_ones_ratio =   predicted_uniform_masked_one_ratio / val_dataloader_length

            epoch_val_pop_corret_masked_ones_ratio = 0.0
            epoch_pop_corret_masked_ones_ratio = predicted_pop_masked_one_ratio / val_dataloader_length

            epoch_val_loss = 0.0
            epoch_val_loss = epoch_total_val_loss.item() / val_dataloader_length

            return (epoch_val_uniform_corret_masked_ones_ratio,  epoch_pop_corret_masked_ones_ratio,  epoch_val_loss)


        def train_net(self):


            print_current_time("\n Starting to train net at:")
            first_batch_train = True

            for epoch in range(self.epochs):
                if epoch > 4 and self.use_validation:
                 if (self.early_stopping_check(epoch)):
                      break

                epoch_total_train_loss = 0.0

                self.net.train()
                for x_train in tqdm(self.train_dataloader):
                    x_train = x_train.to(self.device)


                    self.optimizer.zero_grad()

                    x_hat_train = self.net(x_train)

                    CF = BCE_with_logits_loss_weights(y= x_train,   batch_size= len(self.train_dataloader))
                    criterion_func =  CF.get_criterion().to(self.device)

                    loss =   criterion_func (x_hat_train, x_train)

                    loss.backward()
                    epoch_total_train_loss += loss

                    if first_batch_train:
                        first_batch_train = False
                        print ("\nThe loss value for the first batch in the first epoch: ", loss.item(), "\n")


                    self.optimizer.step()


                self.update_train_metrics(epoch, epoch_total_train_loss )
                self.print_metrics(epoch, Train = True)

                if self.use_validation:
                    self.val_correct_unifrom_masked_ones_ratio[epoch], self.val_correct_pop_masked_ones_ratio[epoch] ,\
                    self.val_loss[epoch]  = self.get_val_metrics()

                    self.print_metrics(epoch, Train=False)


        def update_train_metrics(self,epoch, epoch_total_train_loss):

            self.train_loss[epoch] = epoch_total_train_loss.item() / len(self.train_dataloader)

        def print_metrics(self, epoch, Train):

            if epoch % 10 == 0:
              print()
              print("******************************")

              if Train:
                print_current_time("Training metrics for epoch " + str(epoch) + ":\n")
                print("Loss value: ", self.train_loss[epoch])

              else:
                  self.epoch_before_eraly_stop =  epoch
                  print_current_time("Validation metrics for epoch " + str(epoch) + ":\n")
                  print("Correct unifrom masked ones ratio: ", self.val_correct_unifrom_masked_ones_ratio[epoch])
                  print("Correct popularity masked ones ratio: ", self.val_correct_pop_masked_ones_ratio[epoch])
                  print("Loss value: ", self.val_loss[epoch])



        def early_stopping_check(self, epoch):


            if self.val_loss[epoch-4] - self.val_loss[epoch-3] <=0  and self.val_loss[epoch-3]- self.val_loss[epoch-2] <=0  and   self.val_loss[epoch - 2] - self.val_loss[epoch - 1]<=0 :

                 print ("made early stopping after epoch: ", epoch-1)
                 return True


