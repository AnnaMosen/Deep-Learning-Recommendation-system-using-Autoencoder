from training_matrix import *
from CDAE import *
from args import *
from training_net import *
from torch import optim
from predict_test import *





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




cdae_args = args()
training_matrix =training_users_items_matrix("Train.csv" ,
                                             cdae_args,
                                             "rating_matrix.csv"
                                             )
#cdae_args.create_weighted_bce(training_matrix.training_data_weights)


untrained_cdae_net = CDAE(args = cdae_args , num_users = training_matrix.max_user_id, num_items = training_matrix.max_item_id)
untrained_cdae_net.to(device)

if  cdae_args.weight_decay> 0 :
  adam_optimizer = optim.Adam(untrained_cdae_net.parameters(), lr=cdae_args.lr , weight_decay= cdae_args.weight_decay )
else:
    adam_optimizer = optim.Adam(untrained_cdae_net.parameters(), lr=cdae_args.lr)

trained_cdae_net = training_net(net = untrained_cdae_net , train_dataloader = training_matrix.train_dataloader,
                                val_dataloader= training_matrix.val_dataloader, args= cdae_args,
                                optimizer= adam_optimizer, device= device, num_items = training_matrix.max_item_id,
                                item_probabilities= training_matrix.item_probabilities )




rating_matrix = pd.read_csv("rating_matrix.csv")
user_item_matrix_tensor = torch.Tensor(np.array(rating_matrix))
rating_matrix_dataloader = DataLoader(user_item_matrix_tensor, batch_size= cdae_args.tr_batch_size)


cdae_args_no_validation = args(validation_ratio= 0, num_epochs= trained_cdae_net.epoch_before_eraly_stop + 1 )
trained_cdae_net_no_validation= training_net(net = untrained_cdae_net , train_dataloader = rating_matrix_dataloader,
                                val_dataloader= None, args= cdae_args_no_validation,
                                optimizer= adam_optimizer, device= device, num_items = training_matrix.max_item_id,
                                item_probabilities= None )


first_batch = True

for x_batch in  rating_matrix_dataloader:
     x_batch = x_batch.to(device)
     x_hat_batch = trained_cdae_net_no_validation.net(x_batch )
     x_hat_sigmoid_batch = torch.sigmoid(x_hat_batch)
     if first_batch:
         x_hat =  x_hat_sigmoid_batch
         first_batch = False
     else:
         x_hat = torch.cat((x_hat,  x_hat_sigmoid_batch ), 0)

test_results = predict_test(x_hat, "RandomTest.csv")
test_results.get_results()
test_results.test_results_df.to_csv("RandomTestResults.csv")

test_results = predict_test(x_hat, "PopularityTest.csv")
test_results.get_results()
test_results.test_results_df.to_csv("PopularityTestResults.csv")



print ("finish")
