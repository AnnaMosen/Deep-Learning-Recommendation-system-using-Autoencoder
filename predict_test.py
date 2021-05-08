import pandas as pd

class predict_test():

    def __init__(self, x_hat,  test_csv_path):

      self.x_hat =  x_hat
      self.orig_test_df = pd.read_csv(test_csv_path)
      self.test_results_df =  self.orig_test_df.copy()
      self.test_results_df["result"] = 0

      
    
    def get_results(self):
      
      for i in range(len(self.test_results_df)):
         user = self.test_results_df.loc[i, "UserID"]
         first_item =  self.test_results_df.loc[i, "Item1"]
         second_item =  self.test_results_df.loc[i, "Item2"]

         self.get_user_result(i, user-1, first_item-1, second_item-1)

    def get_user_result(self,i, user, item1, item2):


       if self.x_hat[user][item1] > self.x_hat[user][item2]:
            self.test_results_df.loc[i, "result"] = 0
       else:
           self.test_results_df.loc[i, "result"] = 1







      