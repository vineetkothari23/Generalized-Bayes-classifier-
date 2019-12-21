import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from utils import read_mat_file, std_normalize, get_normal_prob, get_mean_std, get_accuracy, plot_clustered_graph, get_multivariate_pdf

class Bayes_Classifier(object):
    """This is a bayes classifier, which predicts classes 
        based on the mean and standard deviations of a feature.

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, data, train_size=100):

        self.data = data
        self.n_samples = data.shape[0]
        self.train_size=train_size
        self.test_data=data[train_size:]
        #self.m_std_train, self.predicted, self.accuracy=
        self.m_std_train={}
        self.ground_truth=np.array([[0,1,2,3,4] for _ in range(self.n_samples)])

    def train(self):
        self.m_std_train={'f1':{0:get_mean_std(self.data[:self.train_size,0]), 
               1:get_mean_std(self.data[:self.train_size,1]), 
               2:get_mean_std(self.data[:self.train_size,2]),   
               3:get_mean_std(self.data[:self.train_size,3]), 
               4:get_mean_std(self.data[:self.train_size,4])}, 
         'f2':{0:get_mean_std(self.data[:self.train_size,0]), 
               1:get_mean_std(self.data[:self.train_size,1]), 
               2:get_mean_std(self.data[:self.train_size,2]),   
               3:get_mean_std(self.data[:self.train_size,3]), 
               4:get_mean_std(self.data[:self.train_size,4])}}
        return self.m_std_train
        
        
    def predict(self):
      # Outputs the class predicted
      prob=[get_normal_prob(self.m_std_train['f1'][i]['m'],\
                            self.m_std_train['f1'][i]['std'],
                            self.test_data) \
            for i in range(5)]
      return np.argmax(prob,axis=0)

    def validate(self,predicted):
      
      from sklearn import metrics
      from sklearn.metrics import accuracy_score
      acc = np.mean(np.array([accuracy_score(predicted[:,i],self.ground_truth[self.train_size:,i]) for i in range(5)]))
      print("Accuracy: {:.2f} Error rate: {:.2f} \n".format(acc,1-acc))
      return acc

    def scatter_plot(x,y,c):
      fig, ax = plt.subplots()
      scatter = ax.scatter(x, y, c=c)
      ax.legend()
      ax.grid(True)
      plt.show()


