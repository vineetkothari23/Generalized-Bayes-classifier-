import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math 
from utils import read_mat_file, std_normalize, get_normal_prob, get_mean_std, get_accuracy, plot_clustered_graph, get_multivariate_pdf

class Multivariate_Bayes_Classifier(object):
  def __init__(self,data,train_size=100):
    '''
    Constructing the classifier basics, and initializing the variables

    Attributes:
    data = A dictionary of multiple features of the dataset. Each feature is a mxn matrix, 
            where m is the no. of samples, n is just to distinguish the datapoints with their respective column indices as labels.
    feature_values = A list of all feature values concatenated together
    feature_names= Names of all the features, basically the keys of dictionary 'data'
    train_size= An integer value to slice the data. The remaining acts as the testing data
    n_features: No. of features attributing to the labels in the data
    n_samples: No. of samples in the data
    n_classes: An integer for o. of the classes foe the 
    '''
    self.data=data
    #self.feature_values = np.fromiter(self.data.values(),dtype=float)
    self.feature_values=np.array(list(self.data.values()))
    self.feature_names=np.array(list(self.data.keys()))
    self.train_size=train_size
    print("Dataset shape: ",self.feature_values.shape)
    self.n_features=self.feature_values.shape[0]
    self.n_samples=self.feature_values.shape[1]
    self.n_classes=self.feature_values.shape[2]
    self.ground_truth=np.array([[0,1,2,3,4] for _ in range(self.n_samples)])

  def train(self):

    self.m_std_train={}
    for class_i in range(self.n_classes):
      temp={'m':[],'cov':[]}
      for feature_name, feature_mat in self.data.items():
        temp['m'].append(np.mean(self.data[feature_name][:self.train_size, class_i]))
      temp['cov']=np.cov(self.data[self.feature_names[0]][:self.train_size, class_i],self.data[self.feature_names[1]][:self.train_size, class_i])
      temp['m']=np.array(temp['m'])
      self.m_std_train[class_i]=temp

    self.test_data=[]
    for sample_i in range(self.n_samples):
      temp_sample=[]
      for class_i in range(self.n_classes):
        temp_val=[]
        for feature_name, feature_mat in self.data.items():
          temp_val.append(feature_mat[sample_i][class_i])
        temp_sample.append(np.array(temp_val))
      self.test_data.append(temp_sample)
    self.test_data=np.array(self.test_data)[self.train_size:]

  def predict(self):
    prob=np.array([get_multivariate_pdf(self.m_std_train[i]['m'],\
                            self.m_std_train[i]['cov'], \
                            self.test_data) \
            for i in range(self.n_classes)])
    return np.argmax(prob,axis=0)

  def validate(self,predicted):
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    acc = np.mean(np.array([accuracy_score(predicted[:,i],self.ground_truth[self.train_size:,i]) for i in range(5)]))
    print("Accuracy: {:.2f} Error rate: {:.2f} \n".format(acc,1-acc))
    return acc
