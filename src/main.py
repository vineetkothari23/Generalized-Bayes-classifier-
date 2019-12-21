import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math 
from utils import read_mat_file, std_normalize, get_normal_prob, get_mean_std, get_accuracy, plot_clustered_graph, get_multivariate_pdf
from Multivariate_Bayes_Classifier import Multivariate_Bayes_Classifier
from Bayes_Classifier import Bayes_Classifier

data_path="data.mat" # Change data path here if required
# Link to example data file : 
training_size=100
variables={"data_path":data_path, \
           "training_size":training_size, \
           "convert_std_normal": False
           }

def main(variables):
  data=read_mat_file(variables["data_path"])
  f1_data,f2_data=np.array(data['F1']),np.array(data['F2'])
  n_samples=f1_data.shape[0]
  ground_truth=np.array([[0,1,2,3,4] for _ in range(n_samples)])
  print("About the data")
  print("Source of data: ",variables["data_path"])
  print("Classes of data: 0,1,2,3,4")
  print("No. of samples: ",n_samples,"\n")

  
  #Training on 100 samples
  
  #m_std is dictionary of f1, f2 for each column, c1 c2 c3 c4 and c5.
  print("\n---------- Section 1: Training -------------")
  print("\n Calculating the means and standard deviations for 100 samples\n")
  train_size=variables['training_size']
  b1=Bayes_Classifier(f1_data,train_size)
  m_std_train=b1.train()

  ## Section 2.1: Testing  
  print("\n---------- Section 2.1: Testing -------------")
  print("\n Predicting the classes for 101: 1000 samples")

  predicted=b1.predict()

  ## Section 2.2: Calculating accuracy and error rate
  print("\n---------- Section 2.2: Calculating accuracy for the classifier -------------")
  print("\nAccuracy for the Bayes classifier: ")
  acc=b1.validate(predicted)

  ## Section 3: Standard Normal (z score)
  print("---------- Section 3: Standard normal(Z Score) -------------")

  # z1_data is the standard normalized data.
  z1_data=np.swapaxes(np.array([std_normalize(f1_data[:,i],m_std_train['f1'][i]['m'],\
                        m_std_train['f1'][i]['std']) 
          for i in range(5)]),0,1)
  print("Plot of Z1 vs F2")
  plot_clustered_graph(z1_data.flatten(),f2_data.flatten(),ground_truth.flatten(),name="z1vsf2.png",labels=['z1','f2'])

  # z1_data is the standard normalized data.
  print("\n Plot of F1 vs F2")
  plot_clustered_graph(f1_data.flatten(),f2_data.flatten(),ground_truth.flatten(),name="f1vsf2.png",labels=['f1','f2'])

  ## Section 4
  ### Case 1: Training with the z1 data
  print("\n---------- Section 4, Case 2: Training with the z1 data -------------")
  b=Bayes_Classifier(z1_data)
  b.train()
  predicted=b.predict()
  acc=b.validate(predicted)

  print("\n---------- Section 4, Case 3: Training with the f2 data -------------")
  b=Bayes_Classifier(f2_data)
  b.train()
  predicted=b.predict()
  acc=b.validate(predicted)

  print("\n---------- Section 4, Case 4: Training with the [z1, f2] data -------------")
  data={'z1':z1_data,'f2':f2_data}
  b=Multivariate_Bayes_Classifier(data)
  b.train()
  predicted=b.predict()
  acc=b.validate(predicted)

main(variables)
