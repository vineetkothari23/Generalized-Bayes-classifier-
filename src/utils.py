import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math 

#Return a numpy arrary of the matrix
def read_mat_file(file_dir):
  data = scipy.io.loadmat(file_dir)
  return data

def std_normalize(mat,mean, std_dev):
  '''
  Returns standard normalized matrix
  z=(x-mean)/std_dev
  '''
  mat=(mat-mean)/std_dev
  return mat

def get_normal_prob(mean, std_dev, test):
    prob = test - mean
    prob = np.multiply(prob,prob)
    prob = -1 * prob / (2 * np.multiply(std_dev,std_dev))
    prob = np.exp(prob)
    prob = prob/(math.sqrt(math.pi*2)*std_dev)
    #prob = np.prod(prob, axis = 1)
    return prob

def predict(variables):
  # Outputs the class predicted
  prob=[get_normal_prob(variables['m_std_train']['f1'][i]['m'],\
                        variables['m_std_train']['f1'][i]['std'],
                        variables['test_data']) \
        for i in range(5)]
  return np.argmax(prob,axis=0)


def get_mean_std(array):
  # m: mean ; std: standard deviation
  dict={'m':np.mean(array),'std':np.std(array)}
  return dict

def get_accuracy(predicted):
  n_samples=predicted.shape[0]
  ground_truth=np.array([[0,1,2,3,4] for _ in range(n_samples)])
  from sklearn import metrics
  from sklearn.metrics import accuracy_score
  acc = np.mean(np.array([accuracy_score(predicted[:,i],ground_truth[:,i]) for i in range(5)]))
  print("Accuracy: {:.2f} Error rate: {:.2f} \n".format(acc,1-acc))
  return acc

def plot_clustered_graph(x,y,c,name="image.png",labels=None):
  classes = ['C1','C2','C3','C4','C5']
  #colors = [plt.cm.jet(i/float(len(unique)-1)) for i in range(len(unique))]

  scatter=plt.scatter(x, y, c=c,label=c)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.xlabel(labels[0])
  plt.ylabel(labels[1])
  plt.grid(True)
  plt.show()
  
  #plt.save(name)

def get_multivariate_pdf(mean, cov, test):
  from scipy.stats import multivariate_normal
  import numpy as np
  y = multivariate_normal.pdf(test, mean=mean, cov=cov)
  return y
