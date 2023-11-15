# Experiment-5-Implementation-of-XOR-using-RBF

## AIM:
  To classify the Binary input patterns of XOR data  by implementing Radial Basis Function Neural Networks.
  
## EQUIPMENTS REQUIRED:

Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
<img width="541" alt="image" src="https://user-images.githubusercontent.com/112920679/201299438-5d1926f9-25e9-4f20-b392-1c112880ef56.png">

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below
<img width="246" alt="image" src="https://user-images.githubusercontent.com/112920679/201299568-d9398233-71d8-41b3-8b08-a39d5b95e3f1.png">

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.

A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.


A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.

<img width="261" alt="image" src="https://user-images.githubusercontent.com/112920679/201300944-5510d7f4-ea0f-45ec-875d-87f463927e9d.png">

The RBF of hidden neuron as gaussian function 

<img width="206" alt="image" src="https://user-images.githubusercontent.com/112920679/201302321-a09f72e9-2352-4f88-838c-3324f6c5f57e.png">


## ALGORIHM:
## Step 1 :
Initialize the input patterns for XOR Gate

## Step 2:
Initialize the desired output of the XOR Gate

## Step 3:
Define the function for RBF and function for prediction.

## Step 4:
Plot the graphs with inputs

## Step 5:
Find the weights

## Step 6:
Plot the graph with transformed inputs using RBF

## Step 7 :
Test for the XOR patterns.

## PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt
def gaussian_rbf(x, landmark,gamma=1):
  return np.exp(-gamma*np.linalg.norm(x-landmark)**2)
def end_to_end(X1,X2,ys,mu1,mu2):
  from_1=[gaussian_rbf(i,mu1) for i in zip(X1,X2)]
  from_2=[gaussian_rbf(i,mu2) for i in zip(X1,X2)]
  #plot
  plt.figure(figsize=(13,5))
  plt.subplot(1,2,1)
  plt.scatter((x1[0],x1[3]),(x2[0],x2[3]),label="Class_0")
  plt.scatter((x1[1],x1[2]),(x2[1],x2[2]),label="Class_1")
  plt.xlabel("$X1$",fontsize=15)
  plt.ylabel("$X2$",fontsize=15)
  plt.title("Xor: Linearly Inseparable",fontsize=15)
  plt.legend()
  plt.subplot(1,2,2)
  plt.scatter(from_1[0],from_2[0],label="Class_0")
  plt.scatter(from_1[1],from_2[1],label="Class_1")
  plt.scatter(from_1[2],from_2[2],label="Class_2")
  plt.scatter(from_1[3],from_2[3],label="Class_3")
  plt.plot([0,0.95],[0.95,0],"k--")
  plt.annotate("Separating hyperplan",xy=(0.4,0.55),xytext=(0.55,0.66),arrowprops=dict(facecolor='black',shrink=0.05))
  plt.xlabel(f"$mu1$: {(mu1)}", fontsize=15)
  plt.ylabel(f"$mu2$: {(mu2)}", fontsize=15)
  plt.title("Transformed Inputs: Linearly Seperable", fontsize=15)
  plt.legend()
  A = []
  for i, j in zip(from_1, from_2):
      temp = []
      temp.append(i)
      temp.append(j)
      temp.append(1)
      A.append(temp)
    
  A = np.array(A)
  W = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(ys)
  print(np.round(A.dot(W)))
  print(ys)
  print(f"Weights: {W}")
  return W
  
def predict_matrix(point, weights):
    gaussian_rbf_0 = gaussian_rbf(np.array(point), mu1)
    gaussian_rbf_1 = gaussian_rbf(np.array(point), mu2)
    A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
    return np.round(A.dot(weights))
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])
w = end_to_end(x1, x2, ys, mu1, mu2)
print(f"Input:{np.array([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}")
print(f"Input:{np.array([0, 1])}, Predicted: {predict_matrix(np.array([0, 1]), w)}")
print(f"Input:{np.array([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")
print(f"Input:{np.array([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")
```


## OUTPUT :
![image](https://github.com/21005688/Experiment-5-Implementation-of-XOR-using-RBF/assets/94747031/e9f055ee-00df-4166-9834-8c05b3038bf3)

![image](https://github.com/21005688/Experiment-5-Implementation-of-XOR-using-RBF/assets/94747031/02e3034b-47f1-4613-968b-0822cb2ae03f)

![image](https://github.com/21005688/Experiment-5-Implementation-of-XOR-using-RBF/assets/94747031/5c46b2dc-1d57-4b39-98d2-b1d00271a6cb)

## RESULT:

Thus XOR is successfully implemeted using RBF.






