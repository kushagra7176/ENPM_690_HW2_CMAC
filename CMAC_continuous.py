
# Importing all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Generating Data
F=100
f=1
sample = 100
x = np.arange(sample)
y = np.sin(2*np.pi*f*x / F)
# Plotting the 1-D function
plt.plot(x,y)
plt.show()
# Dividing training data and testing data
fun = np.stack((x.T,y.T), axis=0)
data = fun.T
np.random.shuffle(data)
train_data = data[:70]
test_data = data[70:]
X_train = train_data[:,0]
Y_train = train_data[:,1]
X_test = test_data[:,0]
Y_test = test_data[:,1]




# Defining a class for associative cells
class associative:
    def __init__(self,index,weight):
        self.index = 0
        self.weight = []




# Plotting the training data
plt.figure(1)
plt.plot(X_train,Y_train,'r+',label = 'input data')
plt.legend()
plt.show()




associative_numbers = 35
g = 5
weights = np.ones((35,1))
lr = 1




def assoc_val(ind,g):
    weights = []
    b = g/2
    weightage = []
    index = []
    # weightages for top edge cell
    index = math.floor(ind - b)
    weightage = math.ceil(ind - b) - (ind - b)
    
    topcell = []
    topcell.append(index)
    topcell.append(weightage)
    
    if weightage != 0:
        weights.append(topcell)
    # weightages for middle cells
    for index in range(math.ceil(ind - b), math.floor(ind + b + 1)):
        midcell = []
        midcell.append(index)
        midcell.append(1)
        weights.append(midcell)
    # weightages for bottom edge cell
    index = math.floor(ind + g/2)
    weightage = (ind + g/2) - math.floor(ind + g/2)
    
    bottomcell = []
    bottomcell.append(index)
    bottomcell.append(weightage)
    
    if weightage!=0:
        weights.append(bottomcell)
     
    return weights




def assoc_ind(i,g,associative_numbers,sample):
    i  = int(i)
    ind = g/2 + ((associative_numbers - 2*(g/2))*i)/sample
    return math.floor(ind)




# Calculating mean square error
def mean_square_error(weights,w,X_train,Y_train):
    meansqer = 0
    for i in range(0,len(w)):
        sum = 0
        for j in w[i]:
            sum = sum + (weights[j[0]]*j[1])
        meansqer = meansqer + ((sum - Y_train[i]))**2
    return meansqer    




# Function for testing the data
def testing(weights,w):
    output = []
    for i in range(0,len(w)):
        sum = 0
        for j in w[i]:
            sum = sum + (weights[j[0]]*j[1])
        output.append(sum)
    return output




# Calling the associative class
train = associative([],[])
test = associative([],[])




# Creating the associative cells
for index in X_train:
    train.index = assoc_ind(index,g,associative_numbers,sample)
    train.weight.append(assoc_val(train.index,g))




# Creating the asociative cells
for iy in X_test:
    test.index = assoc_ind(iy,g,associative_numbers,sample)
    test.weight.append(assoc_val(test.index,g))




err_list = []
err_plot = []




previouserror = 0
currenterror = 10
iterations = 0




while iterations<100 and abs(previouserror - currenterror) > 0.001:
    previouserror = currenterror
    for i in range (0, len(train.weight)):
        sum = 0
        for j in train.weight[i]:
            sum = sum + weights[j[0]]*j[1]
        error = sum - Y_train[i]
        correction  = error/g
        for j in train.weight[i]:
            weights[j[0]] -= lr*correction*j[1]
    currenterror = float(mean_square_error(weights,train.weight,X_train,Y_train))
    err_list.append(currenterror)
    iterations = iterations + 1
    err_plot.append(iterations)




plt.plot(np.asarray(err_plot), np.asarray(err_list), 'r--',label = 'error convergence')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()




output = testing(weights, test.weight)
Accuracy =float(mean_square_error(weights,test.weight,X_test,Y_test))




plt.plot(X_train,Y_train,'g+',label = 'training data')
plt.plot(X_test,Y_test,'r+',label = 'test data')
plt.plot(X_test,np.asarray(output),'bo', label = 'predicted outputs')
plt.legend()
plt.show()











