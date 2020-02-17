#!/usr/bin/env python
# coding: utf-8

# In[189]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random


# In[190]:


#Defining a class for associative cells
class associative:
    def __init__(self,index,weight):
        self.index = 0
        self.weight = []


# In[208]:


# Generating Data
F=100
f=1
sample = 100
x = np.arange(sample)
y = np.sin(2*np.pi*f*x / F)


# In[209]:


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


# In[210]:


# plotting training data
plt.figure(1)
plt.plot(X_train,Y_train,'r+',label = 'input data')
plt.legend()
plt.show()


# In[211]:


associative_numbers = 35
g = 5
weights = np.ones((35,1))
lr = 1


# In[212]:


def assoc_val(index,g):
    weights = []
    b = g//2
    for i in range(index-b, index+b+1):
        weights.append(i)
    return weights    


# In[213]:


def assoc_ind(i,g,associative_numbers,sample):
    i  = int(i)
    ind = g//2 + ((associative_numbers - 2*(g//2))*i)/sample
    return math.floor(ind)


# In[214]:


# Calculating mean square error
def mean_square_error(weights,w,X_train,Y_train):
    meansqer = 0
    for i in range(0,len(w)):
        sum = 0
        for j in w[i]:
            sum = sum + weights[j]
        meansqer = meansqer + ((sum - Y_train[i]))**2
    return meansqer    


# In[215]:


# Function for testing the data
def testing(weights,w):
    output = []
    for i in range(0,len(w)):
        sum = 0
        for j in w[i]:
            sum = sum + weights[j]
        output.append(sum)
    return output


# In[216]:


# Calling the associative class
train = associative([],[])
test = associative([],[])


# In[217]:


# Creating the associative cells 
for index in X_train:
    train.index = assoc_ind(index,g,associative_numbers,sample)
    train.weight.append(assoc_val(train.index,g))


# In[218]:


# Creating the associative cells
for iy in X_test:
    test.index = assoc_ind(iy,g,associative_numbers,sample)
    test.weight.append(assoc_val(test.index,g))


# In[219]:


err_list = []
err_plot = []


# In[220]:


previouserror = 0
currenterror = 15
iterations = 0


# In[221]:


while iterations<100 and abs(previouserror - currenterror) > 0.00001:
    previouserror = currenterror
    for i in range (0, len(train.weight)):
        sum = 0
        for j in train.weight[i]:
            sum = sum + weights[j]
        error = sum - Y_train[i]
        correction  = error/g
        for j in train.weight[i]:
            weights[j] -= lr*correction
    currenterror = float(mean_square_error(weights,train.weight,X_train,Y_train))
    err_list.append(currenterror)
    iterations = iterations + 1
    err_plot.append(iterations)


# In[222]:


# Plotting error convergence
plt.plot(np.asarray(err_plot), np.asarray(err_list), 'r--',label = 'error convergence')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()


# In[223]:


output = testing(weights, test.weight)


# In[224]:


plt.plot(X_train,Y_train,'g+',label = 'training data')
plt.plot(X_test,Y_test,'r+',label = 'test data')
plt.plot(X_test,np.asarray(output),'bo', label = 'predicted outputs')
plt.legend()
plt.show()


# In[225]:


plt.plot(X_test,Y_test,'g+',label = 'test data')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




