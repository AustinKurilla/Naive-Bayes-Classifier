#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Naive bayes classifier algorithm for numerical data.
import numpy as np
import os
import math

trainingFile = input ("Enter training filename: ")
testFile = input ("Enter testing filename: ")
#trainingFile = 'irisTraining.txt'
#testFile = 'irisTesting.txt'


# In[3]:


arrayTrain = []
arrayTest = []

with open(trainingFile, 'r') as f:
        for line in f.readlines():
            arrayTrain.append(line.split(' '))

with open(testFile, 'r') as g:
        for line in g.readlines():
            arrayTest.append(line.split(' '))
            
arrayTrain = np.array(arrayTrain).astype(float)
arrayTest = np.array(arrayTest).astype(float)
cols = len(arrayTrain.T)
#print (cols)


# In[4]:


#sort testing array based on if the class value is 1(trainingP) or -1(trainingN)
trainingP = np.array(arrayTrain[np.where(arrayTrain[:,cols - 1] == 1)]).T.astype(float)
trainingN = np.array(arrayTrain[np.where(arrayTrain[:,cols - 1] == -1)]).T.astype(float)


# In[5]:


#calculate mean of each column, seperate based on 1(P) and -1(N)
meanofnumsP = np.sum(trainingP, axis=1) #mean of all 1's
meanofnumsN = np.sum(trainingN, axis=1) #mean of all -1's
for x in range (cols - 2):
    meanofnumsP[x] = meanofnumsP[x] / len(arrayTrain)
    meanofnumsN[x] = meanofnumsN[x] / len(arrayTrain)
#print (meanofnumsP)
#print (meanofnumsN)

#calculate std deviation of each column, seperate based on 1(P) and -1(N)
devofnumsP = np.std(trainingP, axis=1)
devofnumsN = np.std(trainingN, axis=1)
#print(devofnumsP)
#print(devofnumsN)


# In[6]:


#probability density function
def prob_density(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**(0.5)
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    pdf = num/denom
    if(pdf == 0):
        return 0.000001
    else:
        return pdf
#print(prob_density(120,110,54.54356)) test example from slides


# In[7]:


TP = 0
FP = 0
TN = 0
FN = 0
PYES = len(trainingP[0]) / (len(trainingP[0]) + len(trainingN[0]))
PNO = 1-PYES
resultsP = [] #all records classified to class '1'
resultsN = [] #all records classified to class '-1'
#PofXYES = probablity new record belongs to class '1' PofXNO = probablity new record beleongs to class '-1' whichever 
#number is larger between the 2 decides which class the record belongs too
for x in range (len(arrayTest)):
    PofXYES = 1
    PofXNO = 1
    for y in range (cols - 1):
        PofXYES = (prob_density(arrayTest[x][y],meanofnumsP[y],devofnumsP[y]) * PofXYES)
        PofXNO = (prob_density(arrayTest[x][y],meanofnumsN[y],devofnumsN[y]) * PofXNO)
    PofXYES = PofXYES * PYES        
    PofXNO = PofXNO * PNO
    if PofXYES > PofXNO:
        resultsP.append(arrayTest[x]) 
    else:
        resultsN.append(arrayTest[x])


# In[8]:


#Find number of TP,FP,FN, and TN by sorting arrays into postive and negative, then sorting those arrays based on if 
#their last value is -1 or 1
resultsP = np.array(resultsP).astype(float)
#rint(resultsP) #print all '1' results
if len(resultsP) > 0:
    TP = len(np.array(resultsP[np.where(resultsP[:,cols - 1] == 1)]).astype(float))
    FP = len(np.array(resultsP[np.where(resultsP[:,cols - 1] == -1)]).astype(float))
else:
    TP =0
    FP =0
resultsN = np.array(resultsN).astype(float)
if len(resultsN) > 0:
    FN = len(np.array(resultsN[np.where(resultsN[:,cols - 1] == 1)]).astype(float))
    TN = len(np.array(resultsN[np.where(resultsN[:,cols - 1] == -1)]).astype(float))
else:
    FN =0
    TN =0
#print (TP + FP + FN + TN) #make sure all records are classified


# In[9]:


#calculate data
if (TP + FP + FN + TN) > 0:
    accuracy = (TP + TN) / (TP + FP + FN + TN)
else:
    accuracy = 0
if (TP + FP) > 0:
    percision = TP / (TP + FP)
else:
    percision = 0
if (TP + FN) > 0:
    recall = TP / (TP + FN)
else:
    recall = 0


# In[10]:


#display data
print('Accuracy : ' + str(accuracy))
print('True Positive : ' + str(TP))
print('True Negative : ' + str(TN))
print('False Positive : ' + str(FP))
print('False Negative : ' + str(FN))
print('Percision : ' + str(percision))
print('Recall : ' + str(recall))


# In[ ]:





# In[ ]:




