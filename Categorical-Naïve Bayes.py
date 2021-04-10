#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import math

trainingFile = input ("Enter training filename: ")
testFile = input ("Enter testing filename: ")


# In[3]:


arrayTrain = []
arrayTest = []

with open(trainingFile, 'r') as f:
        for line in f.readlines():
            arrayTrain.append(line.split(' '))

with open(testFile, 'r') as g:
        for line in g.readlines():
            arrayTest.append(line.split(' '))

arrayTrain = np.array(arrayTrain).astype(int)
arrayTest = np.array(arrayTest).astype(int)
cols = len(arrayTrain.T)


# In[4]:


#sort testing array based on if the class value is 1(trainingP) or -1(trainingN)
trainingP = np.array(arrayTrain[np.where(arrayTrain[:,cols - 1] == 1)]).T.astype(int)
trainingN = np.array(arrayTrain[np.where(arrayTrain[:,cols - 1] == -1)]).T.astype(int)
#print (trainingP)
#print (trainingN)


# In[5]:


#these arrays give a summary of the number of 1's,2's, and 3's in each column. SummaryP give a summary of all values 
#with class '1' and summaryN gives a summary of all values with class '-1' 
#the nested arrays represent each column and the values represents the number of 1, 2, and 3's in each column
# [[# of 1's in col 1, # of 2's in col 1. # of 3's in col 1][#of 1's in col 2, etc..][col3][col4]]

summaryP = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
summaryN = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
for x in range (len(trainingP) - 1):
    a = 0
    b = 0
    c = 0
    for y in range (len(trainingP[x])):
        if (trainingP[x][y] == 1):
            a = a + 1
        elif(trainingP[x][y] == 2):
            b = b + 1
        elif(trainingP[x][y] == 3):
            c = c + 1
    summaryP[x][0] = a + 1 #laplace correction
    summaryP[x][1] = b + 1
    summaryP[x][2] = c + 1

for x in range (len(trainingN) - 1):
    a = 0
    b = 0
    c = 0
    for y in range (len(trainingN[x])):
        if (trainingN[x][y] == 1):
            a = a + 1
        elif(trainingN[x][y] == 2):
            b = b + 1
        elif(trainingN[x][y] == 3):
            c = c + 1
    summaryN[x][0] = a + 1 #laplace correction
    summaryN[x][1] = b + 1
    summaryN[x][2] = c + 1
#print(summaryP)
#print(summaryN)


# In[6]:


resultsP = [] #all records classified to class '1'
resultsN = [] #all records classified to class '-1'
PYES = len(trainingP[0]) / (len(trainingP[0]) + len(trainingN[0]))
PNO = 1-PYES
for x in range (len(arrayTest)):
    PofYES = 1
    PofNO = 1
    for y in range(cols - 1):
            PofYES = PofYES * ((summaryP[y][(arrayTest[x][y]-1)])/len(trainingP[0])) #
            PofNO = PofNO * ((summaryN[y][(arrayTest[x][y]-1)])/len(trainingN[0]))
    if PofYES > PofNO:
        resultsP.append(arrayTest[x]) 
    else:
        resultsN.append(arrayTest[x])


# In[7]:


#print (resultsP)
#print (resultsN)


# In[8]:


TP = 0
FP = 0
TN = 0
FN = 0
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




