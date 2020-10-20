#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[75]:



df = pd.read_csv("C:/Users/Faster/Desktop/data.csv") 


# In[76]:



df.drop(['Unnamed: 32','id'], axis = 1) 
print(df.shape)
df.drop(['Unnamed: 32'],axis=1,inplace = True) 
print(df.shape)

print (df)


# In[77]:



def diagnosis_value(diagnosis): 
    if diagnosis == 'M': 
        return 1
    else: 
        return 0
  
df['diagnosis'] = df['diagnosis'].apply(diagnosis_value) 


# In[78]:


import seaborn as sns

sns.lmplot(x = 'radius_mean', y = 'texture_mean', hue = 'diagnosis', data = df) 


# In[79]:



sns.lmplot(x ='smoothness_mean', y = 'compactness_mean',  
           data = df, hue = 'diagnosis') 


# In[80]:



X = np.array(df.iloc[:, 1:]) 
y = np.array(df['diagnosis'])


# In[81]:



from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size = 0.33, random_state = 42) 


# In[99]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train) 


# In[100]:


pred = knn.predict(X_test)
knn.score(X_test, y_test) 


# In[101]:



neighbors = [] 
cv_scores = [] 
  
from sklearn.model_selection import cross_val_score 
# perform 10 fold cross validation 
for k in range(1, 51, 2): 
    neighbors.append(k) 
    knn = KNeighborsClassifier(n_neighbors = k) 
    scores = cross_val_score( 
        knn, X_train, y_train, cv = 20, scoring = 'accuracy') 
    cv_scores.append(scores.mean()) 


# In[102]:



MSE = [1-x for x in cv_scores] 
  
# determining the best k 
optimal_k = neighbors[MSE.index(min(MSE))] 
print('The optimal number of neighbors is % d ' % optimal_k) 
  
# plot misclassification error versus k 
plt.figure(figsize = (10, 6)) 
plt.plot(neighbors, MSE) 
plt.xlabel('Number of neighbors') 
plt.ylabel('Misclassification Error') 
plt.show() 


# In[ ]:




