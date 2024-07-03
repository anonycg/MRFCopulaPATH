#!/usr/bin/env python
# coding: utf-8

# In[1]:


#MRF- general 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from itertools import product
from math import comb
import numpy as np
import math
F=6   #6 variables: exchange sex, condom use, employment, housing, poverty, education

F_x = [i for i in product(range(2), repeat=F)]#generate all combinations of feature values
F_x = np.array(F_x)
F_x=F_x.astype(int)
#print(F_x)




# In[2]:


#HETF
# pairwise joint distributions
pairwise_data =np.array([
 
[0.323901084047373,0.91317134654436,0.92870811780839,0.860544926413633,0.831034886848313,0.869834902077005,0.895484871205071],
[0.0133132016669132,0.0158286534556404,0.000291882191609639,0.0684550735863672,0.0639651131516874,0.0251650979229947,0.0705151287949292],
[0.605098915952627,0.0528286534556404,0.0699518821916096,0.0344550735863672,0.0906900197929577,0.0961650979229947,0.0262400354361995],
[0.0576867983330868,0.0181713465443596,0.00104811780839036,0.0365449264136328,0.0143099802070423,0.00883490207700532,0.00775996456380046],

]
    
    
)
pairwise_J=np.transpose(pairwise_data)


# In[2]:


#MSM
# pairwise joint distributions
pairwise_data =np.array([
[0.504631314996899,0.901687835649135,0.927817291761446,0.842761009982384,0.831034886848313,0.869834902077005,0.895484871205071],
[0.0353686850031011,0.0273121643508652,0.00118270823855391,0.086238990017616,0.0639651131516874,0.0251650979229947,0.0705151287949292],
[0.424368685003101,0.0643121643508652,0.0700027082385539,0.052238990017616,0.0906900197929577,0.0961650979229947,0.0262400354361995],
[0.0356313149968989,0.00668783564913477,0.000997291761446095,0.018761009982384,0.0143099802070423,0.00883490207700532,0.00775996456380046],

    
]
    
    
)
pairwise_J=np.transpose(pairwise_data)


# In[3]:


pairwise_J


# In[3]:

# E_p_emp
if F==6:
    features=np.array([[1,0],[0,2],[0,3],[0,4],[4,5],[4,2],[2,5]])
    E_p_emp = pairwise_J[0:1,:] 
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[1:2,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[2:3,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[3:4,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[4:5,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[5:6,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[6:7,:]),1 )
    print(E_p_emp)


# In[4]:


phi_F=np.ones((2**F,4*np.shape(features)[0]))*-10
for j in range(0,np.shape(features)[0]):
    for i in range(0,2**F):
        if F_x[i,features[j,0]]==0 and F_x[i,features[j,1]]==0: phi_F[i,j*4] = 1 
        else: phi_F[i,j*4] = 0
        
        if F_x[i,features[j,0]]==0 and F_x[i,features[j,1]]==1: phi_F[i,j*4+1] = 1 
        else: phi_F[i,j*4+1] = 0
        
        if F_x[i,features[j,0]]==1 and F_x[i,features[j,1]]==0: phi_F[i,j*4+2] = 1 
        else: phi_F[i,j*4+2] = 0
        
        if F_x[i,features[j,0]]==1 and F_x[i,features[j,1]]==1: phi_F[i,j*4+3] = 1 
        else: phi_F[i,j*4+3] = 0     
#print(phi_F)


# In[5]:


### Theta
theta = np.ones((1,4*np.shape(features)[0]))*0.1
                     
## E-p
E_p = np.ones((1,4*np.shape(features)[0]))*-10
phi_times_theta = np.ones((1,2**F))*-10
dI_by_dTheta= np.ones((1,4*np.shape(features)[0]))*-10
### Gradient Descent parameters: 
m = 1
alpha = 0.8
A = -500
B = 1000


# In[6]:


for m in range (20000):
    for i in range(0,2**F):
        phi_times_theta[0,i]=math.exp(np.dot(theta,phi_F[i,:]))
    p_y_theta=phi_times_theta/np.sum(phi_times_theta)
    #print(p_y_theta)
    for i in range(0,np.shape(phi_F)[1]):
        E_p[0,i]=np.dot(phi_F[:,i],np.transpose(p_y_theta))
    dI_by_dTheta =E_p_emp- E_p
    mu = A / (m + B + 1)**alpha
    theta=theta-dI_by_dTheta*mu
#print(p_y_theta)


# In[7]:


#check MRF fit
plt.plot(range(0,4*np.shape(features)[0]),np.transpose(E_p),label='actual')
plt.plot(range(0,4*np.shape(features)[0]),np.transpose(E_p_emp), label='predict')
plt.legend()
plt.show()


# In[8]:


plt.scatter(E_p_emp, E_p)
plt.show()


# In[9]:


df = pd.DataFrame(F_x, columns=['exchange','condom','emp','hous','pov','edu'])
df



# In[10]:


value_df = pd.DataFrame(p_y_theta.T)
final_df = pd.concat([df, value_df], axis=1)

#final_df.to_excel('MSM_sexual__MRF_paper.xlsx', index=False)
final_df.to_excel('HETF_sexual__MRF_paper.xlsx', index=False)


# In[10]:


p_y_theta_T=(np.transpose(p_y_theta))

# In[ ]:[0,1],	[0,2],	[0,3],	[0,4],	[0,5],	[0,6],	[5,3],	[7,3],	[3,6]
Verify=np.zeros((11,np.shape(features)[0]))
for j in range(0,np.shape(features)[0]):
    
    counter_1 =0
    counter_2 =0
    for i in range(0,2**F):
        A=F_x[i,features[j,0]]
        B=F_x[i,features[j,1]]
        
        #0|0
        if A==0 and B==0:
            counter_1=counter_1 +p_y_theta[0,i]
        if B==0:
            counter_2=counter_2 +p_y_theta[0,i]
    if counter_2 > 0: Verify[0,j]=counter_1/counter_2
    else:  Verify[0,j]=0
    
for j in range(0,np.shape(features)[0]):
    counter_1 =0
    counter_2 =0
    for i in range(0,2**F):
        A=F_x[i,features[j,0]]
        B=F_x[i,features[j,1]]
        
        #0|1
        if A==0 and B==1:
            counter_1=counter_1 +p_y_theta[0,i]
        if B==1:
            counter_2=counter_2 +p_y_theta[0,i]
    if counter_2 > 0: Verify[1,j]=counter_1/counter_2
    else:  Verify[1,j]=0
    if Verify[0,j]>0: Verify[2,j]=Verify[1,j]/Verify[0,j] #RR
    else: Verify[2,j]=0
    
for j in range(0,np.shape(features)[0]):
    counter_1 =0
    counter_2 =0
    for i in range(0,2**F):
        A=F_x[i,features[j,0]]
        B=F_x[i,features[j,1]]
        
        #0|1
        if B==0:
            counter_1=counter_1 +p_y_theta[0,i]
        if B==1:
            counter_2=counter_2 +p_y_theta[0,i]
    Verify[3,j]=counter_1
    Verify[4,j]=counter_2
    
for j in range(0,np.shape(features)[0]):
    counter_1 =0
    counter_2 =0
    for i in range(0,2**F):
        A=F_x[i,features[j,0]]
        B=F_x[i,features[j,1]]
        
        #0|1
        if A==0:
            counter_1=counter_1 +p_y_theta[0,i]
        if A==1:
            counter_2=counter_2 +p_y_theta[0,i]
    Verify[5,j]=counter_1
    Verify[6,j]=counter_2
    Verify[8,j]=1-Verify[0,j]
    Verify[9,j]=1-Verify[1,j]
    Verify[10,j]=Verify[9,j]/Verify[8,j]
Verify  


# In[11]:


Verify = pd.DataFrame(Verify)

Verify.to_excel('sexual_verify_15.xlsx', index=False)


# In[13]:


#relative prevalence (risk) of joint values
RR_matrix =np.zeros((F,F))
for j in range(0,F):
    for k in range(0,F):
        counter_11=0
        counter_12=0
        counter_21=0
        counter_22=0
        for i in range(0,2**F):
            if F_x[i,j]==1 and F_x[i,k]==1: counter_11=counter_11 +p_y_theta[0,i]
            if F_x[i,k]==1 : counter_12=counter_12 +p_y_theta[0,i]
            if F_x[i,j]==1 and F_x[i,k]==0: counter_21=counter_21 +p_y_theta[0,i]
            if F_x[i,k]==0 : counter_22=counter_22 +p_y_theta[0,i]
        RR_matrix[j,k]= (counter_11/counter_12)/(counter_21/counter_22)
RR_matrix    


# In[ ]:




