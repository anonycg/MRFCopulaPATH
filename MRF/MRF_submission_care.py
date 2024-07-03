#!/usr/bin/env python
# coding: utf-8


#MRF- general 

# In[1]:

#required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from itertools import product
from math import comb
import numpy as np
import math
F=8 # 8 variables: VLS	depression	neighborhood	housing	poverty	education	insurance	employment

 
F_x = [i for i in product(range(2), repeat=F)]#generate all combinations of feature values
F_x = np.array(F_x)
F_x=F_x.astype(int)
#print(F_x)



# In[2]:


# mean
pairwise_data =np.array([

[0.511193491385658,0.41515144766147,0.610990788126919,0.408507561176794,0.558004827603413,0.650590498230653,0.759055124718315,0.783949787085726,0.882003193430657,0.666898938613809,0.562295930736709,0.557111442391659,0.530668158356475,0.811470217290628,0.737692446827542,0.839551026607184],
[0.143806508614343,0.23984855233853,0.0440092118730809,0.246492438823206,0.096995172396587,0.00440950176934693,0.0719448752816847,0.0770502129142736,0.0229968065693431,0.0631010613861905,0.00870406926329073,0.0138885576083405,0.0403318416435247,0.0195297827093722,0.0933075531724576,0.133448973392816],
[0.218806508614343,0.20484855233853,0.294009211873081,0.162492438823206,0.272995172396587,0.322409501769347,0.145944875281685,0.121050212914274,0.0909968065693431,0.238101061386191,0.268704069263291,0.41588855760834,0.330331841643525,0.161529782709372,0.123307553172458,0.0214489733928161],
[0.126193491385657,0.14015144766147,0.0509907881269191,0.182507561176794,0.072004827603413,0.0225904982306531,0.0230551247183153,0.0179497870857264,0.00400319343065694,0.0318989386138095,0.160295930736709,0.0131114423916596,0.0986681583564754,0.00747021729062781,0.0456924468275424,0.00555102660718393],   
    
]

)
pairwise_J=np.transpose(pairwise_data)


# In[2]:


#MRF_min
pairwise_data =np.array([

[0.492875435458055,0.4061,0.603871517078076,0.408336555274917,0.557718413637032,0.596260740047082,0.757127767395198,0.782741050729924,0.881475428630986,0.667652101119398,0.562295930736709,0.557111442391659,0.489112746746624,0.811470217290628,0.737692446827542,0.839551026607184],
[0.162124564541945,0.2489,0.0511284829219242,0.246663444725083,0.0972815863629677,0.058739259952918,0.0738722326048015,0.0782589492700762,0.0235245713690135,0.0623478988806022,0.00870406926329073,0.0138885576083405,0.0818872532533759,0.0195297827093722,0.0933075531724576,0.133448973392816],
[0.237124564541945,0.2139,0.301128482921924,0.162663444725083,0.273281586362968,0.308739259952918,0.147872232604801,0.122258949270076,0.0915245713690135,0.237347898880602,0.268704069263291,0.41588855760834,0.371887253253376,0.161529782709372,0.123307553172458,0.0214489733928161],
[0.107875435458055,0.1311,0.0438715170780758,0.182336555274917,0.0717184136370323,0.036260740047082,0.0211277673951985,0.0167410507299238,0.00347542863098651,0.0326521011193978,0.160295930736709,0.0131114423916596,0.0571127467466242,0.00747021729062781,0.0456924468275424,0.00555102660718393],

   
]

)
pairwise_J=np.transpose(pairwise_data)


# In[2]:


#MRF_max                                  
pairwise_data =np.array([

[0.532113616720358,0.432479136811936,0.628906609519513,0.408660261970854,0.558515640038937,0.650934558823529,0.759055124718315,0.784342792829316,0.882406391436972,0.669447969020351,0.562295930736709,0.557111442391659,0.569701557902786,0.811470217290628,0.737692446827542,0.839551026607184],
[0.122886383279642,0.222520863188064,0.0260933904804873,0.246339738029146,0.0964843599610631,0.00406544117647056,0.0719448752816847,0.0766572071706842,0.0225936085630279,0.060552030979649,0.00870406926329073,0.0138885576083405,0.00129844209721391,0.0195297827093722,0.0933075531724576,0.133448973392816],
[0.197886383279642,0.187520863188064,0.276093390480487,0.162339738029146,0.272484359961063,0.254065441176471,0.145944875281685,0.120657207170684,0.0905936085630279,0.235552030979649,0.268704069263291,0.41588855760834,0.291298442097214,0.161529782709372,0.123307553172458,0.0214489733928161],
[0.147113616720358,0.157479136811936,0.0689066095195127,0.182660261970854,0.072515640038937,0.0909345588235294,0.0230551247183153,0.0183427928293157,0.00440639143697211,0.034447969020351,0.160295930736709,0.0131114423916596,0.137701557902786,0.00747021729062781,0.0456924468275424,0.00555102660718393],
   
]

)
pairwise_J=np.transpose(pairwise_data)







### E-P_emp
if F==8:
    features=np.array([[0,1],	[0,2],	[0,3],	[0,4],	[0,5],	[0,6],	[5,3],	[7,3],	[3,6],	[1,3],	[4,5],	[4,6],	[4,7],	[5,6],	[5,7],	[6,7]
])
    E_p_emp = pairwise_J[0:1,:] 
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[1:2,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[2:3,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[3:4,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[4:5,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[5:6,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[6:7,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[7:8,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[8:9,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[9:10,:]),1 )
    
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[10:11,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[11:12,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[12:13,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[13:14,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[14:15,:]),1 )
    E_p_emp = np.concatenate((E_p_emp,pairwise_J[15:16,:]),1 )
    
    print(E_p_emp)



    


# In[5]:


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


# In[6]:


### Theta
theta = np.ones((1,4*np.shape(features)[0]))*0.1
                     
## E-p
E_p = np.ones((1,4*np.shape(features)[0]))*-10
phi_times_theta = np.ones((1,2**F))*-10
dI_by_dTheta= np.ones((1,4*np.shape(features)[0]))*-10
### Gradient Descent parameters: 
m = 1
alpha = 0.8
A = -30
B = 1000


# In[9]:


for m in range (40000):
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



# In[12]:


#check MRF fit
plt.plot(range(0,4*np.shape(features)[0]),np.transpose(E_p),label='actual')
plt.plot(range(0,4*np.shape(features)[0]),np.transpose(E_p_emp), label='predict')
plt.legend()
plt.show()


# In[13]:


plt.scatter(E_p_emp, E_p)
plt.show()


# In[17]:


T_p_y_theta=np.transpose(p_y_theta)

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
    # Verify[8,j]=1-Verify[0,j]
    # Verify[9,j]=1-Verify[1,j]
    # Verify[10,j]=Verify[9,j]/Verify[8,j]
Verify    



# In[23]:


df = pd.DataFrame(F_x, columns=['vls','dep','neigh','hous','pov','edu','insur','emp'])
df



# In[24]:


value_df = pd.DataFrame(p_y_theta.T)
final_df = pd.concat([df, value_df], axis=1)


# In[26]:


final_df.to_excel('new__MRF_paper.xlsx', index=False)


# In[14]:


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


# In[15]:


RR_matrix = pd.DataFrame(RR_matrix)

RR_matrix.to_excel('RR_matrix.xlsx', index=False)


# In[ ]:




