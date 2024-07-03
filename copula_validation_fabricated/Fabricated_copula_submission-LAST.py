#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries

import pandas as pd
import scipy.stats
import scipy.special as sp
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math


# In[2]:


### Bi-variate Bernoulli distribution with high dependency (large w): 
## This represents the national level distribution of two social conditions

p00 = 0.93651105
p01 = 0.01348895
p10 = 0.02048895
p11 = 0.02951105 

px = 0.05
py = 0.043

w = (p00*p11)/(p01*p10)
w


# In[3]:


### Given the initial Bernoulli distribution, we generate 100 samples from that distribution each with sample size of 1000.
### This represents 100 jurisdictions and 1000 people in every jurisdiction

Juri_dict = {}

for j in range(100):

    
    data = {'p00': [], 'p01': [], 'p10': [], 'p11': [], 'px': [], 'py' : []}
    df = pd.DataFrame(data)
    

    for n in range(1000):
        random_value = np.random.rand()
        #print(random_value)
        if random_value <= p00:
            df.at[n, 'p00'] = 1
        else:
            df.at[n, 'p00'] = 0
            

        if random_value > p00 and random_value<= p00+p01:
            df.at[n, 'p01'] = 1
        else:
            df.at[n, 'p01'] = 0
            
            
        if random_value > p00+p01 and random_value <= p00+p01+p10:
            df.at[n, 'p10'] = 1
        else:
            df.at[n, 'p10'] = 0
            
            
        if random_value > p00+p01+p10:
            df.at[n, 'p11'] = 1
        else:
            df.at[n, 'p11'] = 0
            
        df.at[n, 'px'] = df.at[n, 'p10'] + df.at[n, 'p11']
        df.at[n, 'py'] = df.at[n, 'p01'] + df.at[n, 'p11']
        
        
    Juri_dict[j] = df   
        


# In[4]:


Juri_dict[1]


# In[5]:


marginals_x_list = np.array([])
marginals_y_list = np.array([])

p00_list = np.array([])
p01_list = np.array([])
p10_list = np.array([])
p11_list = np.array([])

for key in Juri_dict.keys():
    #print(Juri_dict[key])
    
    
    count_ones_x = (Juri_dict[key]['px'] == 1).sum()
    marginals_x = count_ones_x/Juri_dict[key]['px'].shape[0]
    marginals_x_list = np.append(marginals_x_list, marginals_x)

    count_zeros_y = (Juri_dict[key]['py'] == 1).sum()
    marginals_y = count_zeros_y/Juri_dict[key]['py'].shape[0]
    marginals_y_list = np.append(marginals_y_list, marginals_y)
    
    
    count_ones_p00 = (Juri_dict[key]['p00'] == 1).sum()
    n_p00 = count_ones_p00/len(Juri_dict[key])
    p00_list = np.append(p00_list, n_p00)

    count_ones_p01 = (Juri_dict[key]['p01'] == 1).sum()
    n_p01 = count_ones_p01/len(Juri_dict[key])
    p01_list = np.append(p01_list, n_p01)
    
    
    count_ones_p10 = (Juri_dict[key]['p10'] == 1).sum()
    n_p10 = count_ones_p10/len(Juri_dict[key])
    p10_list = np.append(p10_list, n_p10)
    
    count_ones_p11 = (Juri_dict[key]['p11'] == 1).sum()
    n_p11 = count_ones_p11/len(Juri_dict[key])
    p11_list = np.append(p11_list, n_p11)
    
    
    
df_marginals = pd.DataFrame({'marginals_x': marginals_x_list, 'marginals_y': marginals_y_list, 'n_p00':p00_list, 'n_p01': p01_list, 'n_p10':p10_list, 'n_p11':p11_list})


# In[6]:


df_marginals


# In[7]:


df_pairwise = df_marginals.copy()



# In[4]:


# multiplying by a constant as number of sample in each jurisdiction
df_pairwise['n_sample'] = 1000 
x_values = df_pairwise['n_sample']*df_pairwise['marginals_x']  # Extract X values from the samples
y_values = df_pairwise['n_sample']*df_pairwise['marginals_y']  # Extract Y values from the samples

x_values=x_values.apply(lambda x: round(x))
y_values=y_values.apply(lambda x: round(x))


# In[5]:


# Calculate the mean and standard deviation for X
mean_x = x_values.mean()
std_x = x_values.std()

# Calculate the mean and standard deviation for Y
mean_y = y_values.mean()
std_y = y_values.std()


# In[6]:


x_values.max()


# In[7]:


y_values.max()


# In[8]:


from scipy.stats import truncnorm

# Define the bounds for the truncated normal distribution
lower_bound_x, upper_bound_x = 0, x_values.max()+10  #80
lower_bound_y, upper_bound_y = 0, y_values.max()+10  #70 

# Fit a truncated normal distribution to x_values
a_x = (lower_bound_x - mean_x) / std_x
b_x = (upper_bound_x - mean_x) / std_x

# Fit a truncated normal distribution to y_values
a_y = (lower_bound_y - mean_y) / std_y
b_y = (upper_bound_y - mean_y) / std_y

# Calculate the CDF of the truncated normal distribution for x_values
cdf_x = truncnorm.cdf(x_values, a=a_x, b=b_x, loc=mean_x, scale=std_x)

# Calculate the CDF of the truncated normal distribution for y_values
cdf_y = truncnorm.cdf(y_values, a=a_y, b=b_y, loc=mean_y, scale=std_y)

# The CDF values are the transformed uniform values
uniform_x = cdf_x
uniform_y = cdf_y

u = uniform_x 
v = uniform_y


# In[9]:


u0 = truncnorm.cdf(0.5, a=a_x, b=b_x, loc=mean_x, scale=std_x)
v0 = truncnorm.cdf(0.5, a=a_y, b=b_y, loc=mean_y, scale=std_y)


# In[10]:


plt.subplot(1, 2, 2)
plt.hist(v, bins=30, density=True, alpha=0.7, color='green')
plt.title('Transformed Distribution (Uniform(0,1))')
plt.xlabel('F(Height)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()




# In[11]:


# calculate the Gaussian copula density function: c(u,v)

df_pairwise['x'] = x_values
df_pairwise['y'] = y_values

df_pairwise['u'] = u
df_pairwise['v'] = v

############### calculating 2u-1 & 2v-1
df_pairwise['2u-1'] = (u * 2) - 1
df_pairwise['2v-1'] = (v * 2) - 1

############### calculating the inverse of error function for 2u-1 & 2v-1
df_pairwise['inverse_erf_2u-1'] = sp.erfinv(df_pairwise['2u-1'])
df_pairwise['inverse_erf_2v-1'] = sp.erfinv(df_pairwise['2v-1'])


#################### calculating the a & b
df_pairwise['a'] = math.sqrt(2)*df_pairwise['inverse_erf_2u-1']
df_pairwise['b'] = math.sqrt(2)*df_pairwise['inverse_erf_2v-1']


ro, p = scipy.stats.pearsonr(x_values,y_values)
ro


# In[12]:


def Copula_Gaussian(row):
    a = row['a']
    b = row['b']
    copula = (1/(math.sqrt(1 - ro**2)))*(math.exp(-(((a**2 + b**2) * ro**2 - 2*a*b*ro)/(2*(1-ro**2)))))

    return copula


############## calculating the Copula for each marginal data points 
df_pairwise['c(u, v)'] = df_pairwise.apply(Copula_Gaussian, axis = 1)


# In[13]:


# Create positive and negative offsets
positive_x = x_values + 0.5
negative_x = x_values - 0.5

# Create new columns in the DataFrame
df_pairwise['i'] = negative_x
df_pairwise['o'] = positive_x


# In[14]:


# Create positive and negative offsets
positive_y = y_values + 0.5
negative_y = y_values - 0.5

# Create new columns in the DataFrame
df_pairwise['c'] = negative_y
df_pairwise['d'] = positive_y





# In[15]:



# Iterate over the rows and calculate the integral for each row
integrals = []
for index, row in df_pairwise.iterrows():
    lower_limit = row['i']
    upper_limit = row['o']
    integral = truncnorm.cdf(upper_limit, a=a_x, b=b_x, loc=mean_x, scale=std_x) - truncnorm.cdf(lower_limit, a=a_x, b=b_x, loc=mean_x, scale=std_x)

    integrals.append(integral)

# Add the calculated integrals as a new column in the DataFrame
df_pairwise['pr(i<x<o)'] = integrals


# In[16]:



# Iterate over the rows and calculate the integral for each row
integrals = []
for index, row in df_pairwise.iterrows():
    lower_limit = row['c']
    upper_limit = row['d']
    integral = truncnorm.cdf(upper_limit, a=a_y, b=b_y, loc=mean_y, scale=std_y) - truncnorm.cdf(lower_limit, a=a_y, b=b_y, loc=mean_y, scale=std_y)
    integrals.append(integral)

# Add the calculated integrals as a new column in the DataFrame
df_pairwise['pr(c<y<d)'] = integrals


# In[17]:


df_pairwise['p(i<x<o, c<y<d)_copula']= df_pairwise['c(u, v)']*df_pairwise['pr(i<x<o)']*df_pairwise['pr(c<y<d)']


# In[18]:


df_pairwise['p(i<x<o, c<y<d)_copula'] 


# In[19]:

df_pairwise['u0'] = u0
df_pairwise['v0'] = v0

############### calculating 2u-1 & 2v-1
df_pairwise['2u0-1'] = (u0 * 2) - 1
df_pairwise['2v0-1'] = (v0 * 2) - 1

############### calculating the inverse of error function for 2u-1 & 2v-1
df_pairwise['inverse_erf_2u0-1'] = sp.erfinv(df_pairwise['2u0-1'])
df_pairwise['inverse_erf_2v0-1'] = sp.erfinv(df_pairwise['2v0-1'])


#################### calculating the a & b
df_pairwise['a0'] = math.sqrt(2)*df_pairwise['inverse_erf_2u0-1']
df_pairwise['b0'] = math.sqrt(2)*df_pairwise['inverse_erf_2v0-1']


def Copula_Gaussian_0(row):
    a = row['a0']
    b = row['b0']
    copula = (1/(math.sqrt(1 - ro**2)))*(math.exp(-(((a**2 + b**2) * ro**2 - 2*a*b*ro)/(2*(1-ro**2)))))
    return copula

############## calculating the Copula for each marginal data points 
df_pairwise['c(u0, v0)'] = df_pairwise.apply(Copula_Gaussian_0, axis = 1)


# In[20]:


df_pairwise['u0'] = u0

############### calculating 2u-1 & 2v-1
df_pairwise['2u0-1'] = (u0 * 2) - 1


############### calculating the inverse of error function for 2u-1 & 2v-1
df_pairwise['inverse_erf_2u0-1'] = sp.erfinv(df_pairwise['2u0-1'])



#################### calculating the a & b
df_pairwise['a0'] = math.sqrt(2)*df_pairwise['inverse_erf_2u0-1']



def Copula_Gaussian_0(row):
    a = row['a0']
    b = row['b']
    copula = (1/(math.sqrt(1 - ro**2)))*(math.exp(-(((a**2 + b**2) * ro**2 - 2*a*b*ro)/(2*(1-ro**2)))))
    return copula

############## calculating the Copula for each marginal data points 
df_pairwise['c(u0, v)'] = df_pairwise.apply(Copula_Gaussian_0, axis = 1)


# In[21]:


df_pairwise['v0'] = v0

############### calculating 2u-1 & 2v-1

df_pairwise['2v0-1'] = (v0 * 2) - 1

############### calculating the inverse of error function for 2u-1 & 2v-1

df_pairwise['inverse_erf_2v0-1'] = sp.erfinv(df_pairwise['2v0-1'])


#################### calculating the a & b
df_pairwise['b0'] = math.sqrt(2)*df_pairwise['inverse_erf_2v0-1']


def Copula_Gaussian_0(row):
    a = row['a']
    b = row['b0']
    copula = (1/(math.sqrt(1 - ro**2)))*(math.exp(-(((a**2 + b**2) * ro**2 - 2*a*b*ro)/(2*(1-ro**2)))))
    return copula

############## calculating the Copula for each marginal data points 
df_pairwise['c(u, v0)'] = df_pairwise.apply(Copula_Gaussian_0, axis = 1)



# In[22]:


# Create positive and negative offsets
positive_x = 0 + 0.5
negative_x = 0

# Create new columns in the DataFrame
df_pairwise['x_0'] = negative_x
df_pairwise['x+'] = positive_x



# In[23]:


# Create positive and negative offsets
positive_y = 0 +0.5
negative_y = 0

# Create new columns in the DataFrame
df_pairwise['y_0'] = negative_y
df_pairwise['y+'] = positive_y






# In[24]:


# Iterate over the rows and calculate the integral for each row
integrals = []
for index, row in df_pairwise.iterrows():
    lower_limit = row['x_0']
    upper_limit = row['x+']
    integral = truncnorm.cdf(upper_limit, a=a_x, b=b_x, loc=mean_x, scale=std_x) - truncnorm.cdf(lower_limit, a=a_x, b=b_x, loc=mean_x, scale=std_x)

    integrals.append(integral)

# Add the calculated integrals as a new column in the DataFrame
df_pairwise['pr(0<x<x+)'] = integrals


# In[25]:


# Iterate over the rows and calculate the integral for each row
integrals = []
for index, row in df_pairwise.iterrows():
    lower_limit = row['y_0']
    upper_limit = row['y+']
    integral = truncnorm.cdf(upper_limit, a=a_y, b=b_y, loc=mean_y, scale=std_y) - truncnorm.cdf(lower_limit, a=a_y, b=b_y, loc=mean_y, scale=std_y)
    integrals.append(integral)

# Add the calculated integrals as a new column in the DataFrame
df_pairwise['pr(0<y<y+)'] = integrals


# In[26]:


# calculate the elements of odds ratio fomula: w = P_00*P_xy / P_x0*P_0y

df_pairwise['p(x0, y0)_copula']= df_pairwise['c(u0, v0)']*df_pairwise['pr(0<x<x+)']*df_pairwise['pr(0<y<y+)']
df_pairwise['p(x, y0)_copula']= df_pairwise['c(u, v0)']*df_pairwise['pr(i<x<o)']*df_pairwise['pr(0<y<y+)']
df_pairwise['p(x0, y)_copula']= df_pairwise['c(u0, v)']*df_pairwise['pr(0<x<x+)']*df_pairwise['pr(c<y<d)']

# calculate the odds ratio (w):
df_pairwise['w_xy']=(df_pairwise['p(x0, y0)_copula']*df_pairwise['p(i<x<o, c<y<d)_copula'])/(df_pairwise['p(x, y0)_copula']*df_pairwise['p(x0, y)_copula'])


# In[27]:



df_n = df_pairwise.copy()
df_n['pr_x']=df_pairwise['marginals_x'].copy()
df_n['pr_y']=df_pairwise['marginals_y'].copy()


# In[28]:



df_n['n_marginals_x']=df_n['n_sample']*df_n['pr_x']
df_n['n_marginals_y']=df_n['n_sample']*df_n['pr_y']


# In[29]:


df_n['n_marginals_x'] = df_n['n_marginals_x'].apply(lambda x: round(x))
df_n['n_marginals_y'] = df_n['n_marginals_y'].apply(lambda x: round(x))

df_n


# In[30]:


### calculate the binomial coefficients to form the equation of w (Bernoulli copula) and w_xy (Binomial copula)

from scipy.special import loggamma

number = 0
coef_dic = {}
for index, row in df_n.iterrows():
    
    print('index',index)
    
    n = int(row['n_sample'])
    x = int(row['n_marginals_x'])
    y = int(row['n_marginals_y'])


    
    upper = int(min(x,y))
    lower = int(max(x+y-n,0))

    #print('upper', upper)
    #print('k', k)
    df_n.at[index, 'upper'] = upper
    df_n.at[index, 'lower'] = lower
    
    
    
    total_sum = 0
    coef = np.array([])
    
    for k in range(lower, upper + 1):
        
        print('k', k)
        log_o = loggamma(x+1) + loggamma(n-x+1) + loggamma(y+1) + loggamma(n-y+1) - loggamma(k+1) - loggamma(x-k+1) - loggamma(y-k+1) - loggamma(n-x-y+k+1) - loggamma(n+1)
        
        total_sum = math.exp(log_o)

        coef = np.append(coef, total_sum)
    coef_dic[number] = coef
    number = number + 1


# In[31]:


# forming the polynomial equation and solve it using numpy

if (df_n['lower'] > 0).any():
    for index, value in df_n['lower'][df_n['lower'] > 0].items():
        # Get the number of zeros to prepend
        num_zeros = int(value)

        # Modify coef_dic for the specific index
        temppp = coef_dic[index].copy()
        coef_dic[index] = np.concatenate((np.zeros(num_zeros), temppp))

    print("Modified coef_dic")


solutions_list = []
y_list = df_pairwise['w_xy'].values

for i in range(len(coef_dic)):

    coefficients_reverse = coef_dic[i].copy()   ### modified version
    coefficients = coefficients_reverse[::-1]
    y = [-y_list[i]]



    if pd.isnull(y) == False:

        coefficients[-1] = coefficients[-1] + y
        print(np.roots(coefficients))
        solutions = np.roots(coefficients)


        solutions_list.append(solutions)

    else:
        solutions_list.append([])

    


# In[32]:


if np.any(np.isnan(coefficients[::-1])) or np.any(np.isinf(coefficients[::-1])):
    print("Array contains NaN or inf")
else:
    print("Array does not contain NaN or inf")


# In[33]:


# finding the feasible solution among all the available solutions
solutions_list_without_I = []
for i in range(len(solutions_list)):
    length = len(solutions_list[i])
    empty = []
    for j in range(length):
        if "0j" in str(solutions_list[i][j]):
            empty.append(solutions_list[i][j])
        
        elif solutions_list[i][j] > 0 and "j" not in str(solutions_list[i][j]):
            empty.append(solutions_list[i][j])
            
    solutions_list_without_I.append(empty)
    


# In[34]:


filtered_data = [[val for val in inner_list if val > 0] for inner_list in solutions_list_without_I]
filtered_data


# In[35]:


filtered_data = [inner_list[0] if inner_list else None for inner_list in filtered_data]
df = pd.DataFrame({'Positive Values': filtered_data})
df


# In[36]:


df = pd.DataFrame(filtered_data, columns=['Positive Values'])
df_n['w'] = df.copy()
print(df)


# In[37]:


#$$$$$$$$$$$$$$$$$$$$$$$
df_n['w'] = df_n['w'].apply(lambda x: x.real)
df_n['w']


# In[38]:


num_missing_values = df['Positive Values'].isnull().sum()
num_missing_values


# In[39]:


rows_with_missing_values = df.loc[df['Positive Values'].isnull()].index
rows_with_missing_values


# In[40]:


# calculating the P_11 based on estimated Bernoulli copula (w) and the marginals (Geenens paper)
def bivariate_copula(row):
    w = row['w']
    p_x = row['pr_x']
    p_y = row['pr_y']
    
    if w is None:
        return None
    else:
        inside = (((1+ (w-1)*(p_x+p_y))**2)-4*w*(w-1)*p_x*p_y )
        p11 = (1/(2*(w-1)))*(1+((w-1)*(p_x+p_y))- math.sqrt(inside))
        return p11
 


# In[41]:


df_n['P11_final'] = df_n.apply(bivariate_copula, axis=1)


# In[42]:


df_n['P00_final'] = 1 - df_n['pr_x'] -df_n['pr_y'] + df_n['P11_final']   
df_n['P01_final'] = df_n['pr_y'] - df_n['P11_final']
df_n['P10_final'] = df_n['pr_x'] - df_n['P11_final']


# In[43]:


# multi_variate
df_n['P11_final']


# In[44]:


df_pairwise


# In[45]:


#### creating two contingency tables for each row (each jurisdiction)
# observed (estimated from copula)  
df_n['n00_final'] = df_n['n_sample']*df_n['P00_final']
df_n['n01_final'] = df_n['n_sample']*df_n['P01_final'] 
df_n['n10_final'] = df_n['n_sample']*df_n['P10_final']
df_n['n11_final'] = df_n['n_sample']*df_n['P11_final']

df_n['n00_final'] = df_n['n00_final'].apply(lambda x: round(x))
df_n['n01_final'] = df_n['n01_final'].apply(lambda x: round(x))
df_n['n10_final'] = df_n['n10_final'].apply(lambda x: round(x))
df_n['n11_final'] = df_n['n11_final'].apply(lambda x: round(x))


# expected (real joint)
df_n['n00'] = df_n['n_sample']* df_n['n_p00']
df_n['n01'] = df_n['n_sample']* df_n['n_p01']
df_n['n10'] = df_n['n_sample']* df_n['n_p10']
df_n['n11'] = df_n['n_sample']* df_n['n_p11']


df_n['n00'] = df_n['n00'].apply(lambda x: round(x))
df_n['n01'] = df_n['n01'].apply(lambda x: round(x))
df_n['n10'] = df_n['n10'].apply(lambda x: round(x))
df_n['n11'] = df_n['n11'].apply(lambda x: round(x))

# In[46]:


### calcuating the chi-square statistic for each jurisdiction (comparing copula and real)
df_n['k_1'] = ((df_n['n00_final'] - df_n['n00']) ** 2)/df_n['n00']
df_n['k_2'] = ((df_n['n01_final'] - df_n['n01']) ** 2)/df_n['n01']
df_n['k_3'] = ((df_n['n10_final'] - df_n['n10']) ** 2)/df_n['n10']
df_n['k_4'] = ((df_n['n11_final'] - df_n['n11']) ** 2)/df_n['n11']

df_n['chi_2_copula'] = df_n['k_1']+ df_n['k_2']+ df_n['k_3']+df_n['k_4']
df_n['chi_2_copula']


# In[47]:


### Calculating the distribution by assuming dependency
# That is multipling the marginals directly with each other
df_n['n00_ind'] = df_n['n_sample']*(1-df_n['marginals_x'])*(1-df_n['marginals_y']) 
df_n['n01_ind'] = df_n['n_sample']*(1-df_n['marginals_x'])*(df_n['marginals_y']) 
df_n['n10_ind'] = df_n['n_sample']*(df_n['marginals_x'])*(1-df_n['marginals_y']) 
df_n['n11_ind'] = df_n['n_sample']*(df_n['marginals_x'])*(df_n['marginals_y']) 


df_n['n00_ind'] = df_n['n00_ind'].apply(lambda x: round(x))
df_n['n01_ind'] = df_n['n01_ind'].apply(lambda x: round(x))
df_n['n10_ind'] = df_n['n10_ind'].apply(lambda x: round(x))
df_n['n11_ind'] = df_n['n11_ind'].apply(lambda x: round(x))


# In[48]:


### calcuating the chi-square statistic for each jurisdiction (comparing independent and real)
df_n['k_1_ind'] = ((df_n['n00_ind'] - df_n['n00']) ** 2)/df_n['n00']
df_n['k_2_ind'] = ((df_n['n01_ind'] - df_n['n01']) ** 2)/df_n['n01']
df_n['k_3_ind'] = ((df_n['n10_ind'] - df_n['n10']) ** 2)/df_n['n10']
df_n['k_4_ind'] = ((df_n['n11_ind'] - df_n['n11']) ** 2)/df_n['n11']

df_n['chi_2_ind'] = df_n['k_1_ind']+ df_n['k_2_ind']+ df_n['k_3_ind']+df_n['k_4_ind']
df_n['chi_2_ind']


# In[49]:


### Creating the validation graph
### Comparing the chi-square values with the p-value thresholds
import matplotlib.pyplot as plt
import seaborn as sns

# p-value thresholds for 0.05 and 0.1, from the Chi-square table with 3 degree of freedom
p_value_1 = 7.815
p_value_2 = 6.251


# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create scatter plot with colored points
scatter = ax.scatter(df_n['chi_2_copula'], df_n['chi_2_ind'], cmap='viridis')


# Add vertical lines for p-values
ax.axvline(x=p_value_1, linestyle='--', color='red', label='p-value=0.05')
ax.axvline(x=p_value_2, linestyle='--', color='blue', label='p-value=0.1')

# Add horizontal lines for p-values
ax.axhline(y=p_value_1, linestyle='--', color='green', label='p-value=0.05')
ax.axhline(y=p_value_2, linestyle='--', color='orange', label='p-value=0.1')

# Add legend
ax.legend()

# Add labels and title
ax.set_xlabel('Chi-Square (Copula vs. Real)')
ax.set_ylabel('Chi-Square (Independence vs. Real)')
ax.set_title('Fabricated, Comparison of Copula vs. Independence with Real Distributions')

# Show plot
plt.show()





