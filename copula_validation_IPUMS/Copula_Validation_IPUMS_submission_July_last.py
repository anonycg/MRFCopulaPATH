#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries
import pandas as pd
import scipy.stats
#import scipy.stats as stats
import scipy.special as sp
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm


# In[3]:


dtype_specification = {'FIPS': str}

### importing data
df = pd.DataFrame(pd.read_excel(r"C:\Users\IPUMS_validation.xlsx", dtype=dtype_specification))
df_pairwise = df[['marginal_poverty','marginal_employment', 'FIPS']].copy()
df_pairwise = df_pairwise.rename(columns={'marginal_poverty': 'marginal_var1', 'marginal_employment': 'marginal_var2'})


df_pairwise


# In[4]:


# multiplying by a constant as number of sample in each jurisdiction
df_pairwise['n_sample'] = 1000 
x_values = df_pairwise['n_sample']*df_pairwise['marginal_var1']  # Extract X values from the samples
y_values = df_pairwise['n_sample']*df_pairwise['marginal_var2']  # Extract Y values from the samples

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



# Define the bounds for the truncated normal distribution
lower_bound_x, upper_bound_x = 0, x_values.max()+100
lower_bound_y, upper_bound_y = 0, y_values.max()+100

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


# In[7]:


# calculate the transformed unifrom values for x=0.5 and y=0.5 as an approximation of x=0 and y=0
u0 = truncnorm.cdf(0.5, a=a_x, b=b_x, loc=mean_x, scale=std_x)
v0 = truncnorm.cdf(0.5, a=a_y, b=b_y, loc=mean_y, scale=std_y)


# In[8]:


plt.subplot(1, 2, 2)
plt.hist(u, bins=30, density=True, alpha=0.7, color='green')
plt.title('Transformed Distribution (Uniform(0,1))')
plt.xlabel('u')
plt.ylabel('Density')

plt.tight_layout()
plt.show()


# In[9]:


plt.subplot(1, 2, 2)
plt.hist(v, bins=30, density=True, alpha=0.7, color='green')
plt.title('Transformed Distribution (Uniform(0,1))')
plt.xlabel('v')
plt.ylabel('Density')

plt.tight_layout()
plt.show()


# In[10]:


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

############## the correlation coefficient based on all jurisdictions
ro, p = scipy.stats.pearsonr(x_values,y_values)
ro


# In[11]:


def Copula_Gaussian(row):
    a = row['a']
    b = row['b']
    copula = (1/(math.sqrt(1 - ro**2)))*(math.exp(-(((a**2 + b**2) * ro**2 - 2*a*b*ro)/(2*(1-ro**2)))))

    # using np. instead of math. 
    #copula = (1/(np.sqrt(1 - ro**2)))*(np.exp(-(((a**2 + b**2) * ro**2 - 2*a*b*ro)/(2*(1-ro**2)))))

    return copula


############## calculating the Copula for each marginal data points 
df_pairwise['c(u, v)'] = df_pairwise.apply(Copula_Gaussian, axis = 1)


# In[12]:


# Create positive and negative offsets
positive_x = x_values + 0.5
negative_x = x_values - 0.5

# Create new columns in the DataFrame
df_pairwise['i'] = negative_x
df_pairwise['o'] = positive_x


# In[13]:


# Create positive and negative offsets
positive_y = y_values + 0.5
negative_y = y_values - 0.5

# Create new columns in the DataFrame
df_pairwise['c'] = negative_y
df_pairwise['d'] = positive_y


# In[14]:



# Iterate over the rows and calculate the probablity for each row: F_x(x+0.5)-F_x(x-0.5)
integrals = []
for index, row in df_pairwise.iterrows():
    lower_limit = row['i']
    upper_limit = row['o']
    integral = truncnorm.cdf(upper_limit, a=a_x, b=b_x, loc=mean_x, scale=std_x) - truncnorm.cdf(lower_limit, a=a_x, b=b_x, loc=mean_x, scale=std_x)

    integrals.append(integral)

# Add the calculated integrals as a new column in the DataFrame
df_pairwise['pr(i<x<o)'] = integrals


# In[15]:


df_pairwise['pr(i<x<o)']


# In[16]:



# Iterate over the rows and calculate the probablity for each row: F_y(y+0.5)-F_y(y-0.5)
integrals = []
for index, row in df_pairwise.iterrows():
    lower_limit = row['c']
    upper_limit = row['d']
    integral = truncnorm.cdf(upper_limit, a=a_y, b=b_y, loc=mean_y, scale=std_y) - truncnorm.cdf(lower_limit, a=a_y, b=b_y, loc=mean_y, scale=std_y)
    integrals.append(integral)

# Add the calculated integrals as a new column in the DataFrame
df_pairwise['pr(c<y<d)'] = integrals


# In[17]:


# calculate the P(X=x, Y=y) 
df_pairwise['p(i<x<o, c<y<d)_copula']= df_pairwise['c(u, v)']*df_pairwise['pr(i<x<o)']*df_pairwise['pr(c<y<d)']


# In[18]:


df_pairwise['p(i<x<o, c<y<d)_copula'] 


# In[19]:


# calculate the Gaussian copula density function: c(u0,v0)

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


# calculate the Gaussian copula density function: c(u0,v)


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


# calculate the Gaussian copula density function: c(u,v0)

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


df_pairwise['w_xy']


# In[28]:


df_n = df_pairwise.copy()
df_n['pr_x']=df_pairwise['marginal_var1'].copy()
df_n['pr_y']=df_pairwise['marginal_var2'].copy()


# In[29]:


df_n['n_marginal_var1']=df_n['n_sample']*df_n['pr_x']
df_n['n_marginal_var2']=df_n['n_sample']*df_n['pr_y']


# In[30]:



df_n['n_marginal_var1'] = df_n['n_marginal_var1'].apply(lambda x: round(x))
df_n['n_marginal_var2'] = df_n['n_marginal_var2'].apply(lambda x: round(x))
df_n


# In[ ]:





# In[31]:


### calculate the binomial coefficients to form the equation of w (Bernoulli copula) and w_xy (Binomial copula)

from scipy.special import loggamma

number = 0
coef_dic = {}
for index, row in df_n.iterrows():
    
    #print('index',index)
    
    n = int(row['n_sample'])
    x = int(row['n_marginal_var1'])
    y = int(row['n_marginal_var2'])


    
    upper = int(min(x,y))
    lower = int(max(x+y-n,0))

    #print('upper', upper)
    #print('k', k)
    df_n.at[index, 'upper'] = upper
    df_n.at[index, 'lower'] = lower
    
    
    
    total_sum = 0
    coef = np.array([])
    
    for k in range(lower, upper + 1):
        
        #print('k', k)
        log_o = loggamma(x+1) + loggamma(n-x+1) + loggamma(y+1) + loggamma(n-y+1) - loggamma(k+1) - loggamma(x-k+1) - loggamma(y-k+1) - loggamma(n-x-y+k+1) - loggamma(n+1)
        total_sum = math.exp(log_o)
        coef = np.append(coef, total_sum)
        
        
    coef_dic[number] = coef
    number = number + 1


# In[32]:


if (df_n['lower'] > 0).any():
    for index, value in df_n['lower'][df_n['lower'] > 0].items():
        # Get the number of zeros to prepend
        num_zeros = int(value)

        # Modify coef_dic for the specific index
        temppp = coef_dic[index].copy()
        coef_dic[index] = np.concatenate((np.zeros(num_zeros), temppp))

    print("Modified coef_dic")


# In[33]:


# forming the polynomial equation and solve it using numpy

solutions_list = []
y_list = df_pairwise['w_xy'].values 


solutions_list = []
for i in range(len(coef_dic)):

    #print('i',i)
    #print('coef_dic[i]; ', coef_dic[i])
    coefficients_reverse = coef_dic[i].copy()   ### modified version
    #print('coefficients_reverse', coefficients_reverse)
    coefficients = coefficients_reverse[::-1]
    #print('coefficients', coefficients)
    #print('w_xy_', df_pairwise['w_xy'][i]) 
    y = [-y_list[i]]
    #print('y', y)


    if pd.isnull(y) == False:

        #print('c0:', coefficients[-1])
        coefficients[-1] = coefficients[-1] + y
        #print('c0-w_xy:', coefficients[-1])
        #print('number= ', i)
        print(np.roots(coefficients))
        solutions = np.roots(coefficients)


        solutions_list.append(solutions)

    else:
        solutions_list.append([])


# In[34]:


if np.any(np.isnan(coefficients[::-1])) or np.any(np.isinf(coefficients[::-1])):
    print("Array contains NaN or inf")
else:
    print("Array does not contain NaN or inf")


# In[35]:


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
    


# In[36]:


filtered_data = [[val for val in inner_list if val > 0] for inner_list in solutions_list_without_I]
filtered_data


# In[37]:


filtered_data = [inner_list[0] if inner_list else None for inner_list in filtered_data]
df = pd.DataFrame({'Positive Values': filtered_data})
df


# In[38]:


df = pd.DataFrame(filtered_data, columns=['Positive Values'])
df_n['w'] = df.copy()
print(df)


# In[39]:


#$$$$$$$$$$$$$$$$$$$$$$$
df_n['w'] = df_n['w'].apply(lambda x: x.real)
df_n['w']


# In[40]:


num_missing_values = df['Positive Values'].isnull().sum()
num_missing_values


# In[41]:


rows_with_missing_values = df.loc[df['Positive Values'].isnull()].index
rows_with_missing_values


# In[42]:


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
 


# In[43]:


df_n['P11_final'] = df_n.apply(bivariate_copula, axis=1)


# In[44]:


df_n['P00_final'] = 1 - df_n['pr_x'] -df_n['pr_y'] + df_n['P11_final']   
df_n['P01_final'] = df_n['pr_y'] - df_n['P11_final']
df_n['P10_final'] = df_n['pr_x'] - df_n['P11_final']


# In[ ]:


$$$$$$$$$$$$$$$$


# In[66]:




df1 = pd.read_excel(r"C:\Users\akhos\Downloads\IPUMS_validation.xlsx", sheet_name='real', dtype={'FIPS': str})
df2 = pd.read_excel(r"C:\Users\akhos\Downloads\IPUMS_validation.xlsx", sheet_name='copula', dtype={'FIPS': str})


# Merge DataFrames based on 'FIPS'
merged_df = pd.merge(df1, df2, on='FIPS', how='left')
#merged_df = merged_df.dropna(subset=['w'])


# In[68]:


#### creating two contingency tables for each row (each jurisdiction)
# observed (estimated from copula)  
merged_df['n00_final'] = merged_df['n_sample']*merged_df['P00_final']
merged_df['n01_final'] = merged_df['n_sample']*merged_df['P01_final'] 
merged_df['n10_final'] = merged_df['n_sample']*merged_df['P10_final']
merged_df['n11_final'] = merged_df['n_sample']*merged_df['P11_final']

merged_df['n00_final'] = merged_df['n00_final'].apply(lambda x: round(x))
merged_df['n01_final'] = merged_df['n01_final'].apply(lambda x: round(x))
merged_df['n10_final'] = merged_df['n10_final'].apply(lambda x: round(x))
merged_df['n11_final'] = merged_df['n11_final'].apply(lambda x: round(x))


# expected (real joint)
merged_df['n00'] = merged_df['n_sample']* merged_df['Pr00']
merged_df['n01'] = merged_df['n_sample']* merged_df['Pr01']
merged_df['n10'] = merged_df['n_sample']* merged_df['Pr10']
merged_df['n11'] = merged_df['n_sample']* merged_df['Pr11']



# expected (real joint)
merged_df['n00'] = merged_df['n00'].apply(lambda x: round(x))
merged_df['n01'] = merged_df['n01'].apply(lambda x: round(x))
merged_df['n10'] = merged_df['n10'].apply(lambda x: round(x))
merged_df['n11'] = merged_df['n11'].apply(lambda x: round(x))


# In[69]:


### calcuating the chi-square statistic for each jurisdiction (comparing copula and real)
merged_df['k_1'] = ((merged_df['n00_final'] - merged_df['n00']) ** 2)/merged_df['n00']
merged_df['k_2'] = ((merged_df['n01_final'] - merged_df['n01']) ** 2)/merged_df['n01']
merged_df['k_3'] = ((merged_df['n10_final'] - merged_df['n10']) ** 2)/merged_df['n10']
merged_df['k_4'] = ((merged_df['n11_final'] - merged_df['n11']) ** 2)/merged_df['n11']

merged_df['chi_2_copula'] = merged_df['k_1']+ merged_df['k_2']+ merged_df['k_3']+merged_df['k_4']
merged_df['chi_2_copula']


# In[72]:



### Calculating the distribution by assuming dependency
# That is multipling the marginals directly with each other
merged_df['n00_ind'] = merged_df['n_sample']*(1-merged_df['marginal_var1'])*(1-merged_df['marginal_var2']) 
merged_df['n01_ind'] = merged_df['n_sample']*(1-merged_df['marginal_var1'])*(merged_df['marginal_var2']) 
merged_df['n10_ind'] = merged_df['n_sample']*(merged_df['marginal_var1'])*(1-merged_df['marginal_var2']) 
merged_df['n11_ind'] = merged_df['n_sample']*(merged_df['marginal_var1'])*(merged_df['marginal_var2']) 


merged_df['n00_ind'] = merged_df['n00_ind'].apply(lambda x: round(x))
merged_df['n01_ind'] = merged_df['n01_ind'].apply(lambda x: round(x))
merged_df['n10_ind'] = merged_df['n10_ind'].apply(lambda x: round(x))
merged_df['n11_ind'] = merged_df['n11_ind'].apply(lambda x: round(x))


# In[73]:


### calcuating the chi-square statistic for each jurisdiction (comparing independent and real)
merged_df['k_1_ind'] = ((merged_df['n00_ind'] - merged_df['n00']) ** 2)/merged_df['n00']
merged_df['k_2_ind'] = ((merged_df['n01_ind'] - merged_df['n01']) ** 2)/merged_df['n01']
merged_df['k_3_ind'] = ((merged_df['n10_ind'] - merged_df['n10']) ** 2)/merged_df['n10']
merged_df['k_4_ind'] = ((merged_df['n11_ind'] - merged_df['n11']) ** 2)/merged_df['n11']

merged_df['chi_2_ind'] = merged_df['k_1_ind']+ merged_df['k_2_ind']+ merged_df['k_3_ind']+merged_df['k_4_ind']
merged_df['chi_2_ind']


# In[74]:


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
scatter = ax.scatter(merged_df['chi_2_copula'], merged_df['chi_2_ind'], cmap='viridis')


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


# In[75]:


merged_df.to_excel('IPUMS_validation.xlsx', index=False)


# In[ ]:




