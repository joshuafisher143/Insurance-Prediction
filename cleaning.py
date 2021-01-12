# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 18:55:26 2020

@author: joshu
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:56:55 2020

@author: joshu
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


df = pd.read_csv('C:/Users/joshu/Documents/UNCC_Material/Big_Data_Analytics_for_Comp_Adv/Group_projects/Insurance_BIG/training_data.csv', index_col='PolicyNo', low_memory=False)


df1 = df.drop(columns=['Vehicle_New_Cost_Amount', 'Vehicle_Number_Of_Drivers_Assigned',
                       'Vehicle_Days_Per_Week_Driven',
                       'Vehicle_Annual_Miles', 'EEA_PolicyYear',
                       'Vehicle_Comprehensive_Coverage_Limit', 'Policy_Zip_Code_Garaging_Location',
                       'Severity', 'Claim_Count','Driver_Total',
                       'Driver_Total_Related_To_Insured_Self','Driver_Total_Related_To_Insured_Spouse',
                       'Driver_Total_Related_To_Insured_Child','Driver_Total_Licensed_In_State','Frequency', 'Loss_Ratio'])

#only 5 unknown devices
df1 = df1[df1['Vehicle_Anti_Theft_Device'] != 'Unknown']
#only 81 unknown 
df1 = df1[df1['Vehicle_Passive_Restraint'] != 'Unknown']
#Only a few unknown symbols
df1 = df1[df1['Vehicle_Symbol'] > 0]


#convert -1 to np.nan
df1 = df1.replace(-1,np.nan)

# remove numerical outliers
x = df1[['Vehicle_Physical_Damage_Limit', 'EEA_Policy_Tenure',
        'Annual_Premium']]
x_stats = x.describe()
x_stats.loc['IQR'] = x_stats.loc['75%']-x_stats.loc['25%']
x_stats.loc['lower_fence'] = x_stats.loc['25%'] - 1.5*x_stats.loc['IQR']
x_stats.loc['upper_fence'] = x_stats.loc['75%'] + 1.5*x_stats.loc['IQR']

#remove upper fence from vehicle physical damage limit
df1 = df1[df1['Vehicle_Physical_Damage_Limit'] < x_stats['Vehicle_Physical_Damage_Limit'].loc['upper_fence']]
#remove upper fence from EEA policy tenure
df1 = df1[df1['EEA_Policy_Tenure'] < x_stats['EEA_Policy_Tenure'].loc['upper_fence']]
#remove both upper and lower fence from annual premium
df1 = df1[df1['Annual_Premium'] < x_stats['Annual_Premium'].loc['upper_fence']]
df1 = df1[df1['Annual_Premium'] > x_stats['Annual_Premium'].loc['lower_fence']]

#remove outliers from Loss Amount and Severity
y = df1[df1['Loss_Amount']>0]
y_stats = y['Loss_Amount'].describe()
y_stats.loc['IQR'] = y_stats.loc['75%']-y_stats.loc['25%']
y_stats.loc['lower_fence'] = y_stats.loc['25%'] - 1.5*y_stats.loc['IQR']
y_stats.loc['upper_fence'] = y_stats.loc['75%'] + 1.5*y_stats.loc['IQR']

df1 = df1[df1['Loss_Amount'] < y_stats.loc['upper_fence']]

freq_over_zero = df1[df1['Loss_Amount']>0]

#3 columns have a signficant number of np.nan values
# Vehicle miles to work, vehicle med pay limit, vehicle collision coverage deductible
# replace missing values with most frequent value
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df2 = imp.fit_transform(df1)

#imputer converts df to a numpy array. Convert back to pandas df and add original indexes and columns
df2_imp = pd.DataFrame(df2, columns=df1.columns, index=df1.index)

