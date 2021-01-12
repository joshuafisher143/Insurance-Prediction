# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:55:08 2020

@author: joshu
"""
import pandas as pd
import numpy as np



df = pd.read_csv('C:/Users/joshu/Documents/UNCC_Material/Big_Data_Analytics_for_Comp_Adv/Group_projects/Insurance_BIG/spyder_scripts/run2/cleaned_df_noAP.csv',
                 index_col='PolicyNo', low_memory=False)


features= ['Vehicle_Youthful_Good_Student_Code','Vehicle_Driver_Points','Driver_Total_Teenager_Age_15_19','Driver_Total_College_Ages_20_23',
           'Driver_Total_Young_Adult_Ages_24_29','Driver_Total_Low_Middle_Adult_Ages_30_39',
           'Driver_Total_Middle_Adult_Ages_40_49','Driver_Total_Adult_Ages_50_64','Driver_Total_Senior_Ages_65_69',
           'Driver_Total_Upper_Senior_Ages_70_plus','Vehicle_Usage','Vehicle_Miles_To_Work','Vehicle_Territory','Annual_Premium','Loss_Amount',
           'Driver_Total_Male', 'Driver_Total_Female', 'Driver_Total_Single', 'Driver_Total_Married', 'Vehicle_Performance', 'Vehicle_Age_In_Years']

df = df[features]

df_dum = pd.get_dummies(df)

df_dum = pd.get_dummies(df_dum, columns=['Vehicle_Territory'])

df_dum.columns = df_dum.columns.str.replace(' ','')


df_noclaim = df_dum[df_dum['Loss_Amount'] == 0]
df_claim = df_dum[df_dum['Loss_Amount']> 0]        

def make_sample(size, pct):
    sample_df = pd.DataFrame(columns=df_dum.columns)
    sample_df.index.names =['PolicyNo']
    noclaim = df_noclaim.sample(n=int((1-pct)*size))
    claim = df_claim.sample(n=int(pct*size))
    sample_df = sample_df.append(noclaim)
    sample_df = sample_df.append(claim)
    
    
    agg_training_df = pd.DataFrame(columns=df_dum.columns)
    
    #features to take the mean of. Other columns will take median
    total_cols = ['Annual_Premium','Vehicle_Youthful_Good_Student_Code_EligibleforGoodStudentCredit',
                  'Vehicle_Youthful_Good_Student_Code_NotEligibleforGoodStudentCredit',
                  'Vehicle_Usage_Business','Vehicle_Usage_Farm','Vehicle_Usage_Pleasure','Vehicle_Usage_Work', 'Loss_Amount',
                  'Vehicle_Performance_High','Vehicle_Performance_Intermediate','Vehicle_Performance_Sports',
                  'Vehicle_Performance_SportsPremium','Vehicle_Performance_Standard', 'Vehicle_Age_In_Years']
    
    prop_cols = [prop_col for prop_col in df_dum if prop_col.startswith('Vehicle_Territory')]
    
    for col in sample_df.columns:
        if col in total_cols:
            sample_df[col] = sample_df[col].sum()
        if col in prop_cols:
            sample_df[col] = (sample_df[col].value_counts().loc[0])/len(sample_df)
        else:
            sample_df[col] = sample_df[col].mean()
            
    agg_training_df = agg_training_df.append(sample_df.iloc[0])
            
    return agg_training_df

final_training_df = pd.DataFrame(columns=df_dum.columns)
num = 0
while num < 30000:
    print(num)
    row = make_sample(np.random.choice(np.arange(1000,6000,2000),1)[0], np.random.choice(np.arange(0.01,0.21, 0.01),1)[0])
    final_training_df = final_training_df.append(row)
    num +=1

final_training_df.to_csv('C:/Users/joshu/Documents/UNCC_Material/Big_Data_Analytics_for_Comp_Adv/Group_projects/Insurance_BIG/spyder_scripts/run2/added_cols_df_noPC.csv')





