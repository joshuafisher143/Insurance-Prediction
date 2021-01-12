# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:26:51 2020

@author: joshu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import glob
import pickle

pkl_fname = 'C:/Users/joshu/Documents/run2/models2/rf_model2.pkl'

with open(pkl_fname, 'rb') as file:
    model = pickle.load(file)

final_results_df = pd.DataFrame(columns=['ID', 'ln_LR'])

for fname in glob.glob('C:/Users/joshu/Documents/Insurance_BIG/testing_portfolios/*.csv'):
    df = pd.read_csv(fname, index_col='PolicyNo')
    file_name = os.path.basename(fname)
    no_csv = os.path.splitext(file_name)[0]



    df = df.drop(columns=['Vehicle_New_Cost_Amount', 'Vehicle_Number_Of_Drivers_Assigned',
                       'Vehicle_Days_Per_Week_Driven',
                       'Vehicle_Annual_Miles', 'EEA_PolicyYear',
                       'Vehicle_Comprehensive_Coverage_Limit', 'Policy_Zip_Code_Garaging_Location',
                       'Driver_Total', 'Driver_Total_Male', 'Driver_Total_Female',
                       'Driver_Total_Single', 'Driver_Total_Married','Driver_Total_Related_To_Insured_Self','Driver_Total_Related_To_Insured_Spouse',
                       'Driver_Total_Related_To_Insured_Child','Driver_Total_Licensed_In_State'])

    
    df['Vehicle_Bodily_Injury_Limit'] = df['Vehicle_Bodily_Injury_Limit'].astype(str)
    df['EEA_Prior_Bodily_Injury_Limit'] = df['EEA_Prior_Bodily_Injury_Limit'].astype(str)
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype != np.number:
            df[col] = le.fit_transform(df[col])
    


    test_pred = model.predict(df)
    
    ln_LR = np.log(sum(test_pred)/sum(df['Annual_Premium']))
    
    final_results_df = final_results_df.append({'ID':no_csv, 'ln_LR':ln_LR}, ignore_index=True)
    