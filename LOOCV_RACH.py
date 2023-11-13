#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:48:07 2022

@author: Bailejor
"""

import pandas as pd 
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report, Model, Minimizer
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error, median_absolute_error
from sklearn.model_selection import LeaveOneOut, train_test_split, TimeSeriesSplit, KFold, ShuffleSplit, ParameterGrid
import math
import random
from scipy import optimize
import warnings

warnings.filterwarnings('ignore')
np.seterr(all="ignore")


#####CHANGE DATASET HERE#############
df = pd.read_csv('Rachlin_sim.csv', header = None)

loo = LeaveOneOut()



model_best_params = {}





def disc_rachlin(p0, x_train, y_train):
    kval = p0[0]
    sval = p0[1]
    model = 1.0 / (1 + np.power((kval * x_train), sval))
    return y_train - model
    

def disc_myerson(p0, x_train, y_train):
    kval = p0[0]
    sval = p0[1]
    model = 1.0 / np.power((1 + kval * x_train), sval)
    return y_train - model

def disc_hyperbolic(p0, x_train, y_train):
    kval = p0[0]
    model = 1.0 / (1 + kval * x_train)
    return y_train - model

def disc_expo(p0, x_train, y_train):
    kval = p0[0]
    model = 1 * np.exp(-kval * x_train)
    return y_train - model    
    

def disc_quasi(p0, x_train, y_train):
    bval = p0[0]
    dval = p0[1]
    model = 1.0 * bval * np.exp(-dval * x_train)
    return y_train - model




def plot_rachlin(x_train, kval, sval):
    return 1.0/(1+ np.power((kval * x_train), sval))

def plot_myerson(x_train, kval, sval):
    return 1.0/np.power((1 + kval * x_train), sval)

def plot_hyper(x_train, kval):
    return 1.0/(1 + kval * x_train)

def plot_expo(x_train, kval):
    return 1.0 * np.exp(-kval * x_train)

def plot_quasi(x_train, bval, dval):
    return 1.0 * bval * np.exp(-dval * x_train)
  
low_density = [2, 5, 8, 12]
mid_density = [2, 3, 4, 5, 8, 9, 10, 12]
high_density = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
density_list = [high_density]

jumbo_list_bic = []
jumbo_list_aic = []

#Grid for parameter starting value search
s_list = np.linspace(0.01, 10, num=1000).tolist()
random_sample_s_list = random.sample(s_list, 10)
#k_list = np.linspace(6.1*(10**(-6)), 6.0*(10**(4)), num = 25).tolist()
#random_sample_k_list = random.sample(k_list, 25)



jor = 0
#THIS SECTION FOR LOOCV
low_density = [2, 5, 8, 12]
mid_density = [2, 3, 4, 5, 8, 9, 10, 12]
high_density = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
density_list = [high_density]

rach_final_score = []
myer_final_score = []
hyper_final_score = []
expo_final_score = []
noise_final_score = []
quasi_final_score = []
noise_final_score = []

jumbo_list = [[]]
end_list = []


rach_k = np.linspace(start=-12, stop=12, num = 25)
rach_k = np.exp(rach_k)
rach_s = np.linspace(start=0.01, stop=10, num=25)
params_rach =[{'k': rach_k,
         's': rach_s}]

params_single = [{'k': rach_k}]


bd_b = np.linspace(start=np.exp(-12), stop=1, num = 25)
bd_d = np.linspace(start=np.exp(-12), stop=1, num=25)
params_bd = [{'b': bd_b,
         'd': bd_d}]

for density in density_list:
    #random_sample = random.sample(range(10000), 100)
    random_sample = list(range(1, 1001))

    count_keeper = 0
    for i in random_sample:

        count_keeper = count_keeper + 1
        score_comparison = []
        upper_range = i + 2
        lower_range = upper_range - 1
    
        x = df[0:1]
        y = df[lower_range:upper_range]


####################CHANGE DENSITY HERE###########################################
        #Low density [0, 2, 3, 6, 8, 11]
        #Mid density [0, 2, 4, 5, 6, 7, 8, 9, 11, 13]
        #High density [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        candidate = pd.DataFrame(index = [0, 2, 4, 6, 8, 11])
        candidate['x'] = x.T
        candidate['y'] = y.T
        #candidate = candidate.iloc[density].copy().reset_index()

        x_data = candidate['x']
        y_data = candidate['y']

        x_data = x_data.to_numpy().flatten()
        y_data = y_data.to_numpy().flatten()

        loo.get_n_splits(x_data)
    
   

        rach_row = []
        myer_row = []
        hyper_row = []
        expo_row = []
        bd_row = []
        noise_row = []
    
        for train_index, test_index in loo.split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]


            rach_counter = 0
            rach_scores = []
            rach_params = []


            myer_scores = []
            myer_params = []


            hyper_scores = []
            hyper_params = []

            expo_scores = []
            expo_params = []



            bd_scores = []
            bd_params = []


        ##########################RACHLIN LOOCV##############################################################################################

            for g in ParameterGrid(params_rach):
                p0 = list(g.values())
                rachlinResult = optimize.least_squares(disc_rachlin, p0, method ='lm', args=(x_train, y_train))
                rach_preds = plot_rachlin(x_train, rachlinResult.x[0], rachlinResult.x[1])
                if np.isfinite(rach_preds).all():
                    rach_params.append(rachlinResult.x)
                    try:
                        rach_preds = mean_absolute_error(y_train, rach_preds)
                        rach_scores.append(rach_preds)
                    except:
                        pass
                else:
                    pass


            for g in ParameterGrid(params_rach):
                p0 = list(g.values())
                myerResult = optimize.least_squares(disc_myerson, p0, method ='lm', args=(x_train, y_train))
                myer_preds = plot_myerson(x_train, myerResult.x[0], myerResult.x[1])
                if np.isfinite(myer_preds).all():
                    myer_params.append(myerResult.x)
                    try:
                        myer_preds = mean_absolute_error(y_train, myer_preds)
                        myer_scores.append(myer_preds)
                    except:
                        pass
                else:
                    pass

            for g in ParameterGrid(params_single):
                p0 = list(g.values())
                expoResult = optimize.least_squares(disc_expo, p0, method ='lm', args=(x_train, y_train))
                expo_preds = plot_expo(x_train, expoResult.x[0])
                if np.isfinite(expo_preds).all():
                    expo_params.append(expoResult.x)
                    try:
                        expo_preds = mean_absolute_error(y_train, expo_preds)
                        expo_scores.append(expo_preds)
                    except:
                        pass
                else:
                    pass

            for g in ParameterGrid(params_single):
                p0 = list(g.values())
                hyperResult = optimize.least_squares(disc_hyperbolic, p0, method ='lm', args=(x_train, y_train))
                hyper_preds = plot_hyper(x_train, hyperResult.x[0])
                if np.isfinite(hyper_preds).all():
                    hyper_params.append(hyperResult.x)
                    try:
                        hyper_preds = mean_absolute_error(y_train, hyper_preds)
                        hyper_scores.append(hyper_preds)
                    except:
                        pass
                else:
                    pass

            for g in ParameterGrid(params_bd):
                p0 = list(g.values())
                bdResult=optimize.least_squares(disc_quasi, p0, method ='trf', bounds = (0, 1), args=(x_train, y_train))
                bd_preds = plot_quasi(x_train, bdResult.x[0], bdResult.x[1])
                if np.isfinite(bd_preds).all():
                    bd_params.append(bdResult.x)
                    try:
                        bd_preds = mean_absolute_error(y_train, bd_preds)
                        bd_scores.append(bd_preds)
                    except:
                        pass
                else:
                    pass








            rach_total = np.nanargmin(rach_scores)
            rach_test = rach_params[rach_total]
            rach_total = plot_rachlin(x_test, rach_test[0], rach_test[1])
            rach_assess = mean_absolute_error(y_test, rach_total)
            rach_row.append(rach_assess)

            myer_total = np.nanargmin(myer_scores)
            myer_test = myer_params[myer_total]
            myer_plot = plot_myerson(x_test, myer_test[0], myer_test[1])
            try:
                myer_assess = mean_absolute_error(y_test, myer_plot)
                myer_row.append(myer_assess)
            except:
                pass

            expo_total = np.nanargmin(expo_scores)
            expo_test = expo_params[expo_total]
            expo_plot = plot_expo(x_test, expo_test[0])
            expo_assess = mean_absolute_error(y_test, expo_plot)
            expo_row.append(expo_assess)

            hyper_total = np.nanargmin(hyper_scores)
            hyper_test = hyper_params[hyper_total]
            hyper_plot = plot_hyper(x_test, hyper_test[0])
            hyper_assess = mean_absolute_error(y_test, hyper_plot)
            hyper_row.append(hyper_assess)

            bd_total = np.nanargmin(bd_scores)
            bd_test = bd_params[bd_total]
            bd_plot = plot_quasi(x_test, bd_test[0], bd_test[1])
            bd_assess = mean_absolute_error(y_test, bd_plot)
            bd_row.append(bd_assess)

            noise = np.array(sum(y_train/len(y_train)))
            noise = noise.reshape(1,)
            noise_score = mean_absolute_error(y_test, noise)
            noise_row.append(noise_score)




        rach_final_score.append(sum(rach_row)/len(rach_row))
        myer_final_score.append(sum(myer_row)/len(myer_row))
        expo_final_score.append(sum(expo_row)/len(expo_row))
        hyper_final_score.append(sum(hyper_row)/len(hyper_row))
        quasi_final_score.append(sum(bd_row)/len(bd_row))
        noise_final_score.append(sum(noise_row)/len(noise_row))

        jor = jor + 1
        print("Row number: " + str(jor))
        
        


#print(sum(rach_final_score)/len(rach_final_score))
#print(sum(myer_final_score)/len(myer_final_score))
#print(sum(hyper_final_score)/len(hyper_final_score))
#print(sum(expo_final_score)/len(expo_final_score))
#print(sum(quasi_final_score)/len(quasi_final_score))
#print(sum(noise_final_score)/len(noise_final_score))


jor = pd.DataFrame(list(zip(rach_final_score, myer_final_score, hyper_final_score, expo_final_score, noise_final_score, quasi_final_score)),
               columns =['Rachlin', 'Myer', 'Hyper', 'Expo', 'Noise', 'BD'])
        
jor.to_csv('RACH_LOOCV_LOW.csv')
        ##########################RACHLIN LOOCV##############################################################################################


        

        
