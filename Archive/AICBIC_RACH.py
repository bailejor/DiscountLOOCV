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


'''In order to run select the dataset, then the density'''

#####CHANGE DATASET HERE#############
df = pd.read_csv('Rachlin_sim.csv', header = None)

loo = LeaveOneOut()
#loo = TimeSeriesSplit(n_splits=4, test_size=1)
#loo = KFold(n_splits = 2)


#Convert format to long form

random.seed(4251988)

model_best_params = {}


#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle = False)


def disc_rachlin(p0, x_data, y_data):
    kval = p0[0]
    sval = p0[1]
    model = 1.0 / (1 + np.power((kval * x_data), sval))
    return y_data - model

def disc_rachlin2(rachlinParams, x_train, y_train):
    parvals = rachlinParams.valuesdict()
    kval = parvals['kval']
    sval = parvals['sval']
    model = 1.0 / (1 + np.power((kval * x_train), sval))  
    return y_train - model

def disc_hyperbolic2(hyperParams, x_data, y_data):
    parvals = hyperParams.valuesdict()
    kval = parvals['kval']
    model = 1.0 / (1 + kval * x_data)
    return y_data - model

def disc_expo2(expoParams, x_data, y_data):
    parvals = expoParams.valuesdict()
    kval = parvals['kval']
    model = 1 * np.exp(-kval * x_data)
    return y_data - model   
    

def disc_myerson(p0, x_train, y_train):
    kval = p0[0]
    sval = p0[1]
    model = 1.0 / np.power((1 + kval * x_train), sval)
    return y_train - model


def disc_myerson2(myersonParams, x_data, y_data):
    parvals = myersonParams.valuesdict()
    kval = parvals['kval']
    sval = parvals['sval']
    model = 1.0 / np.power((1 + kval * x_data), sval)
    return y_data - model

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

def disc_quasi2(quasiParams, x_data, y_data):
    parvals = quasiParams.valuesdict()
    bval = parvals['bval']
    dval = parvals['dval']
    model = 1.0 * bval * np.exp(-dval * x_data)
    return y_data - model

def disc_noise(noise_Params, x_train, y_train):
    parvals = noiseParams.valuesdict()
    model = sum(y_train)/len(y_train)
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
  





#AIC + 2*2*(2 + 1) / len(density) - 2 - 1
jor = 0
#THIS SECTION FOR LOOCV
low_density = [2, 5, 8, 12]
mid_density = [2, 3, 4, 5, 8, 9, 10, 12]
high_density = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
density_list = [high_density]

rach_aicc_final = []
rach_bic_final = []
rach_r2_final = []

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


single_k = np.linspace(start=-12, stop=12, num = 25)
single_k = np.exp(single_k)
params_single = [{'k': single_k}]


bd_b = np.linspace(start=np.exp(-12), stop=1, num = 25)
bd_d = np.linspace(start=np.exp(-12), stop=1, num=25)
params_bd = [{'b': bd_b,
         'd': bd_d}]


rach_row_aicc = []
rach_row_bic = []
rach_row_r2 = []

myer_row_aicc = []
myer_row_bic = []
myer_row_r2 = []

hyper_row_aicc = []
hyper_row_bic = []
hyper_row_r2 = []

expo_row_aicc = []
expo_row_bic = []
expo_row_r2 = []

bd_row_aicc = []
bd_row_bic = []
bd_row_r2 = []

noise_row_aicc = []
noise_row_bic = []
noise_row_r2 = []

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
        #Low density [0, 2, 3, 6, 8, 11]
        #Mid density [0, 2, 4, 5, 6, 7, 8, 9, 11, 13]
        #High density [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        #add to single df
        candidate = pd.DataFrame(index = [0, 2, 4, 6, 8, 11])
        candidate['x'] = x.T
        candidate['y'] = y.T
        #candidate = candidate.iloc[density].copy().reset_index()

        x_data = candidate['x']
        y_data = candidate['y']

        x_data = x_data.to_numpy().flatten()
        y_data = y_data.to_numpy().flatten()



        rach_counter = 0
        rach_scores = []
        rach_params = []
        rach_aicc_arr = []
        rach_bic_arr = []
        rach_r2_arr = []



        myer_scores = []
        myer_aicc_arr = []
        myer_bic_arr = []
        myer_r2_arr = []
        myer_params = []
        myer_row = []

        hyper_scores = []
        hyper_aicc_arr = []
        hyper_bic_arr = []
        hyper_r2_arr = []
        hyper_params = []
        hyper_row = []


        expo_scores = []
        expo_aicc_arr = []
        expo_bic_arr = []
        expo_r2_arr = []
        expo_params = []
        expo_row = []


        bd_scores = []
        bd_aicc_arr = []
        bd_bic_arr = []
        bd_r2_arr = []
        bd_params = []
        bd_row = []

        noise_row = []

        ##########################RACHLIN LOOCV##############################################################################################

        '''for g in ParameterGrid(params_rach):
            p0 = list(g.values())
            rachlinResult = optimize.least_squares(disc_rachlin, p0, method ='lm', args=(x_data, y_data))
            if rachlinResult.x[1]> 0:
                rach_plot = plot_rachlin(x_data, rachlinResult.x[0], rachlinResult.x[1])
                if np.isfinite(rach_plot).all():
                    rach_r2 = r2_score(y_data, rach_plot)

                    rach_mse = mean_squared_error(y_data, rach_plot)
                    if rach_mse == 0:
                        rach_mse = 0.000000001
                    #rach_aic = len(x_data) * math.log(rach_mse) + 2 * 2
                    rach_aic = len(x_data) * math.log(rach_mse/len(x_data)) + 2 * 2
                    rach_aicc = rach_aic + 2*2**2 + 2*2 / len(x_data) - 2 - 1
                    #rach_bic = len(x_data) * math.log(rach_mse) + 2 * math.log(len(x_data))
                    rach_bic = len(x_data) * math.log(rach_mse/len(x_data)) + math.log(len(x_data)) * 2

                    rach_aicc_arr.append(rach_aicc)
                    rach_bic_arr.append(rach_bic)
                    rach_r2_arr.append(rach_r2)
                else:
                    pass'''
                    
        for g in ParameterGrid(params_rach):
            rachlinParams = Parameters()
            p0 = list(g.values())
            k = p0[0]
            s = p0[1]
            rachlinParams.add('kval', value = k)
            rachlinParams.add('sval', value = s)
            rachlinResult = minimize(disc_rachlin2, rachlinParams, args = (x_data, y_data), nan_policy = 'propagate', method = 'leastsq')
        
            model_best_params["RachlinS"] = rachlinResult.params['sval'].value
            model_best_params["RachlinK"] = rachlinResult.params['kval'].value
            rach_plot = plot_rachlin(x_data, model_best_params["RachlinK"], model_best_params["RachlinS"])
            if np.isfinite(rach_plot).all():
                rach_score_bic = rachlinResult.bic
                rach_score_aicc = rachlinResult.aic + 2*2*(2 + 1) / len(density) - 2 - 1
                rach_aicc_arr.append(rach_score_aicc)
                rach_bic_arr.append(rach_score_bic)


        for g in ParameterGrid(params_rach):
            myersonParams = Parameters()
            p0 = list(g.values())
            k = p0[0]
            s = p0[1]
            myersonParams.add('kval', value = k)
            myersonParams.add('sval', value = s)
            myersonResult = minimize(disc_myerson2, myersonParams, args = (x_data, y_data), nan_policy = 'propagate', method = 'leastsq')
            model_best_params["MyersonK"] = myersonResult.params['kval'].value
            model_best_params["MyersonS"] = myersonResult.params['sval'].value
            
            myer_plot = plot_myerson(x_data, model_best_params["MyersonK"], model_best_params["MyersonS"])
            if np.isfinite(myer_plot).all():
                myer_score_bic = myersonResult.bic
                myer_score_aicc = myersonResult.aic + 2*2*(2 + 1) / len(density) - 2 - 1
                myer_aicc_arr.append(myer_score_aicc)
                myer_bic_arr.append(myer_score_bic)
    
                    
                    
        

        '''for g in ParameterGrid(params_rach):
            p0 = list(g.values())
            myerResult = optimize.least_squares(disc_myerson, p0, method ='lm', args=(x_data, y_data))
            if myerResult.x[1]> 0:
                myer_plot = plot_myerson(x_data, myerResult.x[0], myerResult.x[1])
                if np.isfinite(myer_plot).all():
                    myer_r2 = r2_score(y_data, myer_plot)

                    myer_mse = mean_squared_error(y_data, myer_plot)
                    if myer_mse == 0:
                        myer_mse = 0.000000001
                    #myer_aic = len(x_data) * math.log(myer_mse) + 2 * 2
                    myer_aic = len(x_data) * math.log(myer_mse/len(x_data)) + 2 * 2
                    myer_aicc = myer_aic + 2*2**2 + 2*2 / len(x_data) - 2 - 1
                    #myer_bic = len(x_data) * math.log(myer_mse) + 2 * math.log(len(x_data))
                    myer_bic = len(x_data) * math.log(myer_mse/len(x_data)) + math.log(len(x_data)) * 2
                    
                    myer_aicc_arr.append(myer_aicc)
                    myer_bic_arr.append(myer_bic)
                    myer_r2_arr.append(myer_r2)
                else:
                    pass'''
                    
                    
                    


        '''for g in ParameterGrid(params_single):
            p0 = list(g.values())
            hyperResult = optimize.least_squares(disc_hyperbolic, p0, method ='lm', args=(x_data, y_data))
        
            hyper_plot = plot_expo(x_data, hyperResult.x[0])
            if np.isfinite(hyper_plot).all():
                hyper_r2 = r2_score(y_data, hyper_plot)

                hyper_mse = mean_squared_error(y_data, hyper_plot)

                if hyper_mse == 0:
                    hyper_mse = 0.000000001                    
                #hyper_aic = len(x_data) * math.log(hyper_mse) + 2 * 1
                hyper_aic = len(x_data) * math.log(hyper_mse/len(x_data)) + 2 * 1
                hyper_aicc = hyper_aic + 2*1**2 + 2*2 / len(x_data) - 1 - 1
                #hyper_bic = len(x_data) * math.log(hyper_mse) + 1 * math.log(len(x_data))
                hyper_bic = len(x_data) * math.log(hyper_mse/len(x_data)) + math.log(len(x_data)) * 1

                hyper_aicc_arr.append(hyper_aicc)
                hyper_bic_arr.append(hyper_bic)
                hyper_r2_arr.append(hyper_r2)
            else:
                pass'''

            
        for g in ParameterGrid(params_single):
            p0 = list(g.values())
            p0 = p0[0]
            hyperParams = Parameters()
            hyperParams.add('kval', value = p0)
            hyperResult = minimize(disc_hyperbolic2, hyperParams, args = (x_data, y_data), nan_policy = 'propagate', method = 'leastsq')
            model_best_params["HyperK"] = hyperResult.params['kval'].value
            model_best_params['HyperBIC'] = hyperResult.bic
            
            hyper_plot = plot_expo(x_data, model_best_params["HyperK"])
            if np.isfinite(hyper_plot).all():
                hyper_score_bic = hyperResult.bic
                hyper_score_aicc = hyperResult.aic + 2*2*(2 + 1) / len(density) - 2 - 1
                hyper_aicc_arr.append(hyper_score_aicc)
                hyper_bic_arr.append(hyper_score_bic)

        


        '''for g in ParameterGrid(params_single):
            p0 = list(g.values())
            expoResult = optimize.least_squares(disc_expo, p0, method ='lm', args=(x_data, y_data))
            expo_plot = plot_expo(x_data, expoResult.x[0])
            
            expo_r2 = r2_score(y_data, expo_plot)
            if np.isfinite(expo_plot).all():
                expo_mse = mean_squared_error(y_data, expo_plot)
                if expo_mse == 0:
                    expo_mse = 0.000000001
                #expo_aic = len(x_data) * math.log(expo_mse) + 2 * 1
                expo_aic = len(x_data) * math.log(expo_mse/len(x_data)) + 2 * 1
                expo_aicc = expo_aic + 2*1**2 + 2*2 / len(x_data) - 1 - 1
                #expo_bic = len(x_data) * math.log(expo_mse) + 1 * math.log(len(x_data))
                expo_bic = len(x_data) * math.log(expo_mse/len(x_data)) + math.log(len(x_data)) * 1
                
                expo_aicc_arr.append(expo_aicc)
                expo_bic_arr.append(expo_bic)
                expo_r2_arr.append(expo_r2)
            else:
                pass'''
                
        for g in ParameterGrid(params_single):
            expoParams = Parameters()
            p0 = list(g.values())
            p0 = p0[0]

            expoParams.add('kval', value = p0)
            expoResult = minimize(disc_expo2, expoParams, args = (x_data, y_data), nan_policy = 'propagate', method = 'leastsq')
        
            model_best_params["ExpoK"] = expoResult.params['kval'].value
            model_best_params['ExpoBIC'] = expoResult.bic

            expo_score_bic = expoResult.bic
            expo_score_aicc = expoResult.aic + 2*1*(1 + 1) / len(density) - 1 - 1
            expo_aicc_arr.append(expo_score_aicc)
            expo_bic_arr.append(expo_score_bic)
            
            
            

        '''for g in ParameterGrid(params_bd):
            p0 = list(g.values())
            bdResult = optimize.least_squares(disc_quasi, p0, method ='trf', bounds = (0, 1), args=(x_data, y_data))

            bd_plot = plot_quasi(x_data, bdResult.x[0], bdResult.x[1])
            if np.isfinite(bd_plot).all():
                bd_r2 = r2_score(y_data, bd_plot)

                bd_mse = mean_squared_error(y_data, bd_plot)
                if bd_mse == 0:
                    bd_mse = 0.000000001
                #bd_aic = len(x_data) * math.log(bd_mse) + 2 * 2
                bd_aic = len(x_data) * math.log(bd_mse/len(x_data)) + 2 * 2
                bd_aicc = bd_aic + 2*2**2 + 2*2 / len(x_data) - 2 - 1
                #bd_bic = len(x_data) * math.log(bd_mse) + 2 * math.log(len(x_data))
                bd_bic = len(x_data) * math.log(bd_mse/len(x_data)) + math.log(len(x_data)) * 2

                bd_aicc_arr.append(bd_aicc)
                bd_bic_arr.append(bd_bic)
                bd_r2_arr.append(bd_r2)
            else:
                pass'''


        for g in ParameterGrid(params_bd):        
            quasiParams = Parameters()
            p0 = list(g.values())
            b = p0[0]
            d = p0[1]
            quasiParams.add('bval', value = b, min = 0.0, max = 1.0)
            quasiParams.add('dval', value = d, min = 0.0, max = 1.0)
            quasiResult = minimize(disc_quasi2, quasiParams, args = (x_data, y_data), nan_policy = 'propagate', method = 'leastsq')
        
            model_best_params["QuasiB"] = quasiResult.params['bval'].value
            model_best_params["QuasiD"] = quasiResult.params['dval'].value
            model_best_params['QuasiBIC'] = quasiResult.bic
            bd_plot = plot_quasi(x_data, model_best_params["QuasiB"], model_best_params["QuasiD"])
            if np.isfinite(bd_plot).all():
                quasi_score_bic = quasiResult.bic
                quasi_score_aicc = quasiResult.aic + 2*2*(2 + 1) / len(density) - 2 - 1
                bd_aicc_arr.append(quasi_score_aicc)
                bd_bic_arr.append(quasi_score_bic)



        noise = np.array(sum(y_data/len(y_data)))
        noise = np.repeat(noise, len(x_data))
        noise = noise.reshape(len(x_data),)
        noise_r2 = r2_score(y_data, noise)
        noise_mse = mean_squared_error(y_data, noise)
        if noise_mse == 0:
            noise_mse = 0.000000001
        noise_aic = len(x_data) * math.log(noise_mse) + 2 * 2
        noise_aicc = noise_aic + 2*2**2 + 2*2 / len(x_data) - 2 - 1
        noise_bic = len(x_data) * math.log(noise_mse) + 2 * math.log(len(x_data))

        noise_row_r2.append(noise_r2)
        noise_row_aicc.append(noise_aicc)
        noise_row_bic.append(noise_bic)




        rach_total_aicc = np.min(rach_aicc_arr)
        rach_total_bic = np.min(rach_bic_arr)
        #rach_total_r2 = np.max(rach_r2_arr)
        rach_row_aicc.append(rach_total_aicc)
        rach_row_bic.append(rach_total_bic)
        #rach_row_r2.append(rach_total_r2)

        myer_total_aicc = np.min(myer_aicc_arr)
        myer_total_bic = np.min(myer_bic_arr)
        #myer_total_r2 = np.max(myer_r2_arr)
        myer_row_aicc.append(myer_total_aicc)
        myer_row_bic.append(myer_total_bic)
        #myer_row_r2.append(myer_total_r2)

        hyper_total_aicc = np.min(hyper_aicc_arr)
        hyper_total_bic = np.min(hyper_bic_arr)
        #hyper_total_r2 = np.max(hyper_r2_arr)
        hyper_row_aicc.append(hyper_total_aicc)
        hyper_row_bic.append(hyper_total_bic)
        #hyper_row_r2.append(hyper_total_r2)

        expo_total_aicc = np.min(expo_aicc_arr)
        expo_total_bic = np.min(expo_bic_arr)
        #expo_total_r2 = np.max(expo_r2_arr)
        expo_row_aicc.append(expo_total_aicc)
        expo_row_bic.append(expo_total_bic)
        #expo_row_r2.append(expo_total_r2)

        bd_total_aicc = np.min(bd_aicc_arr)
        bd_total_bic = np.min(bd_bic_arr)
        #bd_total_r2 = np.max(bd_r2_arr)
        bd_row_aicc.append(bd_total_aicc)
        bd_row_bic.append(bd_total_bic)
        #bd_row_r2.append(bd_total_r2)





        jor = jor + 1
        print("Row number: " + str(jor))
        
        




#jor = pd.DataFrame(list(zip(rach_row_r2, myer_row_r2, hyper_row_r2, expo_row_r2, bd_row_r2, noise_row_r2)),
               #columns =['Rachlin', 'Myer', 'Hyper', 'Expo', 'BD', 'Noise'])
        
#jor.to_csv('TEST_Mazur_HIGH_R2.csv')

jor2 = pd.DataFrame(list(zip(rach_row_aicc, myer_row_aicc, hyper_row_aicc, expo_row_aicc, bd_row_aicc, noise_row_aicc)),
               columns =['Rachlin', 'Myer', 'Hyper', 'Expo', 'BD', 'Noise'])
        
jor2.to_csv('RACH_LOW_AICC.csv')

jor3 = pd.DataFrame(list(zip(rach_row_bic, myer_row_bic, hyper_row_bic, expo_row_bic, bd_row_bic, noise_row_bic)),
               columns =['Rachlin', 'Myer', 'Hyper', 'Expo', 'BD', 'Noise'])
        
jor3.to_csv('RACH_LOW_BIC.csv')


        

        
