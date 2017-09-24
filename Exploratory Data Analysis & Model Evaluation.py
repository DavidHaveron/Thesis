
# coding: utf-8

# # Preprocessing, Combination and Exploratory Data Analysis of Temperature, Traffic and Stress Performance datasets

#    ### The task
# 
# Investigate the performance of multilayer perceptrons for prediction of a stress-based performance indicator for welded bridge joints based on temperature, traffic and strain measurements for the Great Belt Bridge (Denmark)
# 

# <table><tr><td><img src='https://storebaelt-prod.peytz.dk/files/inline-images/luftfoto-oestbro.jpg'></td>

# The necessary libraries are imported for data processing:

# In[1]:

import numpy as np
import pandas as pd


# ## About the Data

# Datasets were provided from three different sources, and represent three variables as per below: 
#     1. Stress
#     2. Traffic
#     3. Temperature
# 
# As with most Machine Learning (ML) models, performance is dependant on the quality of the data on which the ML models are trained.
# The data readings for the traffic, temperature and the stress performance indicator have been recorded at different sampling rates and for this reason some data munging is necessary to preprocess the data before it becomes useful for training/validation and test evaluation. The time duration is chosen to be 1 day (24 hours) and hence the three datasets are processed to reflect the chosen time discretization step Δt:

# # Data Preprocessing

# <h3><center>Temperature - 2012</center></h3>

# In[2]:

Temperature_2012 = pd.read_excel('Data\DATA Storebælt\Temperature\Station 9902_2012.xlsx', header=None)
Temperature_2012.columns = ["Average Temperature (°C)", "Year", "Month", "Day", "Hour", "Minute"]
Temperature_2012['Timestamp'] = pd.to_datetime(Temperature_2012['Year']*10000000000
                                +Temperature_2012['Month']*100000000
                                +Temperature_2012['Day']*1000000
                                +Temperature_2012['Hour']*10000
                                +Temperature_2012['Minute']*100
                                ,format='%Y%m%d%H%M%S')
Temperature_2012 = Temperature_2012.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)
Training_year_2012 = (Temperature_2012['Timestamp'] >= '2012-01-01 00:00:00') & (Temperature_2012['Timestamp'] <= '2012-12-31 23:59:59')
Temperature_2012 = Temperature_2012.loc[Training_year_2012]
Temperature_2012.index = Temperature_2012['Timestamp']
Temperature_2012 = Temperature_2012.drop('Timestamp', axis=1)
Temperature_2012 = Temperature_2012.resample('d').mean()

#Temperature_2012.head()


# <h3><center>Traffic - 2012</center></h3>

# In[3]:

Traffic_2012 = pd.read_excel('Data\DATA Storebælt\Traffic\Trafik_2012.xlsx')
Traffic_2012['Timestamp'] = pd.to_datetime(Traffic_2012['Year']*10000000000
                                +Traffic_2012['Month']*100000000
                                +Traffic_2012['Day']*1000000
                                +Traffic_2012['Hour']*10000
                                ,format = '%Y%m%d%H%M%S')
Traffic_2012.columns = ['Year', 'Month','Day', 'Hour','Class 1 - East', 'Class 2 - East','Class 3 - East', 'Class 4 - East','Class 5 - East', 'Class 6 - East','Class 7 - East', 'Class 8 - East','Class 9 - East', 'Class 10 - East','Class 1 - West', 'Class 2 - West','Class 3 - West', 'Class 4 - West','Class 5 - West', 'Class 6 - West','Class 7 - West', 'Class 8 - West','Class 9 - West', 'Class 10 - West','Total East', 'Total West', 'Total','Timestamp']
Traffic_2012.index = Traffic_2012['Timestamp']
Traffic_2012 = Traffic_2012.drop(['Year', 'Month','Day', 'Hour', 'Class 7 - East', 'Class 8 - East','Class 9 - East', 'Class 10 - East', 'Class 7 - West', 'Class 8 - West','Class 9 - West', 'Class 10 - West','Total East', 'Total West', 'Total'], axis=1)
Training_year_2012 = (Traffic_2012['Timestamp'] >= '2012-01-01 00:00:00') & (Traffic_2012['Timestamp'] <= '2012-12-31 23:59:59')
Traffic_2012 = Traffic_2012.loc[Training_year_2012]
Traffic_2012 = Traffic_2012.resample('d').sum()

#Traffic_2012.head()


# <h3><center>Stress Performance Indicator - 2012</center></h3>

# In[4]:

Stress_Indicator_2012 = pd.read_excel('Data\DATA Storebælt\Stress\DD2012.xlsx', header=None)
Stress_Indicator_2012 = Stress_Indicator_2012.T
Stress_Indicator_2012 = Stress_Indicator_2012.append(['NaN'])
Stress_Indicator_2012.columns = ['SG1', 'SG2','SG3','SG4','SG5','SG6','SG7','SG8','SG9','SG10','SG11','SG12','SG13','SG14','SG15']
Stress_Indicator_2012 = Stress_Indicator_2012.drop(['SG10','SG11','SG12','SG13','SG14','SG15'], axis=1)
Stress_Indicator_2012.index = pd.date_range('2012-01-01', '2012-12-31', freq='d')
Stress_Indicator_2012.index.name='Timestamp'

#Stress_Indicator_2012.head()


# <h3><center>Merging Temperature, Traffic & Stress Performance Indicator - 2012</center></h3>

# In[5]:

Data_2012 =  pd.concat([Temperature_2012,Traffic_2012,Stress_Indicator_2012], join='inner', axis=1)
Data_2012 = Data_2012.reset_index()
Data_2012 = Data_2012.drop(['Timestamp'], axis=1)
Data_2012 = Data_2012.dropna()

#Data_2012.head()


# <h3><center>Temperature - 2011</center></h3>

# In[6]:

Temperature_2011 = pd.read_excel('Data\DATA Storebælt\Temperature\Temperature.xlsx', header=None, skiprows=5) 
Temperature_2011 = Temperature_2011.loc[:, 0:1]
Temperature_2011.columns = ["Timestamp", "Average Temperature (°C)"]
Temperature_2011["Timestamp"] = Temperature_2011["Timestamp"].astype(str)
Temperature_2011["Timestamp"] = pd.to_datetime(Temperature_2011["Timestamp"])
Temperature_2011.index = Temperature_2011['Timestamp']
Test_year_2011 = (Temperature_2011['Timestamp'] >= '2011-01-01 00:00:00') & (Temperature_2011['Timestamp'] <= '2011-12-31 23:59:59')
Temperature_2011 = Temperature_2011.loc[Test_year_2011]
Temperature_2011 = Temperature_2011.drop('Timestamp', axis=1)
Temperature_2011 = Temperature_2011.resample('d').mean()

#Temperature_2011.head()


# <h3><center>Traffic - 2011</center></h3>

# In[7]:

Traffic_2011 = pd.read_excel('Data\DATA Storebælt\Traffic\Timetrafiksiden1998.xlsx')
Traffic_2011.columns = ['Year', 'Month','Day', 'Hour','Class 1 - East', 'Class 2 - East','Class 3 - East', 'Class 4 - East','Class 5 - East', 'Class 6 - East','Class 7 - East', 'Class 8 - East','Class 9 - East', 'Class 10 - East','Class 1 - West', 'Class 2 - West','Class 3 - West', 'Class 4 - West','Class 5 - West', 'Class 6 - West','Class 7 - West', 'Class 8 - West','Class 9 - West', 'Class 10 - West','Total East', 'Total West', '2011 Total']
Traffic_2011['Timestamp'] = pd.to_datetime(Traffic_2011['Year']*10000000000
                                +Traffic_2011['Month']*100000000
                                +Traffic_2011['Day']*1000000
                                ,format = '%Y%m%d%H%M%S')
Traffic_2011 = Traffic_2011.drop(['Year', 'Month','Day', 'Hour', 'Class 7 - East', 'Class 8 - East','Class 9 - East', 'Class 10 - East','Class 7 - West', 'Class 8 - West','Class 9 - West', 'Class 10 - West','Total East', 'Total West', '2011 Total'], axis=1)
Traffic_2011.index = Traffic_2011['Timestamp']
year_2011 = (Traffic_2011['Timestamp'] >= '2011-01-01 00:00:00') & (Traffic_2011['Timestamp'] <= '2011-12-31 23:59:59')
Traffic_2011 = Traffic_2011.loc[year_2011]
Traffic_2011 = Traffic_2011.resample('d').sum()

#Traffic_2011.head()


# <h3><center>Stress Performance Indicator - 2011</center></h3>

# In[8]:

Stress_Indicator_2011 = pd.read_excel('Data\DATA Storebælt\Stress\DD2011.xlsx', header=None)
Stress_Indicator_2011 = Stress_Indicator_2011.T
Stress_Indicator_2011.columns = ['SG1', 'SG2','SG3','SG4','SG5','SG6','SG7','SG8','SG9','SG10','SG11','SG12','SG13','SG14','SG15']
Stress_Indicator_2011=Stress_Indicator_2011.drop(['SG10','SG11','SG12','SG13','SG14','SG15'], axis=1)
Stress_Indicator_2011.index = pd.date_range('2011-01-01', '2011-12-31', freq='d', )

#Stress_Indicator_2011.head()


# <h3><center>Merging Temperature, Traffic & Stress Performance Indicator - 2011</center></h3>

# In[9]:

Data_2011 =  pd.concat([Temperature_2011,Traffic_2011,Stress_Indicator_2011], join='inner', axis=1)
Data_2011 = Data_2011.reset_index()
Data_2011.rename(columns={'index':'Timestamp'}, inplace=True)
Data_2011 = Data_2011.drop(['Timestamp'], axis=1)
Data_2011 = Data_2011.dropna()

#Data_2011.head()


# Pandas profiling calculates summary statistics and allows for visual intepretation:

# In[10]:

#!pip install pandas_profiling
#import pandas_profiling
#pandas_profiling.ProfileReport(Data_2011)


# Pixiedust provides a flexible interface to explore data graphically:

# In[11]:

#!pip install pixiedust
#import pixiedust
#display(Data_2011)


# The cleaned datasets from 2011 and 2012 are merged into a single dataset and shuffled at random:

# # Model Development & Evaluation

# ### The model performance metrics are defined...

# The Akaike Information Criterion (AIC) is defined as:
# <img src="AIC.jpg" alt="Drawing" style="width: 400px;" position = "centre"/>

# AIC is biased for small samples. Hence, a bias corrected version referred as AICc is preferred
# over the classical formulation, see Burnham & Anderson (2002).
# The Akaike Information Criterion (Corrected) is defined as:
# <img src="AICc.jpg" alt="Drawing" style="width: 300px;" position = left/>

# 
# Define the formula for Mean Square Error:
# <img src="mse.jpg" alt="Drawing" style="width: 320px position = "centre""/>

# Define the formula for Mean Absolute Percentage Error:
# <img src="MAPE.jpg" alt="Drawing" style="width: 380px;"/>

# In[12]:

#!pip uninstall h2o
# Next, use pip to install this version of the H2O Python module.
#!pip install http://h2o-release.s3.amazonaws.com/h2o/master/3978/Python/h2o-3.13.0.3978-py2.py3-none-any.whl

# Relavent reading material
# help(H2ODeepLearningEstimator)
# help(h2o.import_file)
# https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.ipynb
# https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/dl/dl.md
# https://h2o-release.s3.amazonaws.com/h2o/rel-slater/9/docs-website/h2o-docs/booklets/DeepLearning_Vignette.pdf
# https://blog.h2o.ai/2016/06/hyperparameter-optimization-in-h2o-grid-search-random-search-and-the-future/
# https://www.kaggle.com/faizanbatra/house-prices-advanced-regression-techniques/dm-fbatra/run/492655
# https://github.com/h2oai/h2o-3/blob/master/h2o-py/h2o/grid/grid_search.py
# https://blog.dominodatalab.com/deep-learning-with-h2o-ai/ - H20 easy setup


# # Define the grid search parameters:

# H20's Deep learning library is based on a multi-layer feedforward artificial neural networks trained using stochastic gradient decent using back-propagation. Each compute node trains a copy of the global model parameters on its local data with multi-threading (asynchronously) and contributes periodically to the global model via model averaging across the network.

# In[13]:

import h2o
h2o.init()


# In[14]:

# Define the training & cross-validation dataframe
training_frame = Data_2012
training_frame = h2o.H2OFrame(training_frame)
training_frame = training_frame.drop([0], axis=0)

# Define the test dataframe
test_frame = Data_2011
test_frame = h2o.H2OFrame(test_frame)
test_frame = test_frame.drop([0], axis=0)


# In[15]:

len(test_frame)


# In[16]:

#Prepare predictors
x = ['Average Temperature',
    #'Class 1 - East',
    #'Class 2 - East',
    #'Class 3 - East',
    #'Class 4 - East',
    'Class 5 - East',
    'Class 6 - East',
    #'Class 1 - West',
    #'Class 2 - West',
    #'Class 3 - West',
    #'Class 4 - West',
    'Class 5 - West',
    'Class 6 - West' ]

m = len(x)


# Setup the model to train 9 seperate models and calculate performance results
my_target_variables = ['SG1','SG2','SG3','SG4','SG5','SG6','SG7','SG8','SG9']

models = {}

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Potential hyperparameters #[m+1,m+1]
hyper_parameters = { 'activation'  : ["Rectifier"],
                     'hidden' : [[m-1], [m-1,m-1], [m], [m,m], [m,m,m], [m+1,m+1,m+1], [2*m], [20], [30], [40], [50]],
                     'input_dropout_ratio'  : [0,0.05],
                     'l1'  : [0,1e-2, 1e-4,1e-6],
                     'l2'  : [0,1e-2, 1e-4,1e-6]
              
                   }
    

search_criteria  = { 'strategy' : "RandomDiscrete",
                    'stopping_metric': "mse",
                     #'max_runtime_secs' : 3600, 
                     'max_models' : 6, 
                     'seed' : 12345, 
                     'stopping_rounds' : 5, 
                    }

deep_learning_random_grid = H2OGridSearch(H2ODeepLearningEstimator(),hyper_parameters)

for variable in my_target_variables:
    print('Training model '+ variable + ' & calculating training error...'  )
    deep_learning_random_grid.train(
                                  training_frame=training_frame,
                                  nfolds = 10, 
                                  x=x, 
                                  y=variable,
                                  overwrite_with_best_model= True,
                                  epochs=15,
                                  standardize = True,
                                  shuffle_training_data = True,
                                  distribution="gaussian",
                                  variable_importances=True,
                                  adaptive_rate = True,         
                                  train_samples_per_iteration=-2,
                                  score_validation_samples=10, 
                                  score_duty_cycle=0.025,         
                                  max_w2=10,                  
                                  search_criteria = search_criteria)
    grid = deep_learning_random_grid.get_grid(sort_by='mse')
    best_model = grid[4]
    models[variable] =  best_model
    print('')


# In[17]:

grid.show()
#best_model.get_hyperparams_dict


# In[18]:

from sklearn.metrics import mean_squared_error

def evaluate_model(y_observed, y_predicted, m):
    
    # Ensure the vectors are of equal lengthzz
    assert len(y_observed) == len(y_predicted)
    #y_actual, y_predicted = check_array(y_actual, y_predicted)

    # Calculate the Mean Square Error (MSE)
    #MSE = np.mean((y_observed-y_predicted)**2) or MSE = mean_squared_error(y_actual, y_predicted) - Cross Validation is used
    
    #######################################################
    
    # Calculate the Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean(np.abs((y_observed - y_predicted) / y_observed)) * 100
    MAPE = "{0:0.1f}".format(MAPE)
    
    #######################################################
    
    # Calculate the AIC & AICc
    # r is number of features
    n =  len(y_observed) #number of observations

    resid = y_observed - y_predicted
    sse = sum(resid ** 2)

    AIC =  2*m - 2*np.log(sse)
    AICc = AIC + (2*m*(m+1))/(n-m-1)
    AICc = float(AICc)
    AICc = "{0:0.1f}".format(AICc)
    
    #######################################################
    
    # Calculate the Ψ value
    Ψ = sum(y_predicted)/sum(y_observed)
    Ψ = float(Ψ)*100
    Ψ = "{0:0.1f}".format(Ψ)

    return MAPE, AICc, Ψ

def access_dictionary(dictionary_name, *keys):
    for key in keys:
        try:
            dictionary_name = dictionary_name[key]
        except KeyError:
            return None
    return dictionary_name


# In[19]:

# Define the y_observed & y_predicted for training/test error calculation
from sklearn.metrics import mean_squared_error
model_results = {}

def calculate_model_performance(models):
    my_target_variables = ['SG1','SG2','SG3','SG4','SG5','SG6','SG7','SG8','SG9']
    for variable in my_target_variables:
 
#TRAINING PERFORMANCE
        # Define Training y_observed
        y_obs_train = training_frame[variable].as_data_frame(use_pandas=True) # y_observed - training, convert to pandas df
        y_obs_train = y_obs_train.as_matrix() # convert to numpy array
        y_pred_train = models[variable].predict(training_frame).as_data_frame(use_pandas=True) # y_predicted - training
        y_pred_train = y_pred_train.as_matrix()
        
        # Call the Cross_validated MSE values for each Strain Gauge
        MSE_train = models[variable].mse(xval=True) # Cross validation mse calucluated
        MSE_train = "{:.2E}".format(MSE_train)
        
        # Call the evaluate_model function for each Strain Gauge
        MAPE_train, AICc_train, Ψ_train  = evaluate_model(y_obs_train, y_pred_train, m)
        
#TEST PERFORMANCE
        # Define Training y_observed
        y_obs_test = test_frame[variable].as_data_frame(use_pandas=True) # y_observed - training, convert to pandas df
        y_obs_test = y_obs_test.as_matrix() # convert to numpy array
        y_pred_test = models[variable].predict(test_frame).as_data_frame(use_pandas=True) # y_predicted - training
        y_pred_test = y_pred_test.as_matrix()
        
        # Call the Cross_validated MSE values for each Strain Gauge
        MSE_test = np.mean((y_obs_test-y_pred_test)**2) # Cross validation mse calucluated
        MSE_test = "{:.2E}".format(MSE_test)
        
        # Call the evaluate_model function for each Strain Gauge
        MAPE_test, _  , Ψ_test = evaluate_model(y_obs_test, y_pred_test, m)
       
        
# POPULATE A DICTIONARY WITH RESULTS
        model_results[variable] =  { 'MSE_train' : MSE_train,
                           'MAPE_train' : MAPE_train,
                           'AICc_train' : AICc_train,
                           'Ψ_train' : Ψ_train,
                           'MSE_test' : MSE_test, 
                           'MAPE_test' : MAPE_test,
                           'Ψ_test' : Ψ_test}

        
calculate_model_performance(models)


# In[20]:

model_results # 600with13inputs
modelsFIVE['SG9']


# Populate the results

# In[ ]:

# Populate a results table
arrays = [np.hstack([ ['Training/validation']*4, ['Test']*4]), (['AICc','MSE', 'MAPE', 'Ψ(%)']*2)]

columns = pd.MultiIndex.from_arrays(arrays, names=['', 'Model'])
index=['SG1 - linear regression','SG1 - MLP',
       'SG2 - linear regression','SG2 - MLP',
       'SG3 - linear regression','SG3 - MLP',
       'SG4 - linear regression','SG4 - MLP',
       'SG5 - linear regression','SG5 - MLP',
       'SG6 - linear regression','SG6 - MLP',
       'SG7 - linear regression','SG7 - MLP',
       'SG8 - linear regression','SG8 - MLP',
       'SG9 - linear regression','SG9 - MLP',
      ]

results =pd.DataFrame(np.zeros((18,8)),columns=columns,index= index)


#SG1
results.loc['SG1 - linear regression'] = ['-703.5','3.41E+11','15.9','98.3','N/A','2.08E+11','35.1','106.9']
results.loc['SG1 - MLP'] = [access_dictionary(model_results, 'SG1', 'AICc_train'),
                              access_dictionary(model_results, 'SG1', 'MSE_train'),
                              access_dictionary(model_results, 'SG1', 'MAPE_train'),
                              access_dictionary(model_results, 'SG1', 'Ψ_train'),
                              'N/A',
                              access_dictionary(model_results, 'SG1', 'MSE_test'),
                              access_dictionary(model_results, 'SG1', 'MAPE_test'),
                              access_dictionary(model_results, 'SG1', 'Ψ_test')]


#SG2
results.loc['SG2 - linear regression'] = ['-911.7','6.87E+11','8.8','102.1','N/A','1.52E+12','35.0','119.1']
results.loc['SG2 - MLP'] = [access_dictionary(model_results, 'SG2', 'AICc_train'),access_dictionary(model_results, 'SG2', 'MSE_train'),access_dictionary(model_results, 'SG2', 'MAPE_train'),access_dictionary(model_results, 'SG2', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG2', 'MSE_test'),access_dictionary(model_results, 'SG2', 'MAPE_test'),access_dictionary(model_results, 'SG2', 'Ψ_test')]

#SG3
results.loc['SG3 - linear regression'] = ['-464.9','2.00E+11','27.1','103.0','N/A','1.26E+11','41.5','99.5']
results.loc['SG3 - MLP'] = [access_dictionary(model_results, 'SG3', 'AICc_train'),access_dictionary(model_results, 'SG3', 'MSE_train'),access_dictionary(model_results, 'SG3', 'MAPE_train'),access_dictionary(model_results, 'SG3', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG3', 'MSE_test'),access_dictionary(model_results, 'SG3', 'MAPE_test'),access_dictionary(model_results, 'SG3', 'Ψ_test')]


#SG4
results.loc['SG4 - linear regression'] = ['-615.1','3.48E+10','22.2','97.6','N/A','7.01E+10','32.4','104.5']
results.loc['SG4 - MLP'] = [access_dictionary(model_results, 'SG4', 'AICc_train'),access_dictionary(model_results, 'SG4', 'MSE_train'),access_dictionary(model_results, 'SG4', 'MAPE_train'),access_dictionary(model_results, 'SG4', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG3', 'MSE_test'),access_dictionary(model_results, 'SG4', 'MAPE_test'),access_dictionary(model_results, 'SG4', 'Ψ_test')]


#SG5
results.loc['SG5 - linear regression'] = ['-799.3','1.76E+10','13.5','99.2','N/A','5.05E+11','45.1','125.2']
results.loc['SG5 - MLP'] = [access_dictionary(model_results, 'SG5', 'AICc_train'),access_dictionary(model_results, 'SG5', 'MSE_train'),access_dictionary(model_results, 'SG5', 'MAPE_train'),access_dictionary(model_results, 'SG5', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG5', 'MSE_test'),access_dictionary(model_results, 'SG5', 'MAPE_test'),access_dictionary(model_results, 'SG5', 'Ψ_test')]


#SG6
results.loc['SG6 - linear regression'] = ['-641.2','3.17E+11','20.4','97.5','N/A','5.00E+11','36.8','111.7']
results.loc['SG6 - MLP'] = [access_dictionary(model_results, 'SG6', 'AICc_train'),access_dictionary(model_results, 'SG6', 'MSE_train'),access_dictionary(model_results, 'SG6', 'MAPE_train'),access_dictionary(model_results, 'SG6', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG6', 'MSE_test'),access_dictionary(model_results, 'SG6', 'MAPE_test'),access_dictionary(model_results, 'SG6', 'Ψ_test')]


#SG7
results.loc['SG7 - linear regression'] = ['-622.2','3.54E+11','22.1','97.2','N/A','3.70E+11','54.6','122.0']
results.loc['SG7 - MLP'] = [access_dictionary(model_results, 'SG7', 'AICc_train'),access_dictionary(model_results, 'SG7', 'MSE_train'),access_dictionary(model_results, 'SG7', 'MAPE_train'),access_dictionary(model_results, 'SG7', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG7', 'MSE_test'),access_dictionary(model_results, 'SG7', 'MAPE_test'),access_dictionary(model_results, 'SG7', 'Ψ_test')]


#SG8
results.loc['SG8 - linear regression'] = ['-900.2','1.31E+12','10.0','99.9','N/A','4.05E+12','38.6','116.7']
results.loc['SG8 - MLP'] = [access_dictionary(model_results, 'SG8', 'AICc_train'),access_dictionary(model_results, 'SG8', 'MSE_train'),access_dictionary(model_results, 'SG8', 'MAPE_train'),access_dictionary(model_results, 'SG8', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG8', 'MSE_test'),access_dictionary(model_results, 'SG8', 'MAPE_test'),access_dictionary(model_results, 'SG8', 'Ψ_test')]

#SG9
results.loc['SG9 - linear regression'] = ['-795.9','1.08E+11','13.2','98.9','N/A','1.43E+10','32.9','107.3']
results.loc['SG9 - MLP'] = [access_dictionary(model_results, 'SG9', 'AICc_train'),access_dictionary(model_results, 'SG9', 'MSE_train'),access_dictionary(model_results, 'SG9', 'MAPE_train'),access_dictionary(model_results, 'SG9', 'Ψ_train'),'N/A',access_dictionary(model_results, 'SG9', 'MSE_test'),access_dictionary(model_results, 'SG9', 'MAPE_test'),access_dictionary(model_results, 'SG9', 'Ψ_test')]
results



# In[ ]:

# Plotting AICc
Index = ['SG1','SG2','SG3','SG4','SG5','SG6','SG7','SG8','SG9']
AICc_Training_Isaac = [-703.5, -911.7, -464.9, -615.1, -799.3, -641.2, -622.2, -900.2, -795.9]
AICc_Training_David = [access_dictionary(model_results, 'SG1', 'AICc_train'),access_dictionary(model_results, 'SG2', 'AICc_train'),access_dictionary(model_results, 'SG3', 'AICc_train'),access_dictionary(model_results, 'SG4', 'AICc_train'), access_dictionary(model_results, 'SG5', 'AICc_train'), access_dictionary(model_results, 'SG6', 'AICc_train'), access_dictionary(model_results, 'SG7', 'AICc_train'), access_dictionary(model_results, 'SG8', 'AICc_train'), access_dictionary(model_results, 'SG9', 'AICc_train')]
AICc_Training_David = list(map(float, AICc_Training_David))

# Plotting MSE - Training/Test
MSE_Training_Isaac = [3.41E+11,6.87E+11,2.00E+11, 3.48E+10,1.76E+10,3.17E+11,3.54E+11,1.31E+12,1.08E+11]
MSE__Training_David = [access_dictionary(model_results, 'SG1', 'MSE_train'),access_dictionary(model_results, 'SG2', 'MSE_train'),access_dictionary(model_results, 'SG3', 'MSE_train'),access_dictionary(model_results, 'SG4', 'MSE_train'),access_dictionary(model_results, 'SG5', 'MSE_train'),access_dictionary(model_results, 'SG6', 'MSE_train'),access_dictionary(model_results, 'SG7', 'MSE_train'),access_dictionary(model_results, 'SG8', 'MSE_train'),access_dictionary(model_results, 'SG9', 'MSE_train')]
MSE__Training_David = list(map(float, MSE__Training_David))

MSE_Test_Isaac = [2.08E+11, 1.52E+12, 1.26E+11, 7.01E+10, 5.05E+11, 5.00E+11, 3.70E+11, 4.05E+12, 1.43E+10]
MSE__Test_David = [access_dictionary(model_results, 'SG1', 'MSE_test'),access_dictionary(model_results, 'SG2', 'MSE_test'),access_dictionary(model_results, 'SG3', 'MSE_test'),access_dictionary(model_results, 'SG4', 'MSE_test'),access_dictionary(model_results, 'SG5', 'MSE_test'),access_dictionary(model_results, 'SG6', 'MSE_test'),access_dictionary(model_results, 'SG7', 'MSE_test'),access_dictionary(model_results, 'SG8', 'MSE_test'),access_dictionary(model_results, 'SG9', 'MSE_test')]
MSE__Test_David = list(map(float, MSE__Test_David))

# Plotting MAPE - Training/Test
MAPE_Training_Isaac = [15.9, 8.8, 27.1, 22.2, 13.5, 20.4, 22.1, 10.0z, 13.2]
MAPE__Training_David = [access_dictionary(model_results, 'SG1', 'MAPE_train'),access_dictionary(model_results, 'SG2', 'MAPE_train'),access_dictionary(model_results, 'SG3', 'MAPE_train'),access_dictionary(model_results, 'SG4', 'MAPE_train'),access_dictionary(model_results, 'SG5', 'MAPE_train'),access_dictionary(model_results, 'SG6', 'MAPE_train'),access_dictionary(model_results, 'SG7', 'MAPE_train'),access_dictionary(model_results, 'SG8', 'MAPE_train'),access_dictionary(model_results, 'SG9', 'MAPE_train')]
MAPE__Training_David = list(map(float, MAPE__Training_David))

MAPE_Test_Isaac = [35.1, 35.0, 41.5, 32.4, 45.1, 36.8, 54.6, 38.6, 32.9]
MAPE__Test_David = [access_dictionary(model_results, 'SG1', 'MAPE_test'),access_dictionary(model_results, 'SG2', 'MAPE_test'),access_dictionary(model_results, 'SG3', 'MAPE_test'),access_dictionary(model_results, 'SG4', 'MAPE_test'),access_dictionary(model_results, 'SG5', 'MAPE_test'),access_dictionary(model_results, 'SG6', 'MAPE_test'),access_dictionary(model_results, 'SG7', 'MAPE_test'),access_dictionary(model_results, 'SG8', 'MAPE_test'),access_dictionary(model_results, 'SG9', 'MAPE_test')]
MAPE__Test_David = list(map(float, MAPE__Test_David))

# Plotting Ψ(%) - Training/Test
Ψ_Training_Isaac = [98.3, 102.1, 103.0, 97.6, 99.2, 97.5, 97.2, 99.9, 98.9]
Ψ__Training_David = [access_dictionary(model_results, 'SG1', 'Ψ_train'),access_dictionary(model_results, 'SG2', 'Ψ_train'),access_dictionary(model_results, 'SG3', 'Ψ_train'),access_dictionary(model_results, 'SG4', 'Ψ_train'),access_dictionary(model_results, 'SG5', 'Ψ_train'),access_dictionary(model_results, 'SG6', 'Ψ_train'),access_dictionary(model_results, 'SG7', 'Ψ_train'),access_dictionary(model_results, 'SG8', 'Ψ_train'),access_dictionary(model_results, 'SG9', 'Ψ_train')]
Ψ__Training_David = list(map(float, Ψ__Training_David))

Ψ_Test_Isaac = [106.9, 119.1, 99.5, 104.5, 125.2, 111.7, 122.0, 116.7, 107.3]
Ψ__Test_David = [access_dictionary(model_results, 'SG1', 'Ψ_test'),access_dictionary(model_results, 'SG2', 'Ψ_test'),access_dictionary(model_results, 'SG3', 'Ψ_test'),access_dictionary(model_results, 'SG4', 'Ψ_test'),access_dictionary(model_results, 'SG5', 'Ψ_test'),access_dictionary(model_results, 'SG6', 'Ψ_test'),access_dictionary(model_results, 'SG7', 'Ψ_test'),access_dictionary(model_results, 'SG8', 'Ψ_test'),access_dictionary(model_results, 'SG9', 'Ψ_test')]
Ψ__Test_David = list(map(float, Ψ__Test_David))

AICc_data = pd.DataFrame(
    { 'Strain Gauge' : Index,
     'AICc - Training Error (multiple linear regression)' : AICc_Training_Isaac,
     'AICc - Training Error (MLP)': AICc_Training_David})

MSE_data = pd.DataFrame(
    { 'Strain Gauge' : Index,
     'MSE - Training Error (multiple linear regression)' : MSE_Training_Isaac,
     'MSE - Test Error (multiple linear regression)' : MSE_Test_Isaac,
     'MSE - Training Error (MLP)': MSE__Training_David,
     'MSE - Test Error (MLP)': MSE__Test_David})

MAPE_data = pd.DataFrame(
    { 'Strain Gauge' : Index,
     'MAPE - Training Error (multiple linear regression)' : MAPE_Training_Isaac,
     'MAPE - Test Error (multiple linear regression)' : MAPE_Test_Isaac,
     'MAPE - Training Error (MLP)': MAPE__Training_David,
     'MAPE - Test Error (MLP)': MAPE__Test_David})

Ψ_data = pd.DataFrame(
    { 'Strain Gauge' : Index,
    ' Ψ - Training Error (multiple linear regression)' : Ψ_Training_Isaac,
     'Ψ - Test Error (multiple linear regression)' : Ψ_Test_Isaac,
     'Ψ - Training Error (MLP)': Ψ__Training_David,
     'Ψ - Test Error (MLP)': Ψ__Test_David})


# In[ ]:

import pixiedust
display(MSE_data)


# In[ ]:

Strain_gauge = 'SG9'
y_obs_test = test_frame[Strain_gauge].as_data_frame(use_pandas=True)
y_pred_test = models[Strain_gauge].predict(test_frame).as_data_frame(use_pandas=True)
combined =  pd.concat([y_obs_test,y_pred_test], join='inner', axis=1)
import pixiedust
display(combined)


# In[ ]:

get_ipython().system('pip install lolviz')


# In[21]:

get_ipython().system('pip install pandoc')

