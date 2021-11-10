import random
import pandas as pd
from skopt import BayesSearchCV
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer
import time
import datetime
import os
import numpy as np
from calendar import isleap
########################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.stats import norm, skew, kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV


import matplotlib.dates as mdates
from datetime import datetime, timedelta

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
########################################################################################
# to read directories
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as plticker
import warnings
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------
import math
#from wdmtoolbox import wdmtoolbox 
import matplotlib.dates as mdates
from dateutil import relativedelta
import numpy.polynomial.polynomial as poly
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from functools import reduce
from PIL import Image
from itertools import islice
import csv
# to avoid error warnings 'implicitly registered datetime coverter'
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# to center figures and tables throughout the report
from IPython.display import display, HTML
display(HTML("""<style>.output {
	display: flex;
	align-items: center;
	text-align: center;}</style> """))
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
#from sklearn.model_selection import BayesSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
####
import shap
shap.initjs()
from time import time
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import datetime
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.model_selection import KFold,cross_val_score
import hydroeval as he




#===========================================================================#
						# NSE#
#===========================================================================#

def NSE(predicted,actual):
	nsee = he.evaluator(he.nse, predicted,actual)
	return nsee



#===========================================================================#
						# Read files functions #
#===========================================================================#
def read_tabular_data(file_name,base_dir):
	df      = os.path.join(base_dir,file_name)
	df_read = pd.read_csv(df, thousands=r',')                                   
	return df_read
def data_reader(file_name,base_dir, skip_rows):
	df      = base_dir+file_name
	df_read = pd.read_csv(df, thousands=r',', skiprows = skip_rows) 
	return df_read



from datetime import datetime


#Creates a folder according to the time

def create_folder_to_save(name):
	now = datetime.now()
	dt = now.strftime("%H:%M:%S")
	path_ = os.getcwd()
	result = os.path.join(path_+name+'_'+dt)
	os.mkdir(result)
	return result


def make_path_to_save(Results,name): 
	path_ = os.path.join(Results,name)
	save_path = os.mkdir(path_)
	return path_



def Load_df_AI(file_path):
	
	df_read = pd.read_csv(file_path)
	df_read.index = pd.DatetimeIndex(df_read['Unnamed: 0'])
	df_read = df_read.drop(columns = ['Unnamed: 0', 'Month'])
	df_read=df_read.rename(columns = {"TMIN":"$T_{min}$ [$^oC$]",
									  "TMAX":"$T_{max}$ [$^oC$]",
									  "PRCP":"$P$ [mm]",
									  "SanMarcos_SF(m3)":"SF$[m^3/s]$"})
	df_read.index =df_read.index.rename('DATE')
	df_AI = pd.DataFrame()
	df_AI['$T_{min}$ [$^oC$]'] = df_read['$T_{min}$ [$^oC$]'].resample('W').mean()
	df_AI['$T_{max}$ [$^oC$]'] = df_read['$T_{max}$ [$^oC$]'].resample('W').mean()
	df_AI['$P$ [mm]'] = df_read['$P$ [mm]'].resample('W').sum()
	return df_AI



def Load_Springs(df_AI,Approved_date,skip_rows,base_dir,filename):
	
	### --- Preprocess the csv file --- ####

	Spring_Flow_or = data_reader(filename, base_dir, skip_rows)
	Spring_Flow_or
	Spring_Flow = []
	Spring_Flow = Spring_Flow_or[['20d', '14n']].copy()
	Spring_Flow = Spring_Flow.rename(columns = {'20d' : 'Date', '14n' : 'SF(cfs)'})
	Spring_Flow['Date']  =  pd.to_datetime(Spring_Flow['Date'])
	Spring_Flow['SF$[m^3/s]$'] = round(Spring_Flow['SF(cfs)'].astype(float)*0.028316847, 3)
	Spring_Flow.drop(index=0,inplace=True)
	Spring_Flow.index = pd.DatetimeIndex(Spring_Flow['Date'])
	Spring_Flow.drop('Date',axis =1,inplace=True)
	### --- San Marcos dataframe
	Spring_Flow
	df_numeric_values = pd.to_numeric(Spring_Flow['SF$[m^3/s]$'], errors='coerce')
	sf_data = df_numeric_values.to_frame()
	### --- Data is Transformed to weekly  mean from daily data and truncated 
	SF_Data =sf_data.resample('W').mean()
	SF_Data = SF_Data[(SF_Data.index >= '1960-09-01')]
	### --- Spring flow data until 15/11/2020 is the "approved" data. The remaining data is provisional. 
	### --- Provisional data will be used after being approved
	df_AI['SF$[m^3/s]$'] = (SF_Data[(SF_Data.index <= str(Approved_date))])
	df_AI = (df_AI[(df_AI.index <= str(Approved_date))])
	df_AI.describe()
	
	return df_AI,sf_data

  

#===========================================================================#
				#Print the results of the model#
#===========================================================================#
	

def results_from(path_to_Save,Y_Train_Pred):
	fig, ax1 = plt.subplots(1,1, figsize=(5,3.5))
	ax1.scatter(df_AI_Train['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				pd.Series(Y_Train_Pred).sort_values(ascending=True).to_numpy(), 
				marker='o', color = 'black')
	RSQ = np.round((r2_score(df_AI_Train['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				pd.Series(Y_Train_Pred).sort_values(ascending=True).to_numpy())),3)
	ax1.text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
			verticalalignment='bottom', horizontalalignment='right',
			transform=ax1.transAxes,
			color='black', fontsize=21)
	ax1.set_xlabel('Recorded', fontsize = 18)
	ax1.tick_params(axis = "x", labelsize = 14, rotation=34)
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('Predicted', fontsize = 18)
	ax1.tick_params(axis = "y", labelsize = 14)
	ax1.yaxis.set_tick_params(pad=5)
	plt.tight_layout(pad=1.2)
	fig_train = fig
	fig.savefig(path_to_Save + str('/Train_Plot_score.jpg'), format='jpeg', dpi=300, bbox_inches='tight')
	
   ############################################################################                          #Test set plot                                      ############################################################################
	fig, ax1 = plt.subplots(1,1, figsize=(5,3.5))
	ax1.scatter(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				marker='o', color = 'black')
	RSQ = np.round((r2_score(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy())),3)
	ax1.text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
			verticalalignment='bottom', horizontalalignment='right',
			transform=ax1.transAxes,
			color='black', fontsize=21)
	ax1.set_xlabel('Recorded', fontsize = 18)
	ax1.tick_params(axis = "x", labelsize = 14, rotation=34)
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('Predicted', fontsize = 18)
	ax1.tick_params(axis = "y", labelsize = 14)
	ax1.yaxis.set_tick_params(pad=5)
	plt.tight_layout(pad=1.2)
	fig_test = fig
	fig.savefig(path_to_Save + str('/Test_Plot_score.jpg'), format='jpeg', dpi=300, bbox_inches='tight')

	return fig_train , fig_test

def print_results(model, X_train,y_train,save):
	Y_train_pred = model.predict(X_train)
	Model_SF = model
	RMS = np.sqrt(mean_squared_error(y_train,Y_train_pred))
	RSQ = r2_score(y_train,Y_train_pred)
	MAE = np.mean(np.abs((y_train - Y_train_pred))) 
	NSEe = he.evaluator(he.nse, Y_train_pred, y_train) 
	print("############ Model Accuracy -Training set ############")
	print("MAE: %0.3f" % MAE,'m/s^3')
	print("NSE: %0.3f" % NSEe)
	print("RMS: %0.3f" % RMS, 'm/s^3')
	print("R-squared: %0.3f" % RSQ)
	print("########################################")
	print(" ")
	np.savetxt(save+'/train_pred.out',Y_train_pred)
	np.savetxt(save+'/train.out',y_train)
	
################################################################################################
	# Make predictions on Testing data and Test model accuracy
	#################################################################################################
	df_AI_Test['SF$^{pred} [m^3/s]$'] = 0
	Test_X = df_AI_Test[Features].iloc[0].to_numpy().reshape(1, -1)
	df_AI_Test['SF$^{pred} [m^3/s]$'].iloc[0] = Model_SF.predict(Test_X)[0]

	i = 0
	for row in df_AI_Test.itertuples():
		df_AI_Test['SF$^{lag1} [m^3/s]$'][i+1] = df_AI_Test['SF$^{pred} [m^3/s]$'].iloc[i]

		df_AI_Test['SF$^{lag2} [m^3/s]$'][i+1] = df_AI_Test['SF$^{lag1} [m^3/s]$'].iloc[i]

		Test_X = df_AI_Test[Features].iloc[i+1].to_numpy().reshape(1, -1)
		df_AI_Test['SF$^{pred} [m^3/s]$'].iloc[i+1] = Model_SF.predict(Test_X)[0]
		i = i+1
		if i == (len(df_AI_Test)-1):
			break

	RMS = np.sqrt(mean_squared_error(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(),
									  df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy()))
	RSQ = r2_score(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(),
				   df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy())
	MAE = np.mean(np.abs(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy() - 
						 df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy()))
	NSEe = he.evaluator(he.nse,df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy(),
			  df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy())
	print("############ Model Accuracy -Test set############")
	print("MAE: %0.3f" % MAE,'m/s^3')
	print("NSE: %0.3f" % NSEe)
	print("R-squared: %0.3f" % RSQ)
	print("RMS: %0.3f" % RMS, 'm/s^3')
	print("########################################")
	print(" ")

	results_from(save,Y_train_pred)

#===========================================================================#
						#Optimizartion Genetic Algorithm #
#===========================================================================#
def Genetic_algorithm(Model, objective_parameters,X_train,y_train,save_extra ):
		

	scoring_param = make_scorer(mean_squared_error,greater_is_better=False)
	
	algorithm_param = {'max_num_iteration': 50,
					   'population_size':100,
					   'mutation_probability':0.1,
					   'elite_ratio': 0.01,
					   'crossover_probability': 0.5,
					   'parents_portion': 0.3,
					   'crossover_type':'uniform',
					   'max_iteration_without_improv':8}
	
	
	##Define the function
	
	def objective_function(args):
		clf = Model(**args)
		
		#tscv = TimeSeriesSplit(n_splits=3)
		#scoring_param = r2_score
		g = cross_val_score(clf, X_train, y_train.ravel(), cv=3, scoring= scoring_param) 
		mse = g.mean()
		
		
		return mse
	
	
	
	## Call the Genetic Algorithm search and 
	evo_algo = ea(function=objective_function, 
			   parameters=objective_parameters,
				  algorithm_parameters=algorithm_param,function_timeout=100)
	
	
	# Run EA
	evo_algo.run()
	
	# Access best model parameters
	evo_algo.best_parameters
	
	params = evo_algo.best_parameters   



	model = Model(**params)
	

	model.fit(X_train,y_train)
	print_results(model, X_train,y_train,save_extra)
	
	
	return params

#===========================================================================#
				# Model train after using genetic algorithm #
#===========================================================================#
def Data_training(Model, path_to_Save):


	import sys
	# SpringFlow Model - Linear Regression
	t0 = time()

	#Fit the model 
	Model_SF =Model.fit(train_X, train_y.ravel())

	t1 = time()
	#make the time to be in minutes
	Time_Taken = (t1-t0)/60
	print("Time taken to train the model: %0.2f" % Time_Taken,"minutes")
	
	
	
	Y_Train_Pred = Model_SF .predict(train_X)
	RMS = np.sqrt(mean_squared_error(y_train,Y_train_pred))
	RSQ = r2_score(y_train,Y_train_pred)
	MAE = np.mean(np.abs((y_train - Y_train_pred))) 
	NSEe = NSE(Y_train_pred,y_train)
	print("############ Model Accuracy -Training set ############")
	print("MAE: %0.3f" % MAE,'m/s^3')
	print("NSE: %0.3f" % NSEe)
	print("RMS: %0.3f" % RMS, 'm/s^3')
	print("R-squared: %0.3f" % RSQ)
	print("########################################")
	print(" ")
	#################################################################################################
	# Make predictions on Testing data and Test model accuracy
	#################################################################################################
	df_AI_Test['SF$^{pred} [m^3/s]$'] = 0
	Test_X = df_AI_Test[Features].iloc[0].to_numpy().reshape(1, -1)
	df_AI_Test['SF$^{pred} [m^3/s]$'].iloc[0] = Model_SF.predict(Test_X)[0]
	i = 0
	for row in df_AI_Test.itertuples():
		df_AI_Test['SF$^{lag1} [m^3/s]$'][i+1] = df_AI_Test['SF$^{pred} [m^3/s]$'].iloc[i]
		df_AI_Test['SF$^{lag2} [m^3/s]$'][i+1] = df_AI_Test['SF$^{lag1} [m^3/s]$'].iloc[i]
		Test_X = df_AI_Test[Features].iloc[i+1].to_numpy().reshape(1, -1)
		df_AI_Test['SF$^{pred} [m^3/s]$'].iloc[i+1] = Model_SF.predict(Test_X)[0]
		i = i+1
		if i == (len(df_AI_Test)-1):
			break
	RMS = np.sqrt(mean_squared_error(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(),
									  df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy()))
	RSQ = r2_score(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(),
				   df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy())
	MAE = np.mean(np.abs(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy() - 
						 df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy()))
	NSEe = NSE(df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy(),
			  df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy())
	print("############ Model Accuracy -Test set ############")
	print("MAE: %0.3f" % MAE,'m/s^3')
	print("NSE: %0.3f" % NSEe)
	print("RMS: %0.3f" % RMS, 'm/s^3')
	print("R-squared: %0.3f" % RSQ)
	print("########################################")
	print(" ")
	return df_AI_Train,df_AI_Test,Y_Train_Pred

#===========================================================================#
						# Plots functions #
#===========================================================================#

def plot_training_test(df_AI_Train,df_AI_Test,Y_Train_Pred,path_to_Save):
	fig, ax1 = plt.subplots(1,1, figsize=(5,3.5))
	ax1.scatter(df_AI_Train['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				pd.Series(Y_Train_Pred[:, 0]).sort_values(ascending=True).to_numpy(), 
				marker='o', color = 'black')
	RSQ = np.round((r2_score(df_AI_Train['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				pd.Series(Y_Train_Pred[:, 0]).sort_values(ascending=True).to_numpy())),3)
	ax1.text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
			verticalalignment='bottom', horizontalalignment='right',
			transform=ax1.transAxes,
			color='black', fontsize=21)
	
	ax1.set_xlabel('Recorded', fontsize = 18)
	ax1.tick_params(axis = "x", labelsize = 14, rotation=34)
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('Predicted', fontsize = 18)
	ax1.tick_params(axis = "y", labelsize = 14)
	ax1.yaxis.set_tick_params(pad=5)
	plt.title('Training Set', fontsize = 18)
	plt.tight_layout(pad=1.2)
	fig_train = fig
	fig.savefig(path_to_Save + str('/Train_Plot_score.jpg'), format='jpeg', 
				dpi=300, bbox_inches='tight')
	
#===========================================================================#
						# XGBoost function #
#===========================================================================#
	fig, ax1 = plt.subplots(1,1, figsize=(5,3.5))
	ax1.scatter(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				marker='o', color = 'black')
	RSQ = np.round((r2_score(df_AI_Test['SF$[m^3/s]$'].sort_values(ascending=True).to_numpy(), 
				df_AI_Test['SF$^{pred} [m^3/s]$'].sort_values(ascending=True).to_numpy())),3)
	ax1.text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
			verticalalignment='bottom', horizontalalignment='right',
			transform=ax1.transAxes,
			color='black', fontsize=21)
	ax1.set_xlabel('Recorded', fontsize = 18)
	ax1.tick_params(axis = "x", labelsize = 14, rotation=34)
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('Predicted', fontsize = 18)
	ax1.tick_params(axis = "y", labelsize = 14)
	ax1.yaxis.set_tick_params(pad=5)
	plt.title('Test Set', fontsize = 18)
	plt.tight_layout(pad=1.2)
	fig_test = fig
	fig.savefig(path_to_Save + str('/Test_Plot_score.jpg'), format='jpeg', dpi=300, bbox_inches='tight')


	
#===========================================================================#
						# Plot Test results #
#===========================================================================#
def plot_test(df_AI_Test,save_Model, name):
	fig = plt.figure(figsize=(14,5),dpi=300)
	ax = fig.add_subplot(1,1,1)
	
	ax.plot(df_AI_Test['SF$[m^3/s]$'].index,
			df_AI_Test['SF$^{pred} [m^3/s]$'] , 
			'--', color = 'red', lw = 3, label = str(name))
	ax.plot(df_AI_Test['SF$[m^3/s]$'].index,
			df_AI_Test['SF$[m^3/s]$'],'--', 
			color = 'black', lw = 3, alpha=0.7, label = 'Measured')
	
	ax.set_xlabel('Data Points', fontsize = 28)
	ax.tick_params(axis = "x", labelsize = 25)
	ax.xaxis.set_tick_params(pad=5)
	ax.set_ylabel('SF$[m^3/s]$', fontsize = 36)
	
	ax.tick_params(axis = "y", labelsize = 25)
	ax.yaxis.set_tick_params(pad=5) # gap between ticks and axis
	yoffset = 0.05                    #gap between ticks and label
	################################################ 
	plt.legend(bbox_to_anchor=(0,1.01),
			   loc="lower left", ncol=3, 
			   borderaxespad=0., prop={'size': 25})
	
	plt.tight_layout(pad=1.08)
	fig.savefig(save_Model + str('/Prelim_Data_Analysis__Complete.jpg'), 
				format='jpg', dpi=300, bbox_inches = 'tight')	



#===========================================================================#
#                        Shapley analysis plot                              #
#===========================================================================#
def interaction_collage(dataframe,F,S,T,FR, save_at):
	fig = plt.figure(figsize=(36,12))
	gs = fig.add_gridspec(2, 4)
	ax1 = fig.add_subplot(gs[0, 0])
	shap.dependence_plot(F,shap_values, dataframe[Features].to_numpy(),
						 Features, ax=ax1, show=False,
						 interaction_index=F)
	ax2 = fig.add_subplot(gs[1, 0])
	shap.dependence_plot(S,shap_values, dataframe[Features].to_numpy(),
						 Features, ax=ax2, show=False,
						 interaction_index=S)
	ax3 = fig.add_subplot(gs[0, 1])
	shap.dependence_plot(T, shap_values, dataframe[Features].to_numpy(),
						 Features, ax=ax3, show=False,
						 interaction_index=T)
	ax4 = fig.add_subplot(gs[1, 1])
	shap.dependence_plot(FR, shap_values, dataframe[Features].to_numpy(),
						 Features, ax=ax4, show=False,
						 interaction_index=FR)
	plt.savefig(save_at +str('/Interaction.jpg'), format='jpeg', dpi=300,
				bbox_inches ='tight')
#===========================================================================#
#                         Projection functions                              #
#===========================================================================#


def save_all_features(Features,path_to_save):
	Pictures = {}
	for count,features in enumerate(Features, start=0):
		fig = plt.figure(figsize=(6,3.3),dpi=300)
		ax2 = fig.add_subplot(1,1,1)
		shap.dependence_plot(features,shap_values, 
							 df_AI[Features].to_numpy(),
							 Features, ax=ax2, 
							 show=False,interaction_index=features)
		plt.ylabel('SHAP value for'+'\n'+str(features),fontsize = 14)
		plt.xlabel(str(features),fontsize = 14)
		Pictures[features] = fig
		plt.savefig(path_to_save +str('/Feature Interaction_'+str(count)+'.jpg'),format='jpeg', dpi=300, bbox_inches ='tight')      
	return Pictures



def RCP(rcp_dataframe,df_AI, model, name, save_to): 
	date_rng = pd.date_range(start='1/01/2009', 
								end='01/03/2100', 
							 freq='W', closed='left')
	df_future = pd.DataFrame(index=date_rng, 
							 columns=['Month','$T_{min}$ [$^oC$]',
									  '$T_{max}$ [$^oC$]','$P$ [mm]',
									  '$P^{lag1}$ [mm]','$P^{lag2}$ [mm]',
									  'SF$^{lag1} [m^3/s]$','SF$^{lag2} [m^3/s]$','SF$[m^3/s]$'])
	df_future['Month'] = df_future.index.month
	df_future['$P$ [mm]'] = rcp_dataframe['$P$ [mm]']
	df_future['$T_{max}$ [$^oC$]'] = rcp_dataframe['$T_{max}$ [$^oC$]']
	df_future['$T_{min}$ [$^oC$]'] = rcp_dataframe['$T_{min}$ [$^oC$]']
	df_future['$P^{lag1}$ [mm]'][0] = df_AI['$P$ [mm]']['2008-12-28']
	df_future['$P^{lag2}$ [mm]'][0] = df_AI['$P$ [mm]']['2008-12-21']
	df_future['SF$^{lag1} [m^3/s]$'][0] = df_AI['SF$[m^3/s]$']['2008-12-28']
	df_future['SF$^{lag2} [m^3/s]$'][0] = df_AI['SF$[m^3/s]$']['2008-12-21']
	##########################################################################################
	future_x = df_future[['Month','$T_{min}$ [$^oC$]','$T_{max}$ [$^oC$]','$P$ [mm]',
						  '$P^{lag1}$ [mm]','$P^{lag2}$ [mm]','SF$^{lag1} [m^3/s]$',
						  'SF$^{lag2} [m^3/s]$']].iloc[0].to_numpy().reshape(1, -1)
	df_future['SF$[m^3/s]$'].iloc[0] = model.predict(future_x)[0]
	##########################################################################################
	i = 0
	for row in df_future.itertuples():
		df_future['$P^{lag1}$ [mm]'][i+1] = df_future['$P$ [mm]'].iloc[i]
		df_future['$P^{lag2}$ [mm]'][i+1] = df_future['$P^{lag1}$ [mm]'].iloc[i]
		df_future['SF$^{lag1} [m^3/s]$'][i+1] = df_future['SF$[m^3/s]$'].iloc[i]
		df_future['SF$^{lag2} [m^3/s]$'][i+1] = df_future['SF$^{lag1} [m^3/s]$'].iloc[i]
		future_x = df_future[['Month','$T_{min}$ [$^oC$]','$T_{max}$ [$^oC$]','$P$ [mm]',
							  '$P^{lag1}$ [mm]','$P^{lag2}$ [mm]','SF$^{lag1} [m^3/s]$',
							  'SF$^{lag2} [m^3/s]$']].iloc[i+1].to_numpy().reshape(1, -1)
		df_future['SF$[m^3/s]$'].iloc[i+1] = model.predict(future_x)[0]
		i = i+1
		if i == (len(date_rng)-1):   
			break
	#from out training set
	Y_future = df_AI['SF$[m^3/s]$'] \
				['2009-01-01':str(Approved_date)] \
				.sort_values(ascending=True).to_numpy()
	
	#from or trained model 
	Y_future_Pred = df_future['SF$[m^3/s]$']\
					['2009-01-01':str(Approved_date)]\
					.sort_values(ascending=True).to_numpy()
	RMS = np.sqrt(mean_squared_error(Y_future,Y_future_Pred))
	RSQ = r2_score(Y_future,Y_future_Pred)
	MAE = np.mean(np.abs((Y_future - Y_future_Pred))) 
	NSEe = he.evaluator(he.nse, Y_future_Pred.tolist(), Y_future.tolist()) 
	txt = "############ Model Validation set Accuracy RCP{rcp: }###########"
	print(txt.format(rcp =name))
	print("MAE: %0.3f" % MAE,'m/s^3')
	print("NSE: %0.3f" % NSEe)
	print("RMS: %0.3f" % RMS, 'm/s^3')
	print("R-squared: %0.3f" % RSQ)
	print("########################################")
	print(" ")
	print("########################################")
	print(" ")  
	df_future['SF$[m^3/s]$'] = df_future['SF$[m^3/s]$'].astype(float)
	df_future.to_excel(save_to+'/df_future.xlsx')
	fig, ax1 = plt.subplots(1,1, figsize=(5,3.5))
	#######################################################################################
	ax1.scatter(Y_future, 
				df_future['SF$[m^3/s]$']\
				['2009-01-01':str(Approved_date)]\
				.sort_values(ascending=True).to_numpy(),
							marker='o', color = 'black')
	RSQ = np.round((r2_score(Y_future,df_future['SF$[m^3/s]$']\
										['2009-01-01':str(Approved_date)]\
										.sort_values(ascending=True).to_numpy())),3)

	ax1.text(0.95, 0.1, ("$R^2$: %0.03f" % RSQ),
			verticalalignment='bottom', horizontalalignment='right',
			transform=ax1.transAxes,
			color='black', fontsize=21)

	ax1.set_xlabel('Recorded', fontsize = 18)
	ax1.set_ylabel('Predicted', fontsize = 18)

	plt.tight_layout(pad=1.2)
	fig.savefig(save_to + str('/Validation_plot_RCP'+str(name)+'.jpg'),
				format='jpeg', dpi=300, bbox_inches='tight')   
		
	return df_future

def plot_projections(df_future,df_AI, name, save_to):
	if name == 4.5:
		colour = '#FFA203'
	else:
		colour = '#C70B18'
	# ###################################################################################################
	# Springflow levels - Historical + Projections
	# ###################################################################################################

	fig, ax1 = plt.subplots(1,1, figsize=(16,7))
	#Plot the updated model SF with the exisitng data
	ax1.plot(df_AI.index,
				df_AI['SF$[m^3/s]$'], 
				'-', color = 'black',
				lw = 3, label = 'Historical Data')

	ax1.plot(df_future[str(Approved_date):'2100-01-03'].index,
			 df_future['SF$[m^3/s]$'][str(Approved_date):'2100-01-03'], '-', 
			 color = colour, lw = 3, label = 'Future_'+str(name)+ 'Projection')

	temp = pd.concat([df_AI['SF$[m^3/s]$'],
					  df_future['SF$[m^3/s]$']\
						  [str(Approved_date):'2100-01-03']])
	phase1 = np.zeros(len(temp))
	phase1[phase1 == 0] = 96*0.028316847
	phase2 = np.zeros(len(temp))
	phase2[phase2 == 0] = 80*0.028316847
	
	ax1.plot(temp.index, phase1, 
			 '-', color='#f3ae85', lw=5,
			 label = 'Critical Stage I')
	ax1.plot(temp.index, phase2, 
			 '-', color='#f17e53', 
			 lw=5, label = 'Critical Stage II')
	
	ax1.tick_params(axis = "x", labelsize = 36, rotation=30)
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('SF$[m^3/s]$', fontsize = 36)
	ax1.tick_params(axis = "y", labelsize = 34)
	ax1.yaxis.set_tick_params(pad=5)
	
	tw = np.array (temp.index)
	
	plt.fill_between(tw, phase2[0] , phase1[0], alpha=0.20, facecolor='#ffa579')
	plt.fill_between(tw, 0, phase2[0], alpha=0.3, facecolor='#e9683c')
	
	font = {'family': 'serif',
			'color' :  '#f3ae85', 
			'weight': 'bold', 'size': 20} 
	plt.text(tw[-1], phase1[0]-0.2, r'CS1', fontdict=font)
	
	font = {'family': 'serif', 
			'color' :  '#f17e52',
			'weight': 'bold', 'size': 20} 
	plt.text(tw[-1], phase2[0]-0.6, r'CS2', fontdict=font)

	font = {'family': 'serif', 
			'color':  colour, 
			'weight': 'bold', 'size': 28}
	plt.text(tw[-1000],phase1[0]+9.5,
			 r'RCP ' + str(name), fontdict=font)
	
	#--------------Annotate the plot -------------####
	plt.axvline(x= datetime(2020, 11, 29),
				color = 'gray', lw = 4, linestyle='--')
	font = {'family': 'serif',
			'color':  colour, 
			'weight': 'bold', 'size': 26,} 
	plt.text(tw[3350],phase1[0]+9.5, 
			 r'Projected', fontdict=font)
	font = {'family': 'serif', 
			'color':  'black', 
			'weight': 'bold', 'size': 26,} 
	plt.text(tw[0],phase1[0]+9.5,
			 r'Historical', fontdict=font)
	#-------------------------------------------- FORMAT ----------------------------
	plt.tight_layout(pad=0.6)
	fig.savefig(save_to + str('/Decremental'+str(name)+'_SF.jpg'), 
                                format='jpeg', dpi=300, bbox_inches='tight')
	fig = fig 

	return 



def MACA_data_SM(scenario,model):
	data_M_Hist  = []
	path_t = os.getcwd()+'/Maca/RCP'+str(scenario)+'/'+str(model)+'.xlsx'
	data_M_Hist  = pd.read_excel(path_t, engine = 'openpyxl')

	data_M_Hist.rename(columns = {'DATE' : 'Date',
								  'tasmin(K)' : 'Tmin[K]', 
								  'tasmax(K)' : 'Tmax[K]' , 
								  'pr(mm)' : 'Precip[mm]'}, 
								  inplace = True)

	data_M_Hist['Tmin[C]'] = (data_M_Hist['Tmin[K]'] - 273.15).astype(float)
	data_M_Hist['Tmax[C]'] = (data_M_Hist['Tmax[K]'] - 273.15).astype(float)

	data_M_Hist['Tmin[C]'].interpolate(method = 'linear', limit_direction = 'both', inplace =True)
	data_M_Hist['Tmax[C]'].interpolate(method = 'linear', limit_direction = 'both', inplace =True)
	data_M_Hist['Precip[mm]'].interpolate(method = 'linear', limit_direction = 'both', inplace =True)
	return data_M_Hist

