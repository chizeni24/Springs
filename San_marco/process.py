import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import tabulate
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

import random
import pandas as pd

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.model_selection import KFold,cross_val_score, TimeSeriesSplit
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.model_selection import KFold,cross_val_score, TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from prettytable import PrettyTable
from sklearn.pipeline import Pipeline
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer
import time
import numpy as np
import datetime
import os
from calendar import isleap
########################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew, kurtosis
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
from itertools import groupby
# to avoid error warnings 'implicitly registered datetime coverter'
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# to center figures and tables throughout the report
from IPython.display import display, HTML
display(HTML("""<style>.output {
	display: flex;
	align-items: center;
	text-align: center;}</style> """))
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import datetime
import folium 
from sklearn.ensemble import ExtraTreesRegressor
## Import all the libraries
## Import all the modules 
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse,r2_score
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import tabulate
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer 
import random
import pandas as pd

# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer
import time
import numpy as np
import datetime
import os
from calendar import isleap
########################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew, kurtosis
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
########################################################################################
# to read directories
import os
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.model_selection import KFold,cross_val_score, TimeSeriesSplit
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.model_selection import KFold,cross_val_score, TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from prettytable import PrettyTable
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from IPython.display import display, HTML
display(HTML("""<style>.output {
    display: flex;
    align-items: center;
    text-align: center;}</style> """))
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import datetime
import folium 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold,cross_val_score, TimeSeriesSplit



# to avoid error warnings 'implicitly registered datetime coverter'

# to center figures and tables throughout the report

def status(df):
	table = [[i,
	  len(df[i]), df[i].isna().sum(),
	  "{:.1%}".format(df[i].isna().sum()/len(df[i]))]
	 for i in df.columns] 
	headers  = ['Features','Observations', 'No of missing','% Missing ']
	print(tabulate(table, headers, tablefmt = 'pretty', numalign= 'center'))
#----------------------------------------------------------------------------------
# compare two data sets  
#----------------------------------------------------------------------------------

def compare_methods_against(ax, df, data1, data2, y_min, y_max, x_label, y_label, fnt_size):

	df_straightline = pd.DataFrame ({'x' : (y_min, y_max), \
									 'y' : (y_min, y_max)      }) 

	ax.plot(df[data1], df[data2] , 'bo', mfc='none', markersize = 12)
	ax.plot(df_straightline.x, df_straightline.y , '--r', mfc='none', markersize = 12, lw = 2)

	ax.set_xlabel(x_label, fontsize = fnt_size)
	ax.set_xlim([y_min, y_max])
	ax.tick_params(axis = "x", labelsize = fnt_size)
	ax.xaxis.set_tick_params(pad=5)
	ax.xaxis.set_tick_params(pad=5)  

	ax.set_ylabel(y_label, fontsize = fnt_size)
	ax.set_ylim([ y_min, y_max ])
	ax.tick_params(axis = "y", labelsize = fnt_size)
	ax.yaxis.set_tick_params(pad=5)   

	corr_2_method = df.corr(method = 'pearson')
	corr_strength = round(corr_2_method[data1][0], 2)
	corr_strength
	annot_text = 'R$^2$ = ' + str(corr_strength) 
	ax.annotate(annot_text, xy=(0.01 * y_max, 0.85 * y_max), fontsize = fnt_size )
	
	RMSE_val  = round (np.sqrt (mean_squared_error(df[data1], df[data2])), 3)
	
	annot_text = '$RMSE$ = ' + str(round(RMSE_val,2)) + ''
	ax.annotate(annot_text, xy=(0.01 * y_max, 0.73 * y_max), fontsize = fnt_size )
	
plt.show()

def get_missing_sequence(df_SM_NOAA):
	
	"""Obtain the largest missing sequence we have"""
	
	df =df_SM_NOAA
	df = df.fillna(-999)
	"""Create the table of the missing range"""
	table = [[v.index[0],
			  v.index[-1],len(v)]for k, v in df[df['PRCP'] == -999]
			 .groupby((df['PRCP'] != -999).cumsum())]


	df_missing = pd.DataFrame(table, columns=['start_Date','End_Date','Frequency'])
	df_missing.sort_values(by = ['Frequency'] , ascending= False).head(20)
	
def print_results(model, X_train,y_train,X_test,y_test):
## prints results of models.
	Y_train_pred = model.predict(X_train)
	
	rmse = np.sqrt(mean_squared_error(y_train,Y_train_pred))
	RSQ = r2_score(y_train,Y_train_pred)
	MAPE = np.mean(np.abs((y_train - Y_train_pred) / y_train)) * 100
	MAE = np.mean(np.abs(y_train - Y_train_pred))
	print("############ Model Train Accuracy ############")
	print("RMSE: %0.3f" % rmse,"mm.")
	print("MAE: %0.3f" % MAE,"mm.")
	print("R-squared: %0.3f" % RSQ)
	print("MAPE: %0.3f" % MAPE)
	print("########################################")
	
	Y_Test_Pred = model.predict(X_test)
	
	rmse = np.sqrt(mean_squared_error(y_test,Y_Test_Pred))
	RSQ = r2_score(y_test,Y_Test_Pred)
	MAPE = np.mean(np.abs((y_test - Y_Test_Pred) / y_test)) * 100
	MAE = np.mean(np.abs(y_test - Y_Test_Pred))
	print("############ Model Test Accuracy  ############")
	print("RMSE: %0.3f" % rmse,"mm.")
	print("MAE: %0.3f" % MAE,"mm.")
	print("R-squared: %0.3f" % RSQ)
	print("MAPE: %0.3f" % MAPE)
	print("########################################")	


def GAA_tuned(Model, objective_parameters = [
		{'name' : 'n_estimators',
		 'bounds' : [100,1000],
		 'type' : 'int'},
		{'name' : 'max_depth',   
		 'bounds' : [6,12],
		 'type' : 'int'} ]):
		

	
	
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
		scoring_param = r2_score
		g = cross_val_score(clf, X_train, y_train.ravel(), cv=3, scoring='r2') 
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
	#################################################################################################
	
	print_results(model,X_train,y_train)
	

def join_first_df(df1, df2):
    df = pd.concat([df1,df2], axis=1, join='inner')
    return df


def results(x,y,x_t,y_t,pipelines):
	table = PrettyTable()

	# Fit the pipelines
	[pipe.fit(x, y) for pipe in pipelines]

	pipe_dict = {0: 'RandomForest', 1: 'XGBoost', 2: 'ExtraTree'}
	
	results = [[pipe_dict[i],"{:.2%}".format(r2_score(model.predict(x), y))]
	           for i,model in enumerate(pipelines)]
	table.title = 'Training set Score'
	table.field_names= ['Algorithm','R-square']
	table.add_rows(results)

	print(table)

	table = PrettyTable()

	results2 = [[pipe_dict[i],"{:.2%}".format(r2_score(model.predict(x_t), y_t))]
	           for i,model in enumerate(pipelines)]
	table.title = 'Test set Score'
	table.field_names= ['Algorithm','R-square']
	table.add_rows(results2)
	
	
	print(table)

def pipeline(Tmax,df_PRISM):
	# call the concate
	df = join_first_df(Tmax,df_PRISM)

	X = df.iloc[:,-1:]
	y = df.iloc[:,:-1]

	
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
	
	
	pipeline_RF = Pipeline([('RandomForest', RandomForestRegressor())])
	pipeline_XGB = Pipeline([('XGBoost', XGBRegressor())])
	pipeline_ExTree = Pipeline([('ExtraTree', ExtraTreesRegressor())])
	pipelines = [pipeline_RF,pipeline_XGB,pipeline_ExTree]
	
	
	
	results(X_train,y_train,X_test,y_test,pipelines)
	

	return df,pipelines