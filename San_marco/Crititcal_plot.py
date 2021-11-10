\

#-----------------------------------------------------------------------------------------------
#
# ------------- Forecasted Decadal Reductions in SpringFlow--------------------
# ----------------------------------------------------------------------------------------------
# -----------Using Tiered Critical Management Plan (CMP) and MACA data -------------------------
#-----------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------
#--- import libraries and packages
#-----------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import math
from wdmtoolbox import wdmtoolbox 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime   
from dateutil import relativedelta

import numpy.polynomial.polynomial as poly

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime, timedelta

from functools import reduce

from PIL import Image

from itertools import islice

import csv

import matplotlib.dates as mdates

from pandas import ExcelWriter
from pandas import ExcelFile


import os
from matplotlib import rc
import matplotlib.ticker as mtick         

# center all the figures and tables throughout the report
from IPython.display import display, HTML
display(HTML("""<style>.output {
    display: flex;
    align-items: center;
    text-align: center;}</style> """))

print (' All libraries, tools, and functions have been succesfully uploaded  ')




base_directory = os.getcwd()
Models = ['catboost_untuned']
rcp_scenerio = ["RCP45","RCP85"]


for i in Models:
    b = os.path.join(base_directory, i)
    for j in rcp_scenerio:
        k = os.path.join(b,j)


        #-------------------------------------------------------------------------------------------------------
        # J17 Groundwater Level [m]
        #-------------------------------------------------------------------------------------------------------
        df_SpringFlow_original  = pd.read_excel(k+"/df_future.xlsx",engine = "openpyxl")
        df_SpringFlow_original  = df_SpringFlow_original.rename(columns = {'Unnamed: 0' : 'Date'})
        df_SpringFlow_data      =df_SpringFlow_original[['Date','SF$[m^3/s]$' ]]
        
        
        # In[358]:
        
        
        #-------------------------------------------------------------------------------------------------------
        # remove the data before 12/31/2019 
        #-------------------------------------------------------------------------------------------------------
        
        condition_to_slice     = (df_SpringFlow_data.Date > '2020-11-29') 
        df_SpringFlow_data_reduced = []
        df_SpringFlow_data_reduced = df_SpringFlow_data[condition_to_slice]
        df_SpringFlow_data_reduced.reset_index(drop=True, inplace=True)
        #df_SpringFlow_data_reduced
        
        
        # In[359]:
        
        
        #-----------------------------------------------------------------------------------------------------------
        # start and end dates
        #------------------------------------------------------------------------------------------------------------
        
        time_start = str( df_SpringFlow_data_reduced.Date.iloc[0].floor('d')  ).rstrip(' 00:00:00')
        time_end   = str( df_SpringFlow_data_reduced.Date.iloc[-1].floor('d') ).rstrip(' 00:00:00')
        
        print('SpringFlow data at J17 starts on :', time_start)
        print('SpringFlow data at J17 ends on :', time_end)
        
        
        # In[360]:
        
        
        #-------------------------------------------------------------------------------------------------------
        # numeric values of GWL data
        #-------------------------------------------------------------------------------------------------------
        df_numeric_values_J17 = pd.to_numeric(df_SpringFlow_data_reduced['SF$[m^3/s]$' ], errors='coerce')
        df_numeric_values_J17
        
        df_SpringFlow = df_numeric_values_J17.to_frame()
        df_SpringFlow
        
        df_SpringFlow['Date'] = df_SpringFlow_data_reduced.Date
        df_SpringFlow
        
        cols = list(df_SpringFlow.columns)
        cols.insert(0, cols.pop(cols.index('Date')))
        df_SpringFlow = df_SpringFlow.loc[:, cols]
        #df_SpringFlow
        #plt.plot(df_SpringFlow)
        
        
        # In[361]:
        
        
        df_SpringFlow_data
        
        
        
        
        #---------------------------------------------------------------------------------------
        # calculate number of missing J17 gw data 
        #---------------------------------------------------------------------------------------
        
        data_len = len(df_SpringFlow)
        print('Data length             :', data_len)
        
        data_non_null = df_SpringFlow['SF$[m^3/s]$'].count()
        miss_data     = data_len - data_non_null
        print('Number missing data     :', miss_data)
        
        percent_missing_data = df_SpringFlow['SF$[m^3/s]$'].count()
        print('% missing data          :', round ( (100 * miss_data / data_len), 1) )
        
        per_miss_data = 100 * miss_data / data_len
        per_miss_data
        
        
        # In[365]:
        
        
        if per_miss_data > 1.0 :  # if <1% data is missing
            #---------------------------------------------------------------------------------------
            # impute missing data by linear interpolation - temp solution
            #---------------------------------------------------------------------------------------
            print('linear interpolation is implemented')
            df_SpringFlow['SF$[m^3/s]$'].interpolate(method = 'linear', limit_direction = 'both', inplace =True)
            data_non_null = df_SpringFlow['SF$[m^3/s]$'].count()
            data_non_null
        
        
        # In[366]:
        
        
        #----------------------------------------------------------------------------------------------------------------------
        # plot data - basic
        #----------------------------------------------------------------------------------------------------------------------
        def plot_basic(Station, df, data, fnt_size, y_ax_label, ymin, ymax, data_label):
            fig = plt.figure()
            fig      = plt.figure(figsize=(20,28))
            ax = fig.add_subplot(4,2,1)
            ax.plot(df.Date, df[data],'-', color = 'red', lw = 1.5, mfc= 'None', markersize=13, label = data_label)
        
            #ax.grid()
            ax.grid(True, which='major', axis='x' )
            ax.grid(True, which='major', axis='y' )
            ax.set_xlabel('Date', fontsize = fnt_size)
            ax.tick_params(axis = "x", labelsize = fnt_size)
            ax.xaxis.set_tick_params(pad=5)
            #ax.xaxis.set_major_locator(mdates.YearLocator())
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
            ax.set_ylabel(y_ax_label, fontsize = fnt_size), ax.set_ylim([ymin, ymax])
            ax.tick_params(axis = "y", labelsize = fnt_size)
            ax.yaxis.set_tick_params(pad=5) 
            plt.xticks(rotation = 30)
            yoffset=0.05                    
            ax.set_title(Station, fontsize = fnt_size)
            shiftx_annot =  0
            shifty_annot = 0.40        
            plt.show()
        #-------------------------------------------------------------------------------------------------------------
        # plot groundwater elevation at J-17
        #-------------------------------------------------------------------------------------------------------------
        cfs_to_m3  = 0.028316847
        fnt_size      = 20
        leg_loc       = 2
        Station       = 'USGS 08170000 San Marcos Springs at New Bruanfels'
        label         = 'San Antonio'
        y_ax_label    =  'SF$[m^3/s]$'
        y_min         = 0
        y_max         = 14 
        num_cs        = 5   # number of critical stages 
        cs1           = 96 * cfs_to_m3      # 660 # ft
        cs2           = 80 * cfs_to_m3      # 650 # ft 
        #cs3           = 150 * cfs_to_m3      # 640 # ft
        #cs4           = 100 * cfs_to_m3      # 630 # ft
        #cs5           = 45 * cfs_to_m3      # 625 # ft
        cs_line_thick = 0.1
        legend_loc    = 2

  
        #----------------------------------------------------------------------------------------------------------------------
        #
        #------------------------------------------------------------------------------------------------------
        # - genarate the begining and end of decades as a list
        #------------------------------------------------------------------------------------------------------
        
        fmt                 =  "%Y-%m-%d"
        frequency_start     = '10AS'
        frequency_end       = '10A'
        
        def date_list_generator(date_begin, num_per, frequency, fmt):
            date_list = []
            dates_raw = (pd.date_range(start = date_begin, periods = num_per, freq = frequency))
            date_list =  dates_raw.strftime(fmt).to_list()
            return date_list
        
        
        # In[372]:
        
        
        start_date_list = []
        date_start = '01-01-2020'
        no_of_periods = 8
        start_date_list = date_list_generator(date_start, no_of_periods, frequency_start, fmt) 
        print(start_date_list)
        
        end_data_list = []
        date_start = '12-31-2029'
        end_data_list = date_list_generator(date_start, no_of_periods, frequency_end, fmt) 
        print(end_data_list)
        
        #time_period_list = ['2020-2030', '2030-2040', '2040-2050', '2050-2060', '2060-2070' , '2070-2080', '2080-2090', '2090-2100'] 
        time_period_list = ['1', '2', '3', '4', '5' , '6', '7', '8']
        #print(time_period_list)
        
        
        # In[373]:
        
        
        CS_dataframe = []
        CS           = np.array([0, 1, 2, 3, 4, 5])
        CS_dataframe = pd.DataFrame({'CS' : CS}) 
        CS_dataframe
        
        #------------------------------------------------------------------------------------------------------------------
        # function that calculates % of unmet critical stages
        #------------------------------------------------------------------------------------------------------------------
        
        def CS_per_calculator (df, var, time_start, time_end, time_period, CS1, CS2 ): 
            
            date_int       = (df.Date > time_start) & (df.Date < time_end) 
            df_temp        = []
            df_temp        = df[date_int]
            df_temp.reset_index(drop = True, inplace = True)
            len_data       = len(df_temp)
            #print(len_data)
            
            condition      = df[var] >= CS1 
            Percent_no_CS  = (df.Date[date_int & condition].count() ) / len_data
            Percent_no_CS
        
            condition      = (df[var] < CS1) & (df[var] >= CS2)
            Percent_CS1    = (df.Date[date_int & condition].count() ) / len_data
            Percent_CS1    
            
            condition      = (df[var] < CS2) 
            Percent_CS2    = (df.Date[date_int & condition].count() ) / len_data
            Percent_CS2
            
                 
            per_CSs = np.array([Percent_no_CS,  Percent_CS1, Percent_CS2])
            CS_dataframe[time_period]= pd.Series(per_CSs)
            sum_col = CS_dataframe[time_period].sum()
           
            # make sure that sums up to 1.0
            print('time period :', time_period, ' sum of CS:', sum_col)   
            
            return CS_dataframe
        
        
        # In[374]:
        
        
        for i in range(len(start_date_list)):
            time_start  = start_date_list[i]
            time_end    = end_data_list[i]
            time_period = time_period_list[i]
            df = CS_per_calculator (df_SpringFlow, 'SF$[m^3/s]$', time_start, time_end, time_period, cs1, cs2) 
              
        df_SF_barblot = []
        df_SF_barblot = df.copy()
        
    
        
        # In[375]:
        
        
        #----------------------------------------------------------------------------------------------------------------------
        # bar plot subroutine
        #----------------------------------------------------------------------------------------------------------------------
        
        from matplotlib import rc
        
        #----------------------------------------------------------------------------------------------------------------------
        # bar plot subroutine
        #----------------------------------------------------------------------------------------------------------------------
        def bar_plot_CS(df, title):
        
            fig = plt.figure()
            fig.set_size_inches(12.5, 8.5)    
            fnt_size = 24
            
            # y-axis in bold
            rc('font', weight='normal')
         
            # Values of each group
            bars1 = list(df.iloc[0,1:])
            bars2 = list(df.iloc[1,1:])
            bars3 = list(df.iloc[2,1:])
            bars4 = list(df.iloc[3,1:])
            bars5 = list(df.iloc[4,1:])
            bars6 = list(df.iloc[5,1:])
            
        
            # Heights of bars(i) + bars(i+1)
            bars   = np.add(bars1, bars2).tolist()
            bars12 = np.add(bars1, bars2).tolist()
            bars23 = np.add(bars12, bars3).tolist()
            bars34 = np.add(bars23, bars4).tolist()
            bars45 = np.add(bars34, bars5).tolist()
        
            # The position of the bars on the x-axis
            r = [0,1,2,3,4, 5, 6, 7]
         
            # Names of group and bar width
            names = df.columns[1:]
            barWidth = 1
         
            # Create blue bars
            plt.bar(r, bars1, color='seashell', edgecolor='black', width=barWidth, label='No restriction')
            # ritical 1
            plt.bar(r, bars2, bottom=bars1, color='#f3ae85', edgecolor='black', width=barWidth, label='Critical Stage I')
            # Create khaki bars 
            plt.bar(r, bars3, bottom=bars12, color='#f17e52', edgecolor='black', width=barWidth, label='Critical Stage II')
            # Create coral bars
            #plt.bar(r, bars4, bottom=bars23, color='coral', edgecolor='white', width=barWidth, label='Critical Stage III')
            # Create crimson bars (top)
            #plt.bar(r, bars5, bottom=bars34, color='crimson', edgecolor='white', width=barWidth, label='Critical Stage IV')
            # Create crimson bars (top)
            #plt.bar(r, bars6, bottom=bars45, color='black', edgecolor='white', width=barWidth, label='Critical Stage V')
         
            # Custom X axis
            plt.xticks(r, names, rotation = 0, fontsize = fnt_size)
            plt.xlabel("Future Decades",  fontsize = fnt_size)
        
            plt.ylabel('Fractional Occurrences', fontsize = fnt_size)
            plt.yticks(fontsize = fnt_size)
            #plt.title(title, fontsize = fnt_size)
        
            #plt.legend(loc=(0.64, 0.03), prop={'size': fnt_size-2})
            legend = plt.legend(bbox_to_anchor=(0., 1.01, 1., 0.), handletextpad=0.5,
                            loc='lower left', ncol=3, mode="expand", borderaxespad=0,
                            numpoints=1, handlelength=1.5, fontsize = fnt_size, frameon=True)
            legend.get_frame().set_linewidth('5')
        
        
            fig.savefig(title + '.jpeg', format='jpg', dpi=300, bbox_inches = 'tight')
            
            # Show graphic
            plt.show()
        
        
        # In[376]:
        
        
        #----------------------------------------------------------------------------------------------------------------------
        # bar plot for J17
        #----------------------------------------------------------------------------------------------------------------------    



        fig_title = k+'/SF_barplot'
        bar_plot_CS(df_SF_barblot, fig_title)
        print(k)

            
            
            
            