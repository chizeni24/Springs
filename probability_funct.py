from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
fmt_fiveyears = mdates.YearLocator(5)


def plot_projections(df_AI, df_future, name,store_at,Approved_date):

	if name == 4.5:
		colour = 'darkorange'
	else:
		colour = 'crimson'

	fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))
	#Plot the updated model SF with the exisitng data
	ax1.plot(df_AI.index,
          df_AI['SF$[m^3/s]$'],
          '-', color='black',
          lw=3, label='Historical Data')
	plt.ylim((0.01, 16))
	plt.xlim(('1960', '2100'))
	ax1.yaxis.set_major_locator(plt.MultipleLocator(2))
	ax1.xaxis.set_minor_locator(fmt_fiveyears)
	ax1.yaxis.set_minor_locator(plt.MultipleLocator(2))
	ax1.plot(df_future[str(Approved_date):'2100-01-03'].index,
          df_future['SF$[m^3/s]$'][str(Approved_date):], '-',
          color=colour, lw=3, label='Future_'+str(name) + 'Projection')

	temp = pd.concat([df_AI['SF$[m^3/s]$'],
                   df_future['SF$[m^3/s]$'][str(Approved_date):]])

	phase1 = np.zeros(len(temp))
	phase1[phase1 == 0] = 225*0.028316847
	phase2 = np.zeros(len(temp))
	phase2[phase2 == 0] = 200*0.028316847
	phase3 = np.zeros(len(temp))
	phase3[phase3 == 0] = 150*0.028316847
	phase4 = np.zeros(len(temp))
	phase4[phase4 == 0] = 100*0.028316847
	phase5 = np.zeros(len(temp))
	phase5[phase5 == 0] = 45*0.028316847

	ax1.plot(temp.index, phase1,
          '-', color='#f3ae85',
          lw=5, label='Critical Stage I')
	ax1.plot(temp.index, phase2,
          '-', color='#f17e52',
          lw=5, label='Critical Stage II')
	ax1.plot(temp.index, phase3,
          '-', color='#dc5328',
          lw=5, label='Critical Stage III')
	ax1.plot(temp.index, phase4,
          '-', color='#b63609',
          lw=5, label='Critical Stage IV')
	ax1.plot(temp.index, phase5,
          '-', color='#420D09',
          lw=5, label='Critical Stage V')

	ax1.set_xlabel('', fontsize=36)
	ax1.tick_params(axis="x", labelsize=26, )
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('SF$[m^3/s]$', fontsize=36, labelpad=15)
	ax1.tick_params(axis="y", labelsize=26, )
	ax1.yaxis.set_tick_params(pad=5)

	tw = np.array(temp.index)
	plt.fill_between(tw, phase2[0], phase1[0],
                  alpha=0.20, facecolor='#ffa579')
	plt.fill_between(tw, phase2[0], phase3[0],
                  alpha=0.2, facecolor='#e9683c')
	plt.fill_between(tw, phase4[0], phase3[0],
                  alpha=0.20, facecolor='#cc4217')
	plt.fill_between(tw, phase5[0], phase4[0],
                  alpha=0.20, facecolor='#993404')
	plt.fill_between(tw,   phase5[0], phase5[0]-3,
                  alpha=0.2, facecolor='#420D09')

	font = {'family': 'serif', 'color':  '#f3ae85', 'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase1[0]-0.2, r'CS1', fontdict=font)
	font = {'family': 'serif', 'color':  '#f17e52', 'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase2[0]-0.2, r'CS2', fontdict=font)
	font = {'family': 'serif', 'color':  '#dc5328', 'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase3[0]-0.2, r'CS3', fontdict=font)
	font = {'family': 'serif', 'color':  '#b63609', 'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase4[0]-0.2, r'CS4', fontdict=font)
	font = {'family': 'serif', 'color':  '#420D09', 'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase5[0]-0.2, r'CS5', fontdict=font)
	font = {'family': 'serif', 'color':  colour, 'weight': 'bold', 'size': 28}
	plt.text(tw[-1000], phase1[0]+8.5, r'RCP'+str(name), fontdict=font)

	#--------------Annotate the plot -------------####
	plt.axvline(x=datetime(2020, 11, 29),
             color='gray', lw=4, linestyle='--')
	font = {'family': 'serif',
         'color':  colour,
         'weight': 'bold', 'size': 26, }
	plt.text(tw[3500], phase1[0]+8.5,
          r'Projected', fontdict=font)
	font = {'family': 'serif',
         'color':  'black',
         'weight': 'bold', 'size': 26, }
	plt.text(tw[200], phase1[0]+8.5,
          r'Historical', fontdict=font)
	plt.tight_layout(pad=0.6)

	fig.savefig(os.getcwd()+store_at+r'/Decremental_SF.jpg', format='jpeg',
	            dpi=300, bbox_inches='tight')
	fig = fig
	return fig







def plot_projections(df_AI, df_future, name, store_at, Approved_date):

	if name == 4.5:
		colour = 'darkorange'
	else:
		colour = 'crimson'
	# ###################################################################################################
	# Springflow levels - Historical + Projections
	# ###################################################################################################

	fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))
	#Plot the updated model SF with the exisitng data
	ax1.plot(df_AI.index,
          df_AI['SF$[m^3/s]$'],
          '-', color='black',
          lw=3, label='Historical Data')

	plt.ylim((0.01, 15))
	plt.xlim(('1960', '2100-05'))
	ax1.yaxis.set_major_locator(plt.MultipleLocator(2))
	ax1.xaxis.set_minor_locator(fmt_fiveyears)
	ax1.yaxis.set_minor_locator(plt.MultipleLocator(2))
	ax1.plot(df_future[str(Approved_date):'2100-01-03'].index,
          df_future['SF$[m^3/s]$'][str(Approved_date):], '-',
          color=colour, lw=3, label='Future_'+str(name) + 'Projection')
	temp = pd.concat([df_AI['SF$[m^3/s]$'],
                   df_future['SF$[m^3/s]$']
                   [str(Approved_date):'2100-01-03']])
				   


	phase1 = np.zeros(len(temp))
	phase1[phase1 == 0] = 96*0.028316847
	phase2 = np.zeros(len(temp))
	phase2[phase2 == 0] = 80*0.028316847

	ax1.plot(temp.index, phase1,
          '-', color='#f3ae85', lw=5,
          label='Critical Stage I')
	ax1.plot(temp.index, phase2,
          '-', color='#f17e53',
          lw=5, label='Critical Stage II')

	ax1.set_xlabel('', fontsize=36)
	ax1.tick_params(axis="x", labelsize=26, )
	ax1.xaxis.set_tick_params(pad=5)
	ax1.set_ylabel('SF$[m^3/s]$', fontsize=36, labelpad=15)
	ax1.tick_params(axis="y", labelsize=26, )
	ax1.yaxis.set_tick_params(pad=5)

	tw = np.array(temp.index)

	plt.fill_between(tw, phase2[0], phase1[0], alpha=0.20, facecolor='#ffa579')
	plt.fill_between(tw, 0, phase2[0], alpha=0.3, facecolor='#e9683c')

	font = {'family': 'serif',
         'color':  '#f3ae85',
         'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase1[0]-0.2, r'CS1', fontdict=font)

	font = {'family': 'serif',
         'color':  '#f17e52',
         'weight': 'bold', 'size': 20}
	plt.text(tw[-1], phase2[0]-0.6, r'CS2', fontdict=font)

	font = {'family': 'serif',
         'color':  colour,
         'weight': 'bold', 'size': 28}

	h = 11
	plt.text(tw[-1000], phase1[0]+h,
			 r'RCP'+str(name), fontdict=font)


	#--------------Annotate the plot -------------####
	plt.axvline(x=datetime(2020, 11, 29),
             color='gray', lw=4, linestyle='--')
	font = {'family': 'serif',
         'color':  colour,
         'weight': 'bold', 'size': 26, }
	plt.text(tw[3500], phase1[0]+h,
          r'Projected', fontdict=font)
	font = {'family': 'serif',
         'color':  'black',
         'weight': 'bold', 'size': 26, }
	plt.text(tw[200], phase1[0]+h,
          r'Historical', fontdict=font)
	plt.tight_layout(pad=0.6)

	#-------------------------------------------- FORMAT ----------------------------
	plt.tight_layout(pad=0.6)
	fig.savefig(os.getcwd()+store_at+r'/Decremental_SF.jpg', format='jpeg',
	            dpi=300, bbox_inches='tight')
	fig = fig;
	return
