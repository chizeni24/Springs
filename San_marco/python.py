



cfs_to_m3  = 0.028316847

cs1 = 225 * cfs_to_m3      # 225 # ft
cs2 = 200 * cfs_to_m3      # 200 # ft 
cs3 = 150 * cfs_to_m3      # 150 # ft
cs4 = 100 * cfs_to_m3      # 100 # ft
cs5 = 45  * cfs_to_m3      # 45  # ft



true_pred = df_history[(df_history['$T_{max}$ [$^oC$]']>30) & 
				  (df_history['SF$[m^3/s]$']<(cs1))]['SF$[m^3/s]$'].count()
				  
total_cases = df_history[(df_history['$T_{max}$ [$^oC$]']>30)]['SF$[m^3/s]$'].count()
print(np.round((100 * true_pred/total_cases),1))

true_pred = df_history[(df_history['$T_{max}$ [$^oC$]']>30) & 
				  (df_history['SF$[m^3/s]$']<(cs2))]['SF$[m^3/s]$'].count()
total_cases = df_history[
					(df_history['$T_{max}$ [$^oC$]']>30)]['SF$[m^3/s]$'].count()
print(np.round((100 * true_pred/total_cases),1))

true_pred = df_history[(df_history['$T_{max}$ [$^oC$]']>30) & 
				  (df_history['SF$[m^3/s]$']<(cs3))]['SF$[m^3/s]$'].count()
total_cases = df_history[
					(df_history['$T_{max}$ [$^oC$]']>30)]['SF$[m^3/s]$'].count()
print(np.round((100 * true_pred/total_cases),1))

true_pred = df_history[(df_history['$T_{max}$ [$^oC$]']>30) & 
				  (df_history['SF$[m^3/s]$']<(cs4))]['SF$[m^3/s]$'].count()
total_cases = df_history[
					(df_history['$T_{max}$ [$^oC$]']>30)]['SF$[m^3/s]$'].count()
print(np.round((100 * true_pred/total_cases),1))

true_pred = df_history[(df_history['$T_{max}$ [$^oC$]']>30) & 
				  (df_history['SF$[m^3/s]$']<(cs5))]['SF$[m^3/s]$'].count()
total_cases = df_history[
					(df_history['$T_{max}$ [$^oC$]']>30)]['SF$[m^3/s]$'].count()
print(np.round((100 * true_pred/total_cases),1))






def temp(df,cs):
	for i in cs :
		true_pred = df_history[(df_history['$T_{max}$ [$^oC$]']>30)\
						& (df_history['SF$[m^3/s]$']<(cs[i]))]['SF$[m^3/s]$'].count()

		total_cases = df_history[(df_history['$T_{max}$ [$^oC$]']>30)]\
											['SF$[m^3/s]$'].count()
		print("Probability of"+ str(i) : {}".format(np.round((100 * true_pred/total_cases),1)))



	
def plot_projections(df_future,df_AI, name, save_to)


Extratree_base = '/Comal/Extratree_tuned_/'
XGBoost_base = '/Comal/XGBoost_tuned_/'

Historical_path = 'df_AI.xlsx'
df_history = read_file(Extratree_base+Historical_path)
## Load the RCP45 dataframe
RCP_45 = '/RCP45/df_future.xlsx'
#df_RCP45 = read_file(RCP_45)
## Load the RCP85 dataframe
RCP_85 = '/RCP45/df_future.xlsx'
#df_RCP85 = read_file(RCP_85)
### Image of Training,Testing adn model importance
train_plot = Image.open(os.getcwd()+base'/Train_Plot_score.jpg')
test_plot = Image.open(os.getcwd()+'/Comal/Extratree_tuned_/Test_Plot_score.jpg') 
important_feature = Image.open(os.getcwd()+'/Comal/Extratree_tuned_/Features_importance/Feature Importances-Hist.jpg')


# XGBoost
Historical_path = '/Comal/XGBoost_tuned_/df_AI.xlsx'
df_history_xg = read_file(Historical_path)
## Load the RCP45 dataframe
RCP_45 = '/Comal/XGBoost_tuned_/RCP45/df_future.xlsx'
df_RCP45_xg = read_file(RCP_45)
## Load the RCP85 dataframe
RCP_85 = '/Comal/XGBoost_tuned_/RCP45/df_future.xlsx'
#df_RCP85_xg = read_file(RCP_85)
### Image of Training,Testing adn model importance
train_plot_xg = Image.open(os.getcwd()+'/Comal/XGBoost_tuned_/Train_Plot_score.jpg')
test_plot_xg = Image.open(os.getcwd()+'/Comal/XGBoost_tuned_/Test_Plot_score.jpg')
important_feature_xg = Image.open(os.getcwd()+'/Comal/XGBoost_tuned_/Features_importance/Feature Importances - Hist.jpg')







# load and show an image with Pillow
from PIL import Image
import matplotlib.pyplot as plt
# load the image
image_1 = Image.open(os.getcwd()+XGBoost_base+'/Features_importance/SF_Lag1 Hist.jpg')
image_2 = Image.open(os.getcwd()+XGBoost_base+'/Features_importance/Prlag1 -.jpg')
image_3 = Image.open(os.getcwd()+XGBoost_base+'/Features_importance/Tmax - Hist.jpg')
image_4 = Image.open(os.getcwd()+XGBoost_base+'/Features_importance/Pr -.jpg')






























true_pred = np.array()



name = 'orange'

fig, ax1 = plt.subplots(1,1, figsize=(16,7))
#Plot the updated model SF with the exisitng data
ax1.plot(df_history.index,
			df_history["$P$ [mm]"], 
			'-', color = 'black',
			lw = 3, label = 'Historical Data')
plt.ylim((0.01,16))
plt.xlim(('1960','2100'))
ax1.yaxis.set_major_locator(plt.MultipleLocator(4))
ax1.xaxis.set_minor_locator(fmt_fiveyears)
ax1.yaxis.set_minor_locator(plt.MultipleLocator(2))
ax1.plot(df_RCP45[str(Approved_date):'2100-01-03'].index,
		 df_RCP45['$P$ [mm]'][(Approved_date):], '-', 
		 color = colour, lw = 3, label = 'Future_'+str(name)+ 'Projection')


temp = pd.concat([df_history['$P$ [mm]'],
					 df_RCP45['$P$ [mm]'][(Approved_date):]])
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
		 lw=5, label = 'Critical Stage I')
ax1.plot(temp.index, phase2,
		'-', color='#f17e52',			
		 lw=5, label = 'Critical Stage II')
ax1.plot(temp.index, phase3, 
		'-', color='#dc5328',
		lw=5, label = 'Critical Stage III')
ax1.plot(temp.index, phase4, 
		'-', color='#b63609',
		 lw=5, label = 'Critical Stage IV')
ax1.plot(temp.index, phase5, 
		'-', color='black',
		 lw=5, label = 'Critical Stage V')

ax1.set_xlabel('', fontsize = 36)
ax1.tick_params(axis = "x", labelsize = 26, )
ax1.xaxis.set_tick_params(pad=5)
ax1.set_ylabel('SF$[m^3/s]$', fontsize = 36,labelpad=15)
ax1.tick_params(axis = "y", labelsize = 26, )
ax1.yaxis.set_tick_params(pad=5)

tw = np.array (temp.index)
plt.fill_between(tw, phase2[0],phase1[0],
				 alpha=0.20, facecolor='#ffa579')
plt.fill_between(tw,phase2[0],phase3[0],
				 alpha=0.3, facecolor='#e9683c')
plt.fill_between(tw, phase4[0] , phase3[0], 
				 alpha=0.20, facecolor='#cc4217')
plt.fill_between(tw, phase5[0] , phase4[0], 
				 alpha=0.20, facecolor='#993404')
plt.fill_between(tw,   phase5[0],phase5[0]-3, 
				 alpha=0.15, facecolor='black')


font = {'family': 'serif', 'color':  '#f3ae85', 'weight': 'bold', 'size': 20} 
plt.text(tw[-1], phase1[0]-0.2, r'CS1', fontdict=font)
font = {'family': 'serif', 'color':  '#f17e52', 'weight': 'bold', 'size': 20} 
plt.text(tw[-1], phase2[0]-0.6, r'CS2', fontdict=font)
font = {'family': 'serif', 'color':  '#dc5328', 'weight': 'bold', 'size': 20} 
plt.text(tw[-1], phase3[0]-0.6, r'CS3', fontdict=font)
font = {'family': 'serif', 'color':  '#b63609', 'weight': 'bold', 'size': 20}
plt.text(tw[-1], phase4[0]-0.6, r'CS4', fontdict=font)
font = {'family': 'serif', 'color':  'black', 'weight': 'bold', 'size': 20}
plt.text(tw[-1], phase5[0]-0.6, r'CS5', fontdict=font)
font = {'family': 'serif', 'color':  colour, 'weight': 'bold', 'size': 28}
plt.text(tw[-1000],phase1[0]+8.5, r'RCP'+str(name), fontdict=font)
#--------------Annotate the plot -------------####
plt.axvline(x= datetime(2020, 11, 29),
			color = 'gray', lw = 4, linestyle='--')
font = {'family': 'serif',
		'color':  colour, 
		'weight': 'bold', 'size': 26,} 
plt.text(tw[3500],phase1[0]+8.5, 
		 r'Projected', fontdict=font)
font = {'family': 'serif', 
		'color':  'black', 
		'weight': 'bold', 'size': 26,} 
plt.text(tw[200],phase1[0]+8.5,
		 r'Historical', fontdict=font)
plt.tight_layout(pad=0.6)











'''

ExtraTree comal springs 

ax4 = plt.subplot2grid((2,2), (0,0),)
ax4.imshow(image_1)
ax4.text(10,80,'D', fontsize= 35,fontweight='medium', )
ax4.set_axis_off()
ax4.hlines(y = 380, xmin= 250, xmax =1000,
		   linestyles = "--", color = 'black', lw = 1.5)
ax4.vlines(x = 1000, ymin= 780, ymax =380,
		   linestyles = "--", color = 'black', lw = 1.5)

ax5 = plt.subplot2grid((2,2), (0,1),)
#ax2.text(270,150, 'Testing', fontsize = 35)
ax5.text(10,80,'E', fontsize= 35,fontweight='medium', )
ax5.imshow(image_3)
ax5.set_axis_off()
ax5.hlines(y = 380, xmin= 250, xmax =1000,
		   linestyles = "--", color = 'black', lw = 1.5)
ax5.vlines(x = 1000, ymin= 780, ymax =380,
		   linestyles = "--", color = 'black', lw = 1.5)

plt.tight_layout(pad =0.75)
plt.savefig(os.getcwd()+'/Comal/Extra_results1.jpg',format='jpeg', dpi=300, bbox_inches ='tight')


#fig = plt.figure(figsize=(24,24))
ax6 = plt.subplot2grid((2,2), (1,0))
#ax2.text(270,150, 'Testing', fontsize = 35)
ax6.text(10,80,'F', fontsize= 35,fontweight='medium', )
ax6.imshow(image_2)
ax6.hlines(y = 657, xmin= 300, xmax =420,
		   linestyles = "--", color = 'black', lw = 1.5)
ax6.vlines(x = 420, ymin= 653, ymax =776,
		   linestyles = "--", color = 'black', lw = 1.5)

ax6.set_axis_off()

ax7 = plt.subplot2grid((2,2), (1,1))
ax7.text(10,80,'G', fontsize= 35,fontweight='medium', )
ax7.imshow(image_4)
ax7.hlines(y = 710, xmin= 250, xmax =400,
		   linestyles = "--", color = 'black', lw = 1.5)
ax7.vlines(x = 400, ymin= 710, ymax =790,
		   linestyles = "--", color = 'black', lw = 1.5)
#ax7.grid()
ax7.set_axis_off()
plt.tight_layout(pad =0.75)


XGBoost Comal

ax4 = plt.subplot2grid((2,2), (0,0),)
ax4.imshow(image_1)
ax4.text(10,80,'D', fontsize= 35,fontweight='medium', )
ax4.hlines(y = 362, xmin= 270, xmax =1000,
		   linestyles = "--", color = 'black', lw = 1.5)
ax4.vlines(x = 1000, ymin= 780, ymax =365,
		   linestyles = "--", color = 'black', lw = 1.5)
ax4.set_axis_off()

ax5 = plt.subplot2grid((2,2), (0,1),)
ax5.imshow(image_2)
ax5.text(10,80,'E', fontsize= 35,fontweight='medium', )
ax5.hlines(y = 632, xmin= 275, xmax =400,
		   linestyles = "--", color = 'black', lw = 1.5)
ax5.vlines(x = 400, ymin= 770, ymax =632,
		   linestyles = "--", color = 'black', lw = 1.5)
ax5.set_axis_off()

ax6 = plt.subplot2grid((2,2), (1,0))
#ax2.text(270,150, 'Testing', fontsize = 35)
ax6.text(10,80,'F', fontsize= 35,fontweight='medium', )
ax6.imshow(image_3)
ax6.hlines(y = 353, xmin= 308, xmax =1250,
		   linestyles = "--", color = 'black', lw = 1.5)
ax6.vlines(x = 1250, ymin= 780, ymax =352,
		   linestyles = "--", color = 'black', lw = 1.5)

ax6.set_axis_off()


ax7 = plt.subplot2grid((2,2), (1,1))
ax7.text(10,80,'G', fontsize= 35,fontweight='medium', )
ax7.imshow(image_4)
ax7.hlines(y = 665, xmin= 260, xmax =400,
		   linestyles = "--", color = 'black', lw = 1.5)
ax7.vlines(x = 400, ymin= 770, ymax =665,
		   linestyles = "--", color = 'black', lw = 1.5)
ax7.grid()
ax7.set_axis_off()
plt.tight_layout(pad =0.75)

plt.subplots_adjust(hspace=-0.6)
plt.savefig(os.getcwd()+XGBoost_base+'/Feature_importance_collage.jpg',

'''


