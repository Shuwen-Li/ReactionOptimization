import matplotlib.pyplot as plt
import numpy as np

def get_all_max100(results_all_cycle):
    all_max100=[]
    for cycle in results_all_cycle:

        tem_max100=[]
        for exp in list(range(5,51,5)):
            tem_max100.append(max(cycle[:exp]))
        if len(tem_max100)==10:
            all_max100.append(tem_max100)
    all_max100=np.array(all_max100)
    recorded_max_mean = np.array(all_max100).mean(axis=0)
    return all_max100

def plot_figure(all_max100,c1='lightblue',c2='blue',title='',ylabel = ''):
    plt.title(title,fontsize=20)
    plt.xticks (fontsize=20)#, fontname='Times New Roman'
    plt.yticks (fontsize=20)
    plt.xlabel('Experiment batch',fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.ylim([0,100])
    plt.xlim([0,10])
    for recorded_max in all_max100:
        plt.plot([tmp_item+1 for tmp_item in  list(range(len(recorded_max)))],recorded_max,linewidth=1,c=c1,alpha=0.8)
    recorded_max_mean = np.array(all_max100).mean(axis=0)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean)))],recorded_max_mean,linewidth=3,c=c2)
    return recorded_max_mean

def plt_figure2(recorded_max_mean1,recorded_max_mean2,recorded_max_mean3):
    plt.title('',fontsize=20)
    plt.xlabel('Experiment batch',fontsize=20)
    plt.ylim([0,100])
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean1)))],recorded_max_mean1,linewidth=3,c='blue')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean2)))],recorded_max_mean2,linewidth=3,c='green')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean3)))],recorded_max_mean3,linewidth=3,c='red')
    print(round(recorded_max_mean1[9],1),round(recorded_max_mean2[9],1),round(recorded_max_mean3[9],1))
    
def plt_figure4(recorded_max_mean1,recorded_max_mean2,recorded_max_mean3,recorded_max_mean4,recorded_max_mean5):
    plt.xlabel('Experiment batch',fontsize=20)
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    plt.ylim([0,100])
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean1)))],recorded_max_mean1,linewidth=3,c='blue')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean2)))],recorded_max_mean2,linewidth=3,c='green')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean3)))],recorded_max_mean3,linewidth=3,c='olivedrab')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean4)))],recorded_max_mean4,linewidth=3,c='orange')  
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean5)))],recorded_max_mean5,linewidth=3,c='red')   
    print(round(recorded_max_mean1[9],1),round(recorded_max_mean2[9],1),round(recorded_max_mean3[9],1),round(recorded_max_mean4[9],1),round(recorded_max_mean5[9],1)) 