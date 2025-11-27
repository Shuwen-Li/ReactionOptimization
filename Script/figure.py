import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def get_exp_num(data,target = 80):
    #y_values = np.mean(all_max100, axis=0)
    x_points = np.arange(len(data)) 
    f_interp = interpolate.interp1d(data, x_points, kind='linear')
    x_at_y80 = f_interp(target)
    return x_at_y80

def get_all_max100(results_all_cycle):
    all_max100=[]
    for cycle in results_all_cycle:

        tem_max100=[]
        for exp in list(range(5,51,5)):
            tem_max100.append(max(cycle[:exp]))
        if len(tem_max100)==10:
            all_max100.append(tem_max100)
    all_max100=np.array(all_max100)
    #recorded_max_mean = np.array(all_max100).mean(axis=0)
    return all_max100

def plt_figure9(ax, recorded_max_mean1=0, recorded_max_mean2=0, recorded_max_mean3=0, recorded_max_mean4=0,
                recorded_max_mean5=0, recorded_max_mean6=0, recorded_max_mean7=0, recorded_max_mean8=0, zoom_x_min=2, zoom_x_max = 4,
                zoom_y_min = 60, zoom_y_max = 90,colors = ['blue', 'green', 'olivedrab', 'darkgoldenrod', 'orange', 'coral', 'red'],plt_subfigure = True):
    """在指定坐标轴上绘制7条曲线和放大子图"""
    # 设置主图属性
    ax.set_xlabel('Experiment batch', fontsize=20)  
    means = []
    for i in [recorded_max_mean1, recorded_max_mean2, recorded_max_mean3, 
             recorded_max_mean4, recorded_max_mean5, recorded_max_mean6, recorded_max_mean7,recorded_max_mean8]:
        if np.sum(i)!=0:
            means.append(i)
    max_length = max(len(m) for m in means)
    ax.set_xticks(range(0, max_length+1,2))
    ax.tick_params(axis='both', labelsize=20)  
    ax.set_ylim([0, 100])
     
    # 绘制所有曲线
    for i, (mean, color) in enumerate(zip(means, colors)):
        ax.plot(range(1, len(mean)+1), mean, linewidth=2, c=color, label=f'序列 {i+1}')
    if plt_subfigure:
        # 在右下角创建子图（调整为更小比例）
        ax_inset = ax.inset_axes([0.4, 0.1, 0.5, 0.4])  # 调整位置和大小 [x位置, y位置, 宽度, 高度]（相对坐标0~1）
        for mean, color in zip(means, colors):
            ax_inset.plot(range(1, len(mean)+1), mean, linewidth=1, c=color)
        ax_inset.set_xlim(zoom_x_min, zoom_x_max)
        ax_inset.set_ylim(zoom_y_min, zoom_y_max)
        ax_inset.set_xticks(np.arange(zoom_x_min, zoom_x_max + 1, 1))
        ax_inset.tick_params(labelsize=15) 
        ax_inset.grid(True, alpha=0.35)
        # 在主图中标记放大区域（使用更细的线）
        ax.indicate_inset_zoom(ax_inset, edgecolor='red', linestyle='--', linewidth=1)
        
def plot_figure_with_results(data,c1='lightblue',c2='blue',title='',ylabel = '',exp_num=1,target=80,plot_value = False,x1=0.42,y1=0.06,x2=0.495,y2=0.16):
    mean_line = np.mean(data, axis=0)
    plt.title(title,fontsize=20)
    plt.xticks (fontsize=20)#, fontname='Times New Roman'
    plt.yticks (fontsize=20)
    plt.xlabel('Experiment batch',fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.ylim([0,100])
    plt.xlim([0,10])
    for recorded_max in data:
        plt.plot([tmp_item+1 for tmp_item in  list(range(len(recorded_max)))],recorded_max,linewidth=1,c=c1,alpha=0.8)
    recorded_max_mean = np.array(data).mean(axis=0)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean)))],recorded_max_mean,linewidth=3,c=c2)
    if plot_value:
        plt.text(x1, y1, f'AMY: {np.mean(data[:,exp_num]):.2f}', #Standard Deviation
                transform=plt.gca().transAxes,fontsize=20) #, bbox=dict(facecolor='white', alpha=0.8)
  
        try:
            plt.text(x2, y2, f'ABT: {get_exp_num(mean_line,target = target)+1:.2f}', #Standard Deviation
                    transform=plt.gca().transAxes,fontsize=20)#, bbox=dict(facecolor='white', alpha=0.8)
        except:
            plt.text(x2, y2, f'ABT: No', #Standard Deviation
                    transform=plt.gca().transAxes,fontsize=20)#, bbox=dict(facecolor='white', alpha=0.8)            
    plt.grid(True, alpha=0.3)
    return recorded_max_mean   

'''
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

def plt_figure1(recorded_max_mean1,recorded_max_mean2):
    plt.title('Comparation',fontsize=20)
    plt.xlabel('Experiment batch',fontsize=20)
    plt.ylim([0,100])
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean1)))],recorded_max_mean1,linewidth=3,c='blue')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean2)))],recorded_max_mean2,linewidth=3,c='red')
    print(round(recorded_max_mean1[9],1),round(recorded_max_mean2[9],1))
    
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
    
def plt_figure3(recorded_max_mean1,recorded_max_mean2,recorded_max_mean3,recorded_max_mean4):
    plt.xlabel('Experiment batch',fontsize=20)
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    plt.ylim([0,100])
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean1)))],recorded_max_mean1,linewidth=3,c='blue')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean2)))],recorded_max_mean2,linewidth=3,c='green')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean4)))],recorded_max_mean3,linewidth=3,c='orange')  
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean4)))],recorded_max_mean4,linewidth=3,c='red')   
    print(round(recorded_max_mean1[9],1),round(recorded_max_mean2[9],1),round(recorded_max_mean3[9],1),round(recorded_max_mean4[9],1))
        
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
    #print(round(recorded_max_mean1[9],1),round(recorded_max_mean2[9],1),round(recorded_max_mean3[9],1),round(recorded_max_mean4[9],1),round(recorded_max_mean5[9],1)) 
    
def plt_figure8(recorded_max_mean1,recorded_max_mean2,recorded_max_mean3,recorded_max_mean4,
                recorded_max_mean5,recorded_max_mean6,recorded_max_mean7):
    
    plt.xlabel('Experiment batch',fontsize=20)
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    plt.ylim([0,100])
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean1)))],recorded_max_mean1,linewidth=3,c='blue')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean2)))],recorded_max_mean2,linewidth=3,c='green')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean3)))],recorded_max_mean3,linewidth=3,c='olivedrab')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean4)))],recorded_max_mean4,linewidth=3,c='darkgoldenrod')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean5)))],recorded_max_mean5,linewidth=3,c='orange')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean6)))],recorded_max_mean6,linewidth=3,c='coral')  
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean7)))],recorded_max_mean7,linewidth=3,c='red')  

def plt_distribution_figure(data,mean_color = 'red',distribution_color = 'blue',ylabel='',title = '',exp_num=1,target = 80):
    mean_line = np.mean(data, axis=0)
    lower_bound = np.min(data, axis=0)  
    upper_bound = np.max(data, axis=0)  
    std_dev = np.std(data, axis=0)  # 每个x点的标准差
    x = np.arange(data.shape[1])

    #plt.figure(figsize=(x_size, y_size))
    plt.xticks(range(0, 12,2))
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    #for line in data:
        #plt.plot(x, line, color='gray', alpha=0.1, linewidth=0.5)
    plt.fill_between(x, lower_bound, upper_bound, color=distribution_color, alpha=0.05)#, label='Min-Max Range'
    plt.plot(x, mean_line, color=mean_color, linewidth=2)#, label='Mean'
    #plt.legend()
    plt.xlabel('Experiment batch',fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.title(title,fontsize=20)
    plt.text(0.20, 0.05, f'Std. Dev.: {np.mean(std_dev):.2f}', #Standard Deviation
            transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),fontsize=20)
    plt.text(0.20, 0.15, f'Avg. Yield: {np.mean(data[:,exp_num]):.2f}', #Standard Deviation
            transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),fontsize=20)
    plt.text(0.20, 0.25, f'Avg. Exp.: {get_exp_num(mean_line,target = target):.2f}', #Standard Deviation
            transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),fontsize=20)
    plt.grid(True, alpha=0.3)
    #plt.show()
    recorded_max_mean = np.array(data).mean(axis=0)
    return recorded_max_mean'''