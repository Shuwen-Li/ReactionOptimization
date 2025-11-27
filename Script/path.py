def plot_hu(x1,x2,y1,y2,c,ax):
    ax.annotate("",
                xy=(x1, y1),
                xytext=(x2, y2),
                size=2, va="center", ha="center",
                arrowprops=dict(color=c,
                                arrowstyle="simple",
                                connectionstyle="arc3,rad=0.2",
                                ))
    
def get_sort_smi(ar_ha_smi,ar_ha_smi_set,yield_value):
    ar_ha_value = [0]*len(ar_ha_smi_set)
    for i in range(len(ar_ha_smi)):
        for index,j in enumerate(ar_ha_smi_set):
            if ar_ha_smi[i] == j:
                ar_ha_value[index] += yield_value[i]
    sorted_pairs = sorted(zip(ar_ha_value, ar_ha_smi_set), reverse=True)
    sorted_ar_ha = [name for value, name in sorted_pairs]
    return sorted_ar_ha

def get_xy3(tem_exp,space):
    for index1,i in enumerate(space):
        for index2,j in enumerate(i):
            if j[0]==tem_exp[0] and j[1]==tem_exp[3] and j[2]==tem_exp[1] and j[3]==tem_exp[2]:
                return index2,index1
            
def get_xy4(tem_exp,space):
    for index1,i in enumerate(space):
        for index2,j in enumerate(i):
            if j[0]==tem_exp[0] and j[1]==tem_exp[1] and j[2]==tem_exp[2] and j[3]==tem_exp[3] and j[4]==tem_exp[4]:
                return index2,index1
            
'''def sort_2d_array(row_value,col_value,matrix):
    sorted_row_indices = sorted(range(len(row_value)), key=lambda k: row_value[k],reverse=True)
    sorted_col_indices = sorted(range(len(col_value)), key=lambda k: col_value[k],reverse=True)
    sorted_matrix = [[matrix[sorted_row_indices[row]][sorted_col_indices[col]] for col in range(len(matrix[0]))] for row in range(len(matrix))]
    return sorted_matrix, sorted_row_indices, sorted_col_indices'''