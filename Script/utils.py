import pandas as pd
import numpy as np
import shutil
import os
import glob
from copy import deepcopy
from itertools import product
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,BaggingRegressor
import torch
from autosklearn.automl import AutoML
from autosklearn.constants import REGRESSION
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def get_domain(defined_chemical_space):
    domain_list = [tmp_combine for tmp_combine in product(*[defined_chemical_space[tmp_key] for tmp_key in defined_chemical_space])]
    domain = pd.DataFrame.from_dict({tmp_category:[domain_list[i][idx] for i in range(len(domain_list))] \
                                     for idx,tmp_category in enumerate(defined_chemical_space)})
    return domain
def getdescdomain(domain,desc_map,defined_chemical_space):
    df=pd.DataFrame([np.concatenate([desc_map[domain.iloc[i][idx]] for idx,tmp_category \
                                       in enumerate(defined_chemical_space)]) for i in range(len(domain))])
    numeric_columns = df.select_dtypes(include='number')
    new_df = pd.DataFrame(numeric_columns)
    return new_df

def random_recom(batch_size,domain,desc_domain,init_pth,random_state=None,target = 'yield',external_init=False):

    if external_init==True and random_state < 50:
        exp_idx = pd.read_csv('/home/hx-gpu3/Jupyter.dir/LSW/Bayesian3/bayesian_results/cn_seed_%d.csv'%(random_state))['Unnamed: 0'].to_list()[:5]
    elif external_init=='low':
        exp_idx = np.array([0,1,2,3,4]) + random_state*5
    elif external_init=='low_rate10':
        np.random.seed(random_state)          
        exp_idx = np.random.randint(0,int(3995*0.1),batch_size) 
    else:
        np.random.seed(random_state)          
        exp_idx = np.random.randint(0,len(domain),batch_size)       
    init_react = domain.iloc[exp_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_react[target] = ['<Enter the result>']*batch_size
    init_desc = desc_domain.iloc[exp_idx]
    init_react.to_csv(init_pth)
    return init_react

def exe_exp(domain_sampled,result_pth='./Data/data_cn/experiment_index.csv'):
    exp_result = pd.read_csv(result_pth)
    arha_smi = domain_sampled['Aryl_halide_SMILES'].to_list()
    add_smi = domain_sampled['Additive_SMILES'].to_list()
    base_smi = domain_sampled['Base_SMILES'].to_list()
    lig_smi = domain_sampled['Ligand_SMILES'].to_list()
    result = []
    for i in range(len(arha_smi)):
        try:
            tmp_targ = float(exp_result[(exp_result['Aryl_halide_SMILES'] == arha_smi[i]) &\
                          (exp_result['Additive_SMILES'] == add_smi[i]) &\
                          (exp_result['Base_SMILES'] == base_smi[i]) &\
                          (exp_result['Ligand_SMILES'] == lig_smi[i])]['yield'])
        except:
            tmp_targ = np.nan
        result.append(tmp_targ)
    return np.array(result,dtype=np.float32)

def add_result(result,new_result_pth='',new_result_pd=None):
    if new_result_pth != '':
        new_result = pd.read_csv(new_result_pth,index_col=0)
        return result._append(new_result).dropna(axis=0)
    else:
        return result._append(new_result_pd).dropna(axis=0)
def result2xy(desc_domain,result_pth='',result=None,scale=0.01,target = 'yield'):
    if result_pth != '':
        result = pd.read_csv(result_pth,index_col=0)
    
    exp_idx = [int(i) for i in result.index]
    train_x = torch.tensor(desc_domain.iloc[exp_idx].to_numpy(),dtype=torch.float32)
    train_y = torch.tensor(result[target].to_numpy(),dtype=torch.float32) * scale
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
    return train_x,train_y

def count_different_columns(df, index1, index2):
    row1 = df.iloc[index1]
    row2 = df.iloc[index2]
    different_columns = 0
    
    for column in df.columns:
        if row1[column] != row2[column]:
            different_columns += 1   
    return different_columns


def get_model(model_name,best_params):
    assert  model_name == 'DT' or model_name == 'ET' or\
            model_name == 'GB' or model_name == 'KNR' or model_name == 'KRR' or\
            model_name == 'LSVR' or model_name == 'RF' or model_name == 'Ridge' or\
            model_name == 'SVR', 'Not support this ML model %s'%model_name
    
    if model_name=='DT':
        model = tree.DecisionTreeRegressor(max_depth=best_params[model_name]['max_depth'])
    elif model_name=='ET':
        model = ExtraTreesRegressor(n_jobs=-1,max_depth=best_params[model_name]['max_depth'],
                                    n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='GB':
        model = GradientBoostingRegressor(n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='KNR':
        model = KNeighborsRegressor(n_neighbors=best_params[model_name]['n_neighbors'])
    elif model_name=='KRR':
        model = KernelRidge(gamma=best_params[model_name]['gamma'])
    elif model_name=='LSVR':
        model = LinearSVR(epsilon=best_params[model_name]['epsilon'])
    elif model_name=='RF':
        model = RandomForestRegressor(n_jobs=-1,max_depth=best_params[model_name]['max_depth'],
                                        n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='Ridge':
        model = linear_model.Ridge(alpha=best_params[model_name]['alpha'])
    elif model_name=='SVR':
        model = SVR(kernel=best_params[model_name]['kernel'],gamma=best_params[model_name]['gamma'])
    return model

def get_best_model_and_param(train_x,train_y):
    random_seed=2024
    param_grid = {    
                    'DT':{'max_depth':[None,10,20,30]},
                    'ET':{'n_estimators':[50,100,200,300,400],'max_depth':[None,10,20,30]},
                    'GB':{'n_estimators':[50,100,200,300,400],'max_depth':[3,4,5]},
                    'KNR':{'n_neighbors':[2,4,6,8,10,12,14]},
                    'KRR':{'gamma':[None,0.01,0.001,0.0001]},
                    'LSVR':{'epsilon':[0.0,0.05,0.1],"C":[1,2,3,4,5,6,7,8,9,10]},
                    'RF':{'n_estimators':[50,100,200,300,400],'max_depth':[None,10,20,30]},
                    'Ridge':{'alpha':[0.5,1.0,1.5]},
                    'SVR':{'kernel':['rbf', 'linear', 'poly'],'gamma':['scale','auto']},
                    }

    models = [
            DecisionTreeRegressor(random_state=random_seed),                 
            ExtraTreesRegressor(n_jobs=-1,random_state=random_seed),
            GradientBoostingRegressor(random_state=random_seed),                   
            KNeighborsRegressor(n_jobs=-1),                    
            KernelRidge( ),                   
            LinearSVR( ),                   
            RandomForestRegressor(n_jobs=-1,random_state=random_seed),
            Ridge(random_state=random_seed),                      
            SVR(),                                
                ]
    model_names = ['DT','ET','GB','KNR','KRR','LSVR','RF','Ridge','SVR']   
    best_params = {}
    model2score={}
    kfold = KFold(n_splits=5,shuffle=True,random_state=random_seed)
    for model_name,model in zip(model_names,models):
        train_val_desc,train_val_target = train_x,train_y
        GS = GridSearchCV(model,param_grid[model_name],cv=kfold,n_jobs=-1,scoring='neg_mean_absolute_error' )#neg_mean_absolute_error neg_mean_squared_error 
        GS.fit(train_val_desc,train_val_target)
        best_param = GS.best_params_
        best_score = GS.best_score_
        best_params[model_name] = best_param
        model2score[model_name] = best_score
        print('Model: %4s, Best Socre: %.4f, Best Param: '%(model_name,best_score),best_param)
    best_model_name=list(model2score.keys())[np.argmax(np.array(list((model2score.values()))))]
    #print('Best Model: %4s, Best Param: '% best_model_name,best_model_params)
    best_model = get_model(best_model_name,best_params)
    return best_model

class auto_yield_optimization_cn():
    def __init__(self,train_x,train_y,random_state,model,n_jobs=-1,run_time = 120, per_run_time = 30):
        self.random_state = random_state
        self.train_x = train_x.cpu().numpy()
        self.train_y = train_y.cpu().numpy()
        self.model = model.lower()
        self.run_time = run_time
        self.per_run_time = per_run_time
        self.n_jobs = n_jobs
        #assert model == 'automl' or model == 'gridsearch' or model == 'rf' or model == 'et' or model == 'xgb', 'Only Auto ML, Grid Search, support RandomForest (RF), ExtraTrees(ET) and XGBoost(XGB) currently.'

        if model == 'DT':
            self.model = tree.DecisionTreeRegressor(random_state=random_state)            
        elif model == 'ET':
            self.model = ExtraTreesRegressor(n_jobs=n_jobs,random_state=random_state)
            
        elif model == 'GB':
            self.model = GradientBoostingRegressor(random_state=random_state)            
        elif model == 'KNR':
            self.model = KNeighborsRegressor(n_jobs=n_jobs)           
        elif model == 'KRR':
            self.model = KernelRidge()            
        elif model == 'LSVR':
            self.model = LinearSVR(random_state=random_state)  
        elif model == 'RF':
            self.model = RandomForestRegressor(n_jobs=n_jobs,random_state=random_state)          
        elif model == 'Ridge':
            self.model = linear_model.Ridge()            
        elif model == 'SVR':
            self.model = SVR()             
        elif model == 'XGB':
            self.model = XGBRegressor(random_state=random_state)                       
        elif model == 'gridsearch':
            self.model = get_best_model_and_param(self.train_x,self.train_y)          
        elif model == 'automl':
            tep_path = f'./tmp/autosklearn_regression_example_tmp_{random_state}'
            if os.path.exists(tep_path):
                shutil.rmtree(tep_path)
            self.model = AutoSklearnRegressor(  
            time_left_for_this_task = run_time,  
            per_run_time_limit = per_run_time,      
            metric = mean_absolute_error, 
            n_jobs = n_jobs,
            #include = {"regressor": ['random_forest', 'decision_tree', 'extra_trees', 'k_nearest_neighbors', 'gradient_boosting',]},#,'kernel_ridge','linear_svr','ridge_regression','support_vector_regression'
            ensemble_size = 1, 
            #memory_limit=-1,
            initial_configurations_via_metalearning=0,
            seed = random_state)  
    def recommend(self,domain,desc_domain,result,batch_size=10,stage=1,cc1=3,cc2=2,cc1_num=100,cc2_num=50,space_num=3955,target = 'yield'):   
        np.random.seed(self.random_state)    
        try:    
            self.model.fit(self.train_x,self.train_y)
            desc_domain_np = desc_domain.to_numpy()
            pred = self.model.predict(desc_domain_np)
        except:
            model = AutoML(time_left_for_this_task = self.run_time,  
            per_run_time_limit = self.per_run_time,       
            n_jobs = self.n_jobs,
            initial_configurations_via_metalearning=0,
            seed = self.random_state)  
            model.fit(self.train_x,self.train_y, task=REGRESSION)
            desc_domain_np = desc_domain.to_numpy()
            pred = model.predict(desc_domain_np)
        sampled_idx = []
        known_idx = [int(tmp_item) for tmp_item in result.index]
        
        if stage==1:
            num = 0
            while len(sampled_idx) < batch_size and num<space_num:
                num = num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min>=cc1:
                    sampled_idx.append(pot_idx)
                    known_idx.append(pot_idx)
        elif stage==2:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min>=cc2:
                    sampled_idx.append(pot_idx) 
                    known_idx.append(pot_idx)
        elif stage==3:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx)
                    known_idx.append(pot_idx)
        tem_stage=1   
        if stage==1 and num < cc1_num: 
            tem_stage=1
        elif stage==1 and num >= cc1_num:  
            tem_stage=2
        elif stage==2 and num < cc2_num:
            tem_stage=2
        elif stage==2 and num >= cc2_num:
            tem_stage=3
        elif stage==3:
            tem_stage=3
        domain_sampled = deepcopy(domain).iloc[sampled_idx]
        domain_sampled[target] = ['<Enter the result>'] * len(domain_sampled)
        return domain_sampled,tem_stage
def exe_exp(domain_sampled,exp_result):
    arha_smi = domain_sampled['Aryl_halide_SMILES'].to_list()
    add_smi = domain_sampled['Additive_SMILES'].to_list()
    base_smi = domain_sampled['Base_SMILES'].to_list()
    lig_smi = domain_sampled['Ligand_SMILES'].to_list()
    result = []
    for i in range(len(arha_smi)):
        try:
            tmp_targ = float(exp_result[(exp_result['Aryl_halide_SMILES'] == arha_smi[i]) &\
                          (exp_result['Additive_SMILES'] == add_smi[i]) &\
                          (exp_result['Base_SMILES'] == base_smi[i]) &\
                          (exp_result['Ligand_SMILES'] == lig_smi[i])]['yield'])
        except:
            tmp_targ = np.nan
        result.append(tmp_targ)
    return np.array(result,dtype=np.float32)


exp_result = pd.read_csv('./Data/data_cn/experiment_index.csv')
def yield_optimization_single_line(seed,des_name,domain,desc_domain,target = 'yield',tem_batch_size=5,ten_cc1=3,ten_cc2=2,tem_cc1num=(3955*0.25),tem_cc2num=int(3955*0.1),
                model='automl',rundata_dir = './Data/rundata_cn/',exp_result = exp_result,random_state = 0,task = '',run_time = 120, per_run_time = 30):
    print('Run seed:',seed)
    results_all_cycle=[]
    all_index=[]
    all_exp_index=[]
    result = pd.DataFrame.from_dict({tmp_key:[] for tmp_key in list(domain.keys()) + [target]})
    init_react = random_recom(tem_batch_size,domain,desc_domain,
                            rundata_dir+f'recommend_ourwork/init_{seed}.csv',random_state=seed,external_init='low_rate10')
    init_target = exe_exp(init_react,exp_result)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_react[target] = init_target
    init_react.to_csv(rundata_dir+f'result_ourwork/result_{seed}.csv')

    result = add_result(result,rundata_dir+f'result_ourwork/result_{seed}.csv')   
    stage=1
    all_stage=[1]
    train_x,train_y = result2xy(desc_domain,result=result)
    yield_optimization = auto_yield_optimization_cn(train_x,train_y,n_jobs=2,model='rf',random_state=random_state,run_time = run_time, per_run_time = per_run_time)
    domain_sampled,stage = yield_optimization.recommend(domain,desc_domain,result,batch_size=tem_batch_size,\
                stage=stage,cc1=ten_cc1,cc2=ten_cc2,cc1_num=tem_cc1num,cc2_num=tem_cc2num,target = 'yield')
    for try_idx in range(1,11):
        domain_sampled.to_csv(rundata_dir+f'recommend_ourwork/cycle_{seed}_{try_idx}.csv')
        new_target = exe_exp(domain_sampled,exp_result)
        new_result = deepcopy(domain_sampled)
        new_result[target] = new_target
        new_result.to_csv(rundata_dir+f'result_ourwork/cycle_{seed}_{try_idx}.csv')
        result = add_result(result,rundata_dir+f'result_ourwork/cycle_{seed}_{try_idx}.csv')
        train_x,train_y = result2xy(desc_domain,result=result)
        if try_idx <= 8:
            tem_model = 'rf'
        else:
            tem_model = model
        yield_optimization = auto_yield_optimization_cn(train_x,train_y,n_jobs=2,model=tem_model,random_state=random_state,run_time = run_time, per_run_time = per_run_time)
        domain_sampled,stage = yield_optimization.recommend(domain,desc_domain,result,batch_size=tem_batch_size,\
                stage=stage,cc1=ten_cc1,cc2=ten_cc2,cc1_num=tem_cc1num,cc2_num=tem_cc2num,target = 'yield')
        all_stage.append(stage)
        stage=max(all_stage)
    if try_idx == 9:
        stage = 3
    results_all_cycle.append(result[target].tolist()[:50])
    all_index.append(result.index.values[:5])
    all_exp_index.append(np.array(result)[:50,:])
    np.save(rundata_dir+f'results_optimization/{task}_{des_name}_{model}_{seed}.npy',results_all_cycle)
    np.save(rundata_dir+f'results_optimization/index_{task}_{des_name}_{model}_{seed}.npy',all_exp_index)
  
def get_total_results(dir,start,end):
    total_results = None
    for file in glob.glob(dir):
        if start<=int(file.split('/')[-1].split('.')[0].split('_')[-1])<end: 
            if total_results is not None:
                total_results = np.concatenate([total_results,np.load(file)],axis = 0)
            else:
                total_results = np.load(file)
                print(file)
    dir_save =dir.split('/')[-1].split('.')[0][:-1]+'total.npy'
    np.save(dir_save,total_results)
    print('Data save in:',dir_save)
    return total_results 