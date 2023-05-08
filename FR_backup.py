import os
import sklearn
import numpy as np
import pandas as pd
import json
import joblib
import sys
import csv
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
import pickle
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

class resultStdout:
    def __init__(self, root, dir_code):
        self.save_stdout = sys.stdout
        self.root = root
        self.dir_code = dir_code
        sys.stdout = self

    def write(self, message):
        if message:
            
            msg_lst = message.split(',')
            epoach = msg_lst[0].split(' ')[-1]
            loss = msg_lst[-1].split(' = ')[-1]

            if epoach == '\n' or loss =='\n':
                return

            with open(self.root+self.dir_code+'/result.csv', 'a') as fp:
                row = [epoach, loss]
                writer = csv.writer(fp)
                writer.writerow(row)


            
            # content.to_csv(self.root+self.dir_code+'/results.csv', index=False)

    def flush(self):
        pass

    def restore(self):
        sys.stdout = self.save_stdout

def load_obj(name):
    with open(f'./processed_data/{name}.pkl', 'rb') as f:
        return pickle.load(f)


def read_csv_file(path, transation_id, tags, fraud, mode, cal_mode):
    
    
    data = pd.read_csv(path)
    
    x_lst = []
    labels_lst = []
    y_lst = []


    for i in trange(len(data)):
        
        x_lst.append(data[transation_id][i])
        y_lst.append(data[fraud][i])     

    if mode == 'read_processed_datalst':
        print(f'reading_processed_{cal_mode}...')
        labels_lst = load_obj(f'processed_{cal_mode}')

    
    
    else:
        for i in trange(len(tags)):
            label_lst = []
            for j in range(len(data)): 
                label_lst.append(data[tags[i]][j])
            labels_lst.append(label_lst)
    
    
        


    #save as files
    if mode == 'save_and_read':
        with open(f'./processed_data/x_lst.txt', 'w') as f:
            f.write(str(x_lst))

        with open(f'./processed_data/labels_lst.txt', 'w') as f:
            f.write(str(labels_lst))

        with open(f'./processed_data/y_lst.txt', 'w') as f:
            f.write(str(y_lst))
    
    
    return x_lst, labels_lst, y_lst


def model(X, Y, mode, model_code, model_type, name_lst, tags, threshold):

    if mode == 'train':
        root = './results/train/'
        dir_code = '1'
        while os.path.exists(root+dir_code):
            dir_code = str(int(dir_code) + 1)

        os.mkdir(root+dir_code)
        

        #split dataset
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

        # sampling
        over = RandomOverSampler(sampling_strategy=0.3)
        X_train, Y_train = over.fit_resample(X_train, Y_train)

        #scale
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        if model_type == 'mlp':
            #model1: mlp

            mlp = MLPClassifier(activation='logistic',
                                solver='sgd',
                                alpha=1e-5, 
                                max_iter=10000,
                                learning_rate_init=0.01,
                                nesterovs_momentum=True,
                                learning_rate='adaptive',
                                hidden_layer_sizes=(100),  
                                tol=1e-4,
                                verbose=True,
                                early_stopping=True
                                )
            
            #initiate result.csv
            with open(root+dir_code+'/result.csv', 'w', encoding='utf-8', newline='') as fp:
                title = ['epoch', 'loss']
                write_title = csv.writer(fp)
                write_title.writerow(title)

            #redirect to the file, save the loss of each iter
            rd_res = resultStdout(root, dir_code)
            
            
            mlp.fit(X_train, Y_train)
            Y_pred_mlp = mlp.predict(X_test)
            prob_lst = mlp.predict_proba(X_test)
            score_mlp = accuracy_score(Y_test, Y_pred_mlp)
            auc_mlp = sklearn.metrics.roc_auc_score(Y_test, Y_pred_mlp)
            f1 = f1_score(Y_test, Y_pred_mlp)
            conf_mlp = confusion_matrix(Y_test, Y_pred_mlp)
            
                    
            joblib.dump(mlp, root+dir_code+'/mlp.m')

            #save confusion matrix
            disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_mlp, display_labels=['Legitimate', 'Fraudulent'])
            disp_conf.plot(include_values=True)
            plt.savefig(root+dir_code+'/conf_mat.png')

            #redirect back
            rd_res.restore()

            plt.figure(figsize=(20,5))                                               #显示面板
            plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
            plt.yticks(range(len(tags)), tags)
            plt.xlabel('Columns in weight matrix')
            plt.ylabel('Input feature')
            plt.colorbar()
            plt.savefig(root+dir_code+'/weights.png')

            
            print('mlp:', score_mlp, auc_mlp, f1, conf_mlp)


        
        elif model_type == 'lgbm':
            #model2: lgbm

            lgbm = LGBMClassifier(application='binary',
                                learning_rate=0.03,
                                is_unbalance=True,
                                sample_pos_weight=0.15,
                                metric='auc',
                                boosting='gbdt',
                                num_boost_round=300,
                                max_depth=8,
                                subsample=0.2,
                                n_estimators=500,
                                feature_fraction=0.8,
                                bagging_fraction=0.8,
                                bagging_freq=15,
                                min_child_weight=0.03,
                                silent=False,
                            )
            
            #initiate result.csv
            with open(root+dir_code+'/result.csv', 'w', encoding='utf-8', newline='') as fp:
                title = ['epoch', 'loss']
                write_title = csv.writer(fp)
                write_title.writerow(title)

            #redirect to the file, save the loss of each iter
            rd_res = resultStdout(root, dir_code)
            
            
            lgbm.fit(X_train, Y_train)
            #set threshold = 0.6
            Y_pred_ = (lgbm.predict_proba(X_test)[:,1] >= threshold).astype(bool)
            prob_lst = lgbm.predict_proba(X_test)
            score_ = accuracy_score(Y_test, Y_pred_)
            auc_ = sklearn.metrics.roc_auc_score(Y_test, Y_pred_)
            f1 = f1_score(Y_test, Y_pred_)
            conf_ = confusion_matrix(Y_test, Y_pred_)
            
                    
            joblib.dump(lgbm, root+dir_code+'/lgbm.m')

            #save confusion matrix
            disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_, display_labels=['Legitimate', 'Fraudulent'])
            disp_conf.plot(include_values=True)
            plt.savefig(root+dir_code+'/conf_mat.png')

            #redirect back
            rd_res.restore()

            
            print('lgbm:', score_, auc_, conf_, f1)

            

        













        elif model_type == 'xgboost':
            #model3: xgboost

            # param_grid = {
            # 'max_depth': [2, 3, 4, 5, 6, 7, 8],
            # 'n_estimators': [30, 50, 100, 300, 500, 1000,2000],
            # 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.5],
            # "gamma":[0.0, 0.1, 0.2, 0.3, 0.4],
            # "reg_alpha":[0.0001,0.001, 0.01, 0.1, 1, 100],
            # "reg_lambda":[0.0001,0.001, 0.01, 0.1, 1, 100],
            # "min_child_weight": [2,3,4,5,6,7,8],
            # "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            # "subsample":[0.6, 0.7, 0.8, 0.9]}

            # gsearch1 = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv=5)
            # gsearch1.fit(X_train, Y_train)

            # print("best_score_:", gsearch1.best_params_, gsearch1.best_score_)

            xgboost = XGBClassifier(n_estimators=500,
                                    max_depth=5,
                                    min_child_weight=3,
                                    gamma=0,subsample=0.8,  
                                    colsample_bytree=0.7,
                                    learning_rate=0.05, 
                                    objective= 'binary:logistic',
                                    reg_alpha = 0.001,
                                    reg_lambda = 0.001,
                                    scale_pos_weight=1,
                                    
                                    )
            
            #initiate result.csv
            with open(root+dir_code+'/result.csv', 'w', encoding='utf-8', newline='') as fp:
                title = ['epoch', 'loss']
                write_title = csv.writer(fp)
                write_title.writerow(title)

            #redirect to the file, save the loss of each iter
            rd_res = resultStdout(root, dir_code)
            
            
            xgboost.fit(X_train, Y_train)
            Y_pred_ = (xgboost.predict_proba(X_test)[:,1] >= threshold).astype(bool)
            # Y_pred_ = xgboost.predict(X)
            prob_lst = xgboost.predict_proba(X_test)
            score_ = accuracy_score(Y_test, Y_pred_)
            auc_ = sklearn.metrics.roc_auc_score(Y_test, Y_pred_)
            f1 = f1_score(Y_test, Y_pred_)
            conf_ = confusion_matrix(Y_test, Y_pred_)
            
                    
            joblib.dump(xgboost, root+dir_code+'/xgboost.m')

            #save confusion matrix
            disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_, display_labels=['Legitimate', 'Fraudulent'])
            disp_conf.plot(include_values=True)
            plt.savefig(root+dir_code+'/conf_mat.png')

            #redirect back
            rd_res.restore()

            
            print('xgboost:', score_, auc_, conf_, f1)



        

        #save probability
        with open(root+dir_code+'/prob.txt', 'w') as f:
            for i in range(len(prob_lst)):
                f.write(str(prob_lst[i]))

        


        
        
    elif mode == 'value1' or mode == 'value2':
        #scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        root = './results/val/'
        dir_code = '1'
        while os.path.exists(root+dir_code):
            dir_code = str(int(dir_code) + 1)

        os.mkdir(root+dir_code)

        model = joblib.load(f'./results/train/{model_code}/{model_type}.m')

        Y_pred_ = model.predict(X)
        score_ = accuracy_score(Y, Y_pred_)
        auc_ = sklearn.metrics.roc_auc_score(Y, Y_pred_)
        f1 = f1_score(Y, Y_pred_)
        conf_ = confusion_matrix(Y, Y_pred_,normalize='all')
            

        #save confusion matrix
        disp_conf = ConfusionMatrixDisplay(confusion_matrix=conf_, display_labels=['Legitimate', 'Fraudulent'])
        disp_conf.plot(include_values=True)
        plt.savefig(root+dir_code+'/conf_mat.png')

        Y_pred = [int(Y_pred_[i]) for i in range(len(Y_pred_))]

        prob_lst = model.predict_proba(X)
        prob_lst = [prob_lst[i][1] for i in range(len(prob_lst))]

        

        

        #save_csv
        df = pd.DataFrame({'TRANSACTION_ID':name_lst, 'PREDICTION': Y_pred, 'PROBABILITY': prob_lst})
        df.to_csv(root+dir_code+'/test.csv', index=False)

        PF_cnt = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1:
                PF_cnt += 1

        print('PF', PF_cnt, score_, auc_, f1)




    else:
        model = joblib.load(f'./results/train/{model_code}/{model_type}.m')

        Y_pred = (model.predict_proba(X)[:,1] >= threshold).astype(bool)
        Y_pred = [int(Y_pred[i]) for i in range(len(Y_pred))]

        prob_lst = model.predict_proba(X)
        prob_lst = [prob_lst[i][1] for i in range(len(prob_lst))]

        root = './results/test/'
        dir_code = '1'
        while os.path.exists(root+dir_code):
            dir_code = str(int(dir_code) + 1)

        os.mkdir(root+dir_code)
        
        

        #save_csv
        df = pd.DataFrame({'TRANSACTION_ID':name_lst, 'PREDICTION': Y_pred, 'PROBABILITY': prob_lst})
        df.to_csv(root+dir_code+'/test.csv', index=False)

        PF_cnt = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1:
                PF_cnt += 1

        print('PF', PF_cnt)

        




def normalize(labels_lst):
    res = []
    for label_lst in labels_lst:
        r = []
        mx = max(label_lst)
        mi = min(label_lst)

        for label in label_lst:
            if mx == mi:
                xi = 0

            else:
                xi = (label - mi)/(mx - mi)
            r.append(xi)

        res.append(r)

    return res

def standarlize(labels_lst):
    res = []
    for label_lst in labels_lst:
        lst = np.array(label_lst)
        mean = lst.mean()
        std = lst.std()

        s = []
        if max(label_lst) == 1:
            s = label_lst

        else:
            for i in range(len(label_lst)):
                xi = (label_lst[i] - mean) / std
                s.append(xi)
        res.append(s)

    return res






def reformat(labels_lst):
    vec_lst = []
    print(len(labels_lst[1]))
    for i in trange(len(labels_lst[1])):
        vec = []
        for label_lst in labels_lst:
            vec.append(label_lst[i])
        vec_lst.append(vec)

    return vec_lst
    






if __name__ == '__main__':
    #operations
    root = './processed_data/'  #processed_data ; data
    mode = 'train' #mode: train, test
    model_code = 3     #best:40; 177 ; 129, 212lgbm; tag, mlp = MLPClassifier(activation='relu',alpha=1e-5,max_iter=10000, hidden_layer_sizes=(100,30))
    model_type = 'lgbm'   #mlp, lgbm, xgboost
    name = '_ScotiaDSD.csv'
    read_new_file = True
    read_method = 'read_processed_datalst'    #read ; save_and_read ; read_processed_datalst
    threshold = 0.85


    # path = root+mode+name
    path = '/Users/donglianghan/Desktop/ai_notes/AF/processed_dat.csv'

    tg = 'AVAIL_CRDT,AMOUNT,CREDIT_LIMIT,CARD_NOT_PRESENT,FLAG_LX,FLAG_ATM,FLAG_AUTO,FLAG_CASH,FLAG_LS,FLAG_DISCOUNT,FLAG_RECREA,FLAG_ELCTRNCS,FLAG_REG_AMT,FLAG_FASTFOOD,FLAG_GAS,FLAG_HIGH_AMT,FLAG_HIGH_RECREA,FLAG_INTERNET,FLAG_INTERNATIONAL,FLAG_JEWELRY,FLAG_LOW_AMT,FLAG_MANUAL_ENTRY,FLAG_PHONE_ORDER,FLAG_PURCHASE_EXCLUDING_GAS,FLAG_PLANNED,FLAG_RISKY,FLAG_SWIPE,FLAG_TRAVEL_ONLY,FLAG_TRAVEL_AND_ENTERTAINMENT,FLAG_WEEKEND,MEAN_AUTO_PAST_7DAY,MEAN_LS_PAST_7DAY,MEAN_RECREA_PAST_7DAY,MEAN_REG_AMT_PAST_7DAY,MEAN_FASTFOOD_PAST_7DAY,MEAN_HIGH_AMT_PAST_7DAY,MEAN_HIGH_RECREA_PAST_7DAY,MEAN_INTERNET_PAST_7DAY,MEAN_INTERNATIONAL_PAST_7DAY,MEAN_JEWELRY_PAST_7DAY,MEAN_LOW_AMT_PAST_7DAY,MEAN_MANUAL_ENTRY_PAST_7DAY,MEAN_PHONE_ORDER_PAST_7DAY,MEAN_PLANNED_PAST_7DAY,MEAN_SWIPE_PAST_7DAY,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,MEAN_WEEKEND_PAST_7DAY,MAX_CASH_PAST_7DAY,MAX_LS_PAST_7DAY,MAX_RECREA_PAST_7DAY,MAX_HIGH_AMT_PAST_7DAY,MAX_HIGH_RECREA_PAST_7DAY,MAX_INTERNET_PAST_7DAY,MAX_PHONE_ORDER_PAST_7DAY,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY,MAX_SWIPE_PAST_7DAY,MAX_WEEKEND_PAST_7DAY,STD_LX_PAST_7DAY,STD_FASTFOOD_PAST_7DAY,STD_HIGH_AMT_PAST_7DAY,STD_INTERNET_PAST_7DAY,STD_LOW_AMT_PAST_7DAY,STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY,STD_SWIPE_PAST_7DAY,STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,SUM_LX_PAST_7DAY,SUM_AUTO_PAST_7DAY,SUM_LS_PAST_7DAY,SUM_RECREA_PAST_7DAY,SUM_GAS_PAST_7DAY,SUM_HIGH_AMT_PAST_7DAY,SUM_INTERNET_PAST_7DAY,SUM_INTERNATIONAL_PAST_7DAY,SUM_LOW_AMT_PAST_7DAY,SUM_MANUAL_ENTRY_PAST_7DAY,SUM_PHONE_ORDER_PAST_7DAY,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY,SUM_PARTIAL_PAST_7DAY,SUM_PLANNED_PAST_7DAY,SUM_SWIPE_PAST_7DAY,SUM_WEEKEND_PAST_7DAY,COUNT_AUTO_PAST_7DAY,COUNT_ELCTRNCS_PAST_7DAY,COUNT_GAS_PAST_7DAY,COUNT_HIGH_AMT_PAST_7DAY,COUNT_INTERNET_PAST_7DAY,COUNT_LOW_AMT_PAST_7DAY,COUNT_MANUAL_ENTRY_PAST_7DAY,COUNT_PHONE_ORDER_PAST_7DAY,COUNT_PURCHASE_EXCLUDING_GAS_PAST_7DAY,COUNT_SWIPE_PAST_7DAY,COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,COUNT_WEEKEND_PAST_7DAY,MEAN_AUTO_PAST_30DAY,MEAN_DISCOUNT_PAST_30DAY,MEAN_RECREA_PAST_30DAY,MEAN_ELCTRNCS_PAST_30DAY,MEAN_REG_AMT_PAST_30DAY,MEAN_HIGH_AMT_PAST_30DAY,MEAN_INTERNET_PAST_30DAY,MEAN_LOW_AMT_PAST_30DAY,MEAN_MANUAL_ENTRY_PAST_30DAY,MEAN_PHONE_ORDER_PAST_30DAY,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY,MEAN_PLANNED_PAST_30DAY,MEAN_SWIPE_PAST_30DAY,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,MEAN_WEEKEND_PAST_30DAY,MAX_AUTO_PAST_30DAY,MAX_LS_PAST_30DAY,MAX_ELCTRNCS_PAST_30DAY,MAX_FASTFOOD_PAST_30DAY,MAX_HIGH_RECREA_PAST_30DAY,MAX_MANUAL_ENTRY_PAST_30DAY,MAX_PHONE_ORDER_PAST_30DAY,MAX_PARTIAL_PAST_30DAY,MAX_RISKY_PAST_30DAY,MAX_WEEKEND_PAST_30DAY,STD_AUTO_PAST_30DAY,STD_LS_PAST_30DAY,STD_RECREA_PAST_30DAY,STD_ELCTRNCS_PAST_30DAY,STD_REG_AMT_PAST_30DAY,STD_HIGH_RECREA_PAST_30DAY,STD_INTERNET_PAST_30DAY,STD_LOW_AMT_PAST_30DAY,STD_MANUAL_ENTRY_PAST_30DAY,STD_PHONE_ORDER_PAST_30DAY,STD_PARTIAL_PAST_30DAY,STD_SWIPE_PAST_30DAY,STD_TRAVEL_ONLY_PAST_30DAY,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,SUM_AUTO_PAST_30DAY,SUM_LS_PAST_30DAY,SUM_DISCOUNT_PAST_30DAY,SUM_RECREA_PAST_30DAY,SUM_ELCTRNCS_PAST_30DAY,SUM_REG_AMT_PAST_30DAY,SUM_FASTFOOD_PAST_30DAY,SUM_GAS_PAST_30DAY,SUM_HIGH_AMT_PAST_30DAY,SUM_HIGH_RECREA_PAST_30DAY,SUM_INTERNET_PAST_30DAY,SUM_INTERNATIONAL_PAST_30DAY,SUM_LOW_AMT_PAST_30DAY,SUM_MANUAL_ENTRY_PAST_30DAY,SUM_PHONE_ORDER_PAST_30DAY,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY,SUM_SWIPE_PAST_30DAY,SUM_TRAVEL_ONLY_PAST_30DAY,SUM_WEEKEND_PAST_30DAY,COUNT_AUTO_PAST_30DAY,COUNT_RECREA_PAST_30DAY,COUNT_REG_AMT_PAST_30DAY,COUNT_FASTFOOD_PAST_30DAY,COUNT_GAS_PAST_30DAY,COUNT_HIGH_AMT_PAST_30DAY,COUNT_INTERNET_PAST_30DAY,COUNT_LOW_AMT_PAST_30DAY,COUNT_MANUAL_ENTRY_PAST_30DAY,COUNT_PHONE_ORDER_PAST_30DAY,COUNT_PURCHASE_EXCLUDING_GAS_PAST_30DAY,COUNT_PLANNED_PAST_30DAY,COUNT_SWIPE_PAST_30DAY,COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,COUNT_WEEKEND_PAST_30DAY'
    tg_lg = 'AVAIL_CRDT,AMOUNT,CREDIT_LIMIT,CARD_NOT_PRESENT,FLAG_LX,FLAG_ATM,FLAG_AUTO,FLAG_CASH,FLAG_LS,FLAG_DISCOUNT,FLAG_RECREA,FLAG_ELCTRNCS,FLAG_REG_AMT,FLAG_FASTFOOD,FLAG_GAS,FLAG_HIGH_AMT,FLAG_HIGH_RECREA,FLAG_INTERNET,FLAG_INTERNATIONAL,FLAG_JEWELRY,FLAG_LOW_AMT,FLAG_MANUAL_ENTRY,FLAG_PHONE_ORDER,FLAG_PURCHASE_EXCLUDING_GAS,FLAG_PLANNED,FLAG_RISKY,FLAG_SWIPE,FLAG_TRAVEL_ONLY,FLAG_TRAVEL_AND_ENTERTAINMENT,FLAG_WEEKEND,MEAN_AUTO_PAST_7DAY,MEAN_LS_PAST_7DAY,MEAN_RECREA_PAST_7DAY,MEAN_REG_AMT_PAST_7DAY,MEAN_FASTFOOD_PAST_7DAY,MEAN_HIGH_AMT_PAST_7DAY,MEAN_HIGH_RECREA_PAST_7DAY,MEAN_INTERNET_PAST_7DAY,MEAN_INTERNATIONAL_PAST_7DAY,MEAN_JEWELRY_PAST_7DAY,MEAN_LOW_AMT_PAST_7DAY,MEAN_MANUAL_ENTRY_PAST_7DAY,MEAN_PHONE_ORDER_PAST_7DAY,MEAN_PLANNED_PAST_7DAY,MEAN_SWIPE_PAST_7DAY,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,MEAN_WEEKEND_PAST_7DAY,MAX_CASH_PAST_7DAY,MAX_LS_PAST_7DAY,MAX_RECREA_PAST_7DAY,MAX_HIGH_AMT_PAST_7DAY,MAX_HIGH_RECREA_PAST_7DAY,MAX_INTERNET_PAST_7DAY,MAX_PHONE_ORDER_PAST_7DAY,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY,MAX_SWIPE_PAST_7DAY,MAX_WEEKEND_PAST_7DAY,STD_LX_PAST_7DAY,STD_FASTFOOD_PAST_7DAY,STD_HIGH_AMT_PAST_7DAY,STD_INTERNET_PAST_7DAY,STD_LOW_AMT_PAST_7DAY,STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY,STD_SWIPE_PAST_7DAY,STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,SUM_LX_PAST_7DAY,SUM_AUTO_PAST_7DAY,SUM_LS_PAST_7DAY,SUM_RECREA_PAST_7DAY,SUM_GAS_PAST_7DAY,SUM_HIGH_AMT_PAST_7DAY,SUM_INTERNET_PAST_7DAY,SUM_INTERNATIONAL_PAST_7DAY,SUM_LOW_AMT_PAST_7DAY,SUM_MANUAL_ENTRY_PAST_7DAY,SUM_PHONE_ORDER_PAST_7DAY,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY,SUM_PARTIAL_PAST_7DAY,SUM_PLANNED_PAST_7DAY,SUM_SWIPE_PAST_7DAY,SUM_WEEKEND_PAST_7DAY,COUNT_AUTO_PAST_7DAY,COUNT_ELCTRNCS_PAST_7DAY,COUNT_GAS_PAST_7DAY,COUNT_HIGH_AMT_PAST_7DAY,COUNT_INTERNET_PAST_7DAY,COUNT_LOW_AMT_PAST_7DAY,COUNT_MANUAL_ENTRY_PAST_7DAY,COUNT_PHONE_ORDER_PAST_7DAY,COUNT_PURCHASE_EXCLUDING_GAS_PAST_7DAY,COUNT_SWIPE_PAST_7DAY,COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,COUNT_WEEKEND_PAST_7DAY,MEAN_AUTO_PAST_30DAY,MEAN_DISCOUNT_PAST_30DAY,MEAN_RECREA_PAST_30DAY,MEAN_ELCTRNCS_PAST_30DAY,MEAN_REG_AMT_PAST_30DAY,MEAN_HIGH_AMT_PAST_30DAY,MEAN_INTERNET_PAST_30DAY,MEAN_LOW_AMT_PAST_30DAY,MEAN_MANUAL_ENTRY_PAST_30DAY,MEAN_PHONE_ORDER_PAST_30DAY,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY,MEAN_PLANNED_PAST_30DAY,MEAN_SWIPE_PAST_30DAY,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,MEAN_WEEKEND_PAST_30DAY,MAX_AUTO_PAST_30DAY,MAX_LS_PAST_30DAY,MAX_ELCTRNCS_PAST_30DAY,MAX_FASTFOOD_PAST_30DAY,MAX_HIGH_RECREA_PAST_30DAY,MAX_MANUAL_ENTRY_PAST_30DAY,MAX_PHONE_ORDER_PAST_30DAY,MAX_PARTIAL_PAST_30DAY,MAX_RISKY_PAST_30DAY,MAX_WEEKEND_PAST_30DAY,STD_AUTO_PAST_30DAY,STD_LS_PAST_30DAY,STD_RECREA_PAST_30DAY,STD_ELCTRNCS_PAST_30DAY,STD_REG_AMT_PAST_30DAY,STD_HIGH_RECREA_PAST_30DAY,STD_INTERNET_PAST_30DAY,STD_LOW_AMT_PAST_30DAY,STD_MANUAL_ENTRY_PAST_30DAY,STD_PHONE_ORDER_PAST_30DAY,STD_PARTIAL_PAST_30DAY,STD_SWIPE_PAST_30DAY,STD_TRAVEL_ONLY_PAST_30DAY,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,SUM_AUTO_PAST_30DAY,SUM_LS_PAST_30DAY,SUM_DISCOUNT_PAST_30DAY,SUM_RECREA_PAST_30DAY,SUM_ELCTRNCS_PAST_30DAY,SUM_REG_AMT_PAST_30DAY,SUM_FASTFOOD_PAST_30DAY,SUM_GAS_PAST_30DAY,SUM_HIGH_AMT_PAST_30DAY,SUM_HIGH_RECREA_PAST_30DAY,SUM_INTERNET_PAST_30DAY,SUM_INTERNATIONAL_PAST_30DAY,SUM_LOW_AMT_PAST_30DAY,SUM_MANUAL_ENTRY_PAST_30DAY,SUM_PHONE_ORDER_PAST_30DAY,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY,SUM_SWIPE_PAST_30DAY,SUM_TRAVEL_ONLY_PAST_30DAY,SUM_WEEKEND_PAST_30DAY,COUNT_AUTO_PAST_30DAY,COUNT_RECREA_PAST_30DAY,COUNT_REG_AMT_PAST_30DAY,COUNT_FASTFOOD_PAST_30DAY,COUNT_GAS_PAST_30DAY,COUNT_HIGH_AMT_PAST_30DAY,COUNT_INTERNET_PAST_30DAY,COUNT_LOW_AMT_PAST_30DAY,COUNT_MANUAL_ENTRY_PAST_30DAY,COUNT_PHONE_ORDER_PAST_30DAY,COUNT_PURCHASE_EXCLUDING_GAS_PAST_30DAY,COUNT_PLANNED_PAST_30DAY,COUNT_SWIPE_PAST_30DAY,COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,COUNT_WEEKEND_PAST_30DAY,PREV_M_INFLATION,PREV_M_UNEMP_RATE,MEAN_AUTO_PAST_7DAY$^0$,MEAN_AUTO_PAST_7DAY$^1$,MEAN_AUTO_PAST_7DAY$^2$,MEAN_AUTO_PAST_7DAY$^3$,MEAN_LS_PAST_7DAY$^0$,MEAN_LS_PAST_7DAY$^1$,MEAN_LS_PAST_7DAY$^2$,MEAN_LS_PAST_7DAY$^3$,MEAN_RECREA_PAST_7DAY$^0$,MEAN_RECREA_PAST_7DAY$^1$,MEAN_RECREA_PAST_7DAY$^2$,MEAN_RECREA_PAST_7DAY$^3$,MEAN_REG_AMT_PAST_7DAY$^0$,MEAN_REG_AMT_PAST_7DAY$^1$,MEAN_REG_AMT_PAST_7DAY$^2$,MEAN_REG_AMT_PAST_7DAY$^3$,MEAN_FASTFOOD_PAST_7DAY$^0$,MEAN_FASTFOOD_PAST_7DAY$^1$,MEAN_FASTFOOD_PAST_7DAY$^2$,MEAN_FASTFOOD_PAST_7DAY$^3$,MEAN_HIGH_AMT_PAST_7DAY$^0$,MEAN_HIGH_AMT_PAST_7DAY$^1$,MEAN_HIGH_AMT_PAST_7DAY$^2$,MEAN_HIGH_AMT_PAST_7DAY$^3$,MEAN_HIGH_RECREA_PAST_7DAY$^0$,MEAN_HIGH_RECREA_PAST_7DAY$^1$,MEAN_HIGH_RECREA_PAST_7DAY$^2$,MEAN_HIGH_RECREA_PAST_7DAY$^3$,MEAN_INTERNET_PAST_7DAY$^0$,MEAN_INTERNET_PAST_7DAY$^1$,MEAN_INTERNET_PAST_7DAY$^2$,MEAN_INTERNET_PAST_7DAY$^3$,MEAN_INTERNATIONAL_PAST_7DAY$^0$,MEAN_INTERNATIONAL_PAST_7DAY$^1$,MEAN_INTERNATIONAL_PAST_7DAY$^2$,MEAN_INTERNATIONAL_PAST_7DAY$^3$,MEAN_JEWELRY_PAST_7DAY$^0$,MEAN_JEWELRY_PAST_7DAY$^1$,MEAN_JEWELRY_PAST_7DAY$^2$,MEAN_JEWELRY_PAST_7DAY$^3$,MEAN_LOW_AMT_PAST_7DAY$^0$,MEAN_LOW_AMT_PAST_7DAY$^1$,MEAN_LOW_AMT_PAST_7DAY$^2$,MEAN_LOW_AMT_PAST_7DAY$^3$,MEAN_MANUAL_ENTRY_PAST_7DAY$^0$,MEAN_MANUAL_ENTRY_PAST_7DAY$^1$,MEAN_MANUAL_ENTRY_PAST_7DAY$^2$,MEAN_MANUAL_ENTRY_PAST_7DAY$^3$,MEAN_PHONE_ORDER_PAST_7DAY$^0$,MEAN_PHONE_ORDER_PAST_7DAY$^1$,MEAN_PHONE_ORDER_PAST_7DAY$^2$,MEAN_PHONE_ORDER_PAST_7DAY$^3$,MEAN_PLANNED_PAST_7DAY$^0$,MEAN_PLANNED_PAST_7DAY$^1$,MEAN_PLANNED_PAST_7DAY$^2$,MEAN_PLANNED_PAST_7DAY$^3$,MEAN_SWIPE_PAST_7DAY$^0$,MEAN_SWIPE_PAST_7DAY$^1$,MEAN_SWIPE_PAST_7DAY$^2$,MEAN_SWIPE_PAST_7DAY$^3$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^0$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^1$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^2$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^3$,MEAN_WEEKEND_PAST_7DAY$^0$,MEAN_WEEKEND_PAST_7DAY$^1$,MEAN_WEEKEND_PAST_7DAY$^2$,MEAN_WEEKEND_PAST_7DAY$^3$,MAX_CASH_PAST_7DAY$^0$,MAX_CASH_PAST_7DAY$^1$,MAX_CASH_PAST_7DAY$^2$,MAX_CASH_PAST_7DAY$^3$,MAX_LS_PAST_7DAY$^0$,MAX_LS_PAST_7DAY$^1$,MAX_LS_PAST_7DAY$^2$,MAX_LS_PAST_7DAY$^3$,MAX_RECREA_PAST_7DAY$^0$,MAX_RECREA_PAST_7DAY$^1$,MAX_RECREA_PAST_7DAY$^2$,MAX_RECREA_PAST_7DAY$^3$,MAX_HIGH_AMT_PAST_7DAY$^0$,MAX_HIGH_AMT_PAST_7DAY$^1$,MAX_HIGH_AMT_PAST_7DAY$^2$,MAX_HIGH_AMT_PAST_7DAY$^3$,MAX_HIGH_RECREA_PAST_7DAY$^0$,MAX_HIGH_RECREA_PAST_7DAY$^1$,MAX_HIGH_RECREA_PAST_7DAY$^2$,MAX_HIGH_RECREA_PAST_7DAY$^3$,MAX_INTERNET_PAST_7DAY$^0$,MAX_INTERNET_PAST_7DAY$^1$,MAX_INTERNET_PAST_7DAY$^2$,MAX_INTERNET_PAST_7DAY$^3$,MAX_PHONE_ORDER_PAST_7DAY$^0$,MAX_PHONE_ORDER_PAST_7DAY$^1$,MAX_PHONE_ORDER_PAST_7DAY$^2$,MAX_PHONE_ORDER_PAST_7DAY$^3$,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^0$,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^1$,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^2$,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^3$,MAX_SWIPE_PAST_7DAY$^0$,MAX_SWIPE_PAST_7DAY$^1$,MAX_SWIPE_PAST_7DAY$^2$,MAX_SWIPE_PAST_7DAY$^3$,MAX_WEEKEND_PAST_7DAY$^0$,MAX_WEEKEND_PAST_7DAY$^1$,MAX_WEEKEND_PAST_7DAY$^2$,MAX_WEEKEND_PAST_7DAY$^3$,STD_LX_PAST_7DAY$^0$,STD_LX_PAST_7DAY$^1$,STD_LX_PAST_7DAY$^2$,STD_LX_PAST_7DAY$^3$,STD_FASTFOOD_PAST_7DAY$^0$,STD_FASTFOOD_PAST_7DAY$^1$,STD_FASTFOOD_PAST_7DAY$^2$,STD_FASTFOOD_PAST_7DAY$^3$,STD_HIGH_AMT_PAST_7DAY$^0$,STD_HIGH_AMT_PAST_7DAY$^1$,STD_HIGH_AMT_PAST_7DAY$^2$,STD_HIGH_AMT_PAST_7DAY$^3$,STD_INTERNET_PAST_7DAY$^0$,STD_INTERNET_PAST_7DAY$^1$,STD_INTERNET_PAST_7DAY$^2$,STD_INTERNET_PAST_7DAY$^3$,STD_LOW_AMT_PAST_7DAY$^0$,STD_LOW_AMT_PAST_7DAY$^1$,STD_LOW_AMT_PAST_7DAY$^2$,STD_LOW_AMT_PAST_7DAY$^3$,STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^0$,STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^1$,STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^2$,STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^3$,STD_SWIPE_PAST_7DAY$^0$,STD_SWIPE_PAST_7DAY$^1$,STD_SWIPE_PAST_7DAY$^2$,STD_SWIPE_PAST_7DAY$^3$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^0$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^1$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^2$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY$^3$,SUM_LX_PAST_7DAY$^0$,SUM_LX_PAST_7DAY$^1$,SUM_LX_PAST_7DAY$^2$,SUM_LX_PAST_7DAY$^3$,SUM_AUTO_PAST_7DAY$^0$,SUM_AUTO_PAST_7DAY$^1$,SUM_AUTO_PAST_7DAY$^2$,SUM_AUTO_PAST_7DAY$^3$,SUM_LS_PAST_7DAY$^0$,SUM_LS_PAST_7DAY$^1$,SUM_LS_PAST_7DAY$^2$,SUM_LS_PAST_7DAY$^3$,SUM_RECREA_PAST_7DAY$^0$,SUM_RECREA_PAST_7DAY$^1$,SUM_RECREA_PAST_7DAY$^2$,SUM_RECREA_PAST_7DAY$^3$,SUM_GAS_PAST_7DAY$^0$,SUM_GAS_PAST_7DAY$^1$,SUM_GAS_PAST_7DAY$^2$,SUM_GAS_PAST_7DAY$^3$,SUM_HIGH_AMT_PAST_7DAY$^0$,SUM_HIGH_AMT_PAST_7DAY$^1$,SUM_HIGH_AMT_PAST_7DAY$^2$,SUM_HIGH_AMT_PAST_7DAY$^3$,SUM_INTERNET_PAST_7DAY$^0$,SUM_INTERNET_PAST_7DAY$^1$,SUM_INTERNET_PAST_7DAY$^2$,SUM_INTERNET_PAST_7DAY$^3$,SUM_INTERNATIONAL_PAST_7DAY$^0$,SUM_INTERNATIONAL_PAST_7DAY$^1$,SUM_INTERNATIONAL_PAST_7DAY$^2$,SUM_INTERNATIONAL_PAST_7DAY$^3$,SUM_LOW_AMT_PAST_7DAY$^0$,SUM_LOW_AMT_PAST_7DAY$^1$,SUM_LOW_AMT_PAST_7DAY$^2$,SUM_LOW_AMT_PAST_7DAY$^3$,SUM_MANUAL_ENTRY_PAST_7DAY$^0$,SUM_MANUAL_ENTRY_PAST_7DAY$^1$,SUM_MANUAL_ENTRY_PAST_7DAY$^2$,SUM_MANUAL_ENTRY_PAST_7DAY$^3$,SUM_PHONE_ORDER_PAST_7DAY$^0$,SUM_PHONE_ORDER_PAST_7DAY$^1$,SUM_PHONE_ORDER_PAST_7DAY$^2$,SUM_PHONE_ORDER_PAST_7DAY$^3$,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^0$,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^1$,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^2$,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY$^3$,SUM_PARTIAL_PAST_7DAY$^0$,SUM_PARTIAL_PAST_7DAY$^1$,SUM_PARTIAL_PAST_7DAY$^2$,SUM_PARTIAL_PAST_7DAY$^3$,SUM_PLANNED_PAST_7DAY$^0$,SUM_PLANNED_PAST_7DAY$^1$,SUM_PLANNED_PAST_7DAY$^2$,SUM_PLANNED_PAST_7DAY$^3$,SUM_SWIPE_PAST_7DAY$^0$,SUM_SWIPE_PAST_7DAY$^1$,SUM_SWIPE_PAST_7DAY$^2$,SUM_SWIPE_PAST_7DAY$^3$,SUM_WEEKEND_PAST_7DAY$^0$,SUM_WEEKEND_PAST_7DAY$^1$,SUM_WEEKEND_PAST_7DAY$^2$,SUM_WEEKEND_PAST_7DAY$^3$,MEAN_AUTO_PAST_30DAY$^0$,MEAN_AUTO_PAST_30DAY$^1$,MEAN_AUTO_PAST_30DAY$^2$,MEAN_AUTO_PAST_30DAY$^3$,MEAN_DISCOUNT_PAST_30DAY$^0$,MEAN_DISCOUNT_PAST_30DAY$^1$,MEAN_DISCOUNT_PAST_30DAY$^2$,MEAN_DISCOUNT_PAST_30DAY$^3$,MEAN_RECREA_PAST_30DAY$^0$,MEAN_RECREA_PAST_30DAY$^1$,MEAN_RECREA_PAST_30DAY$^2$,MEAN_RECREA_PAST_30DAY$^3$,MEAN_ELCTRNCS_PAST_30DAY$^0$,MEAN_ELCTRNCS_PAST_30DAY$^1$,MEAN_ELCTRNCS_PAST_30DAY$^2$,MEAN_ELCTRNCS_PAST_30DAY$^3$,MEAN_REG_AMT_PAST_30DAY$^0$,MEAN_REG_AMT_PAST_30DAY$^1$,MEAN_REG_AMT_PAST_30DAY$^2$,MEAN_REG_AMT_PAST_30DAY$^3$,MEAN_HIGH_AMT_PAST_30DAY$^0$,MEAN_HIGH_AMT_PAST_30DAY$^1$,MEAN_HIGH_AMT_PAST_30DAY$^2$,MEAN_HIGH_AMT_PAST_30DAY$^3$,MEAN_INTERNET_PAST_30DAY$^0$,MEAN_INTERNET_PAST_30DAY$^1$,MEAN_INTERNET_PAST_30DAY$^2$,MEAN_INTERNET_PAST_30DAY$^3$,MEAN_LOW_AMT_PAST_30DAY$^0$,MEAN_LOW_AMT_PAST_30DAY$^1$,MEAN_LOW_AMT_PAST_30DAY$^2$,MEAN_LOW_AMT_PAST_30DAY$^3$,MEAN_MANUAL_ENTRY_PAST_30DAY$^0$,MEAN_MANUAL_ENTRY_PAST_30DAY$^1$,MEAN_MANUAL_ENTRY_PAST_30DAY$^2$,MEAN_MANUAL_ENTRY_PAST_30DAY$^3$,MEAN_PHONE_ORDER_PAST_30DAY$^0$,MEAN_PHONE_ORDER_PAST_30DAY$^1$,MEAN_PHONE_ORDER_PAST_30DAY$^2$,MEAN_PHONE_ORDER_PAST_30DAY$^3$,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^0$,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^1$,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^2$,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^3$,MEAN_PLANNED_PAST_30DAY$^0$,MEAN_PLANNED_PAST_30DAY$^1$,MEAN_PLANNED_PAST_30DAY$^2$,MEAN_PLANNED_PAST_30DAY$^3$,MEAN_SWIPE_PAST_30DAY$^0$,MEAN_SWIPE_PAST_30DAY$^1$,MEAN_SWIPE_PAST_30DAY$^2$,MEAN_SWIPE_PAST_30DAY$^3$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^0$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^1$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^2$,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^3$,MEAN_WEEKEND_PAST_30DAY$^0$,MEAN_WEEKEND_PAST_30DAY$^1$,MEAN_WEEKEND_PAST_30DAY$^2$,MEAN_WEEKEND_PAST_30DAY$^3$,MAX_AUTO_PAST_30DAY$^0$,MAX_AUTO_PAST_30DAY$^1$,MAX_AUTO_PAST_30DAY$^2$,MAX_AUTO_PAST_30DAY$^3$,MAX_LS_PAST_30DAY$^0$,MAX_LS_PAST_30DAY$^1$,MAX_LS_PAST_30DAY$^2$,MAX_LS_PAST_30DAY$^3$,MAX_ELCTRNCS_PAST_30DAY$^0$,MAX_ELCTRNCS_PAST_30DAY$^1$,MAX_ELCTRNCS_PAST_30DAY$^2$,MAX_ELCTRNCS_PAST_30DAY$^3$,MAX_FASTFOOD_PAST_30DAY$^0$,MAX_FASTFOOD_PAST_30DAY$^1$,MAX_FASTFOOD_PAST_30DAY$^2$,MAX_FASTFOOD_PAST_30DAY$^3$,MAX_HIGH_RECREA_PAST_30DAY$^0$,MAX_HIGH_RECREA_PAST_30DAY$^1$,MAX_HIGH_RECREA_PAST_30DAY$^2$,MAX_HIGH_RECREA_PAST_30DAY$^3$,MAX_MANUAL_ENTRY_PAST_30DAY$^0$,MAX_MANUAL_ENTRY_PAST_30DAY$^1$,MAX_MANUAL_ENTRY_PAST_30DAY$^2$,MAX_MANUAL_ENTRY_PAST_30DAY$^3$,MAX_PHONE_ORDER_PAST_30DAY$^0$,MAX_PHONE_ORDER_PAST_30DAY$^1$,MAX_PHONE_ORDER_PAST_30DAY$^2$,MAX_PHONE_ORDER_PAST_30DAY$^3$,MAX_PARTIAL_PAST_30DAY$^0$,MAX_PARTIAL_PAST_30DAY$^1$,MAX_PARTIAL_PAST_30DAY$^2$,MAX_PARTIAL_PAST_30DAY$^3$,MAX_RISKY_PAST_30DAY$^0$,MAX_RISKY_PAST_30DAY$^1$,MAX_RISKY_PAST_30DAY$^2$,MAX_RISKY_PAST_30DAY$^3$,MAX_WEEKEND_PAST_30DAY$^0$,MAX_WEEKEND_PAST_30DAY$^1$,MAX_WEEKEND_PAST_30DAY$^2$,MAX_WEEKEND_PAST_30DAY$^3$,STD_AUTO_PAST_30DAY$^0$,STD_AUTO_PAST_30DAY$^1$,STD_AUTO_PAST_30DAY$^2$,STD_AUTO_PAST_30DAY$^3$,STD_LS_PAST_30DAY$^0$,STD_LS_PAST_30DAY$^1$,STD_LS_PAST_30DAY$^2$,STD_LS_PAST_30DAY$^3$,STD_RECREA_PAST_30DAY$^0$,STD_RECREA_PAST_30DAY$^1$,STD_RECREA_PAST_30DAY$^2$,STD_RECREA_PAST_30DAY$^3$,STD_ELCTRNCS_PAST_30DAY$^0$,STD_ELCTRNCS_PAST_30DAY$^1$,STD_ELCTRNCS_PAST_30DAY$^2$,STD_ELCTRNCS_PAST_30DAY$^3$,STD_REG_AMT_PAST_30DAY$^0$,STD_REG_AMT_PAST_30DAY$^1$,STD_REG_AMT_PAST_30DAY$^2$,STD_REG_AMT_PAST_30DAY$^3$,STD_HIGH_RECREA_PAST_30DAY$^0$,STD_HIGH_RECREA_PAST_30DAY$^1$,STD_HIGH_RECREA_PAST_30DAY$^2$,STD_HIGH_RECREA_PAST_30DAY$^3$,STD_INTERNET_PAST_30DAY$^0$,STD_INTERNET_PAST_30DAY$^1$,STD_INTERNET_PAST_30DAY$^2$,STD_INTERNET_PAST_30DAY$^3$,STD_LOW_AMT_PAST_30DAY$^0$,STD_LOW_AMT_PAST_30DAY$^1$,STD_LOW_AMT_PAST_30DAY$^2$,STD_LOW_AMT_PAST_30DAY$^3$,STD_MANUAL_ENTRY_PAST_30DAY$^0$,STD_MANUAL_ENTRY_PAST_30DAY$^1$,STD_MANUAL_ENTRY_PAST_30DAY$^2$,STD_MANUAL_ENTRY_PAST_30DAY$^3$,STD_PHONE_ORDER_PAST_30DAY$^0$,STD_PHONE_ORDER_PAST_30DAY$^1$,STD_PHONE_ORDER_PAST_30DAY$^2$,STD_PHONE_ORDER_PAST_30DAY$^3$,STD_PARTIAL_PAST_30DAY$^0$,STD_PARTIAL_PAST_30DAY$^1$,STD_PARTIAL_PAST_30DAY$^2$,STD_PARTIAL_PAST_30DAY$^3$,STD_SWIPE_PAST_30DAY$^0$,STD_SWIPE_PAST_30DAY$^1$,STD_SWIPE_PAST_30DAY$^2$,STD_SWIPE_PAST_30DAY$^3$,STD_TRAVEL_ONLY_PAST_30DAY$^0$,STD_TRAVEL_ONLY_PAST_30DAY$^1$,STD_TRAVEL_ONLY_PAST_30DAY$^2$,STD_TRAVEL_ONLY_PAST_30DAY$^3$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^0$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^1$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^2$,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY$^3$,SUM_AUTO_PAST_30DAY$^0$,SUM_AUTO_PAST_30DAY$^1$,SUM_AUTO_PAST_30DAY$^2$,SUM_AUTO_PAST_30DAY$^3$,SUM_LS_PAST_30DAY$^0$,SUM_LS_PAST_30DAY$^1$,SUM_LS_PAST_30DAY$^2$,SUM_LS_PAST_30DAY$^3$,SUM_DISCOUNT_PAST_30DAY$^0$,SUM_DISCOUNT_PAST_30DAY$^1$,SUM_DISCOUNT_PAST_30DAY$^2$,SUM_DISCOUNT_PAST_30DAY$^3$,SUM_RECREA_PAST_30DAY$^0$,SUM_RECREA_PAST_30DAY$^1$,SUM_RECREA_PAST_30DAY$^2$,SUM_RECREA_PAST_30DAY$^3$,SUM_ELCTRNCS_PAST_30DAY$^0$,SUM_ELCTRNCS_PAST_30DAY$^1$,SUM_ELCTRNCS_PAST_30DAY$^2$,SUM_ELCTRNCS_PAST_30DAY$^3$,SUM_REG_AMT_PAST_30DAY$^0$,SUM_REG_AMT_PAST_30DAY$^1$,SUM_REG_AMT_PAST_30DAY$^2$,SUM_REG_AMT_PAST_30DAY$^3$,SUM_FASTFOOD_PAST_30DAY$^0$,SUM_FASTFOOD_PAST_30DAY$^1$,SUM_FASTFOOD_PAST_30DAY$^2$,SUM_FASTFOOD_PAST_30DAY$^3$,SUM_GAS_PAST_30DAY$^0$,SUM_GAS_PAST_30DAY$^1$,SUM_GAS_PAST_30DAY$^2$,SUM_GAS_PAST_30DAY$^3$,SUM_HIGH_AMT_PAST_30DAY$^0$,SUM_HIGH_AMT_PAST_30DAY$^1$,SUM_HIGH_AMT_PAST_30DAY$^2$,SUM_HIGH_AMT_PAST_30DAY$^3$,SUM_HIGH_RECREA_PAST_30DAY$^0$,SUM_HIGH_RECREA_PAST_30DAY$^1$,SUM_HIGH_RECREA_PAST_30DAY$^2$,SUM_HIGH_RECREA_PAST_30DAY$^3$,SUM_INTERNET_PAST_30DAY$^0$,SUM_INTERNET_PAST_30DAY$^1$,SUM_INTERNET_PAST_30DAY$^2$,SUM_INTERNET_PAST_30DAY$^3$,SUM_INTERNATIONAL_PAST_30DAY$^0$,SUM_INTERNATIONAL_PAST_30DAY$^1$,SUM_INTERNATIONAL_PAST_30DAY$^2$,SUM_INTERNATIONAL_PAST_30DAY$^3$,SUM_LOW_AMT_PAST_30DAY$^0$,SUM_LOW_AMT_PAST_30DAY$^1$,SUM_LOW_AMT_PAST_30DAY$^2$,SUM_LOW_AMT_PAST_30DAY$^3$,SUM_MANUAL_ENTRY_PAST_30DAY$^0$,SUM_MANUAL_ENTRY_PAST_30DAY$^1$,SUM_MANUAL_ENTRY_PAST_30DAY$^2$,SUM_MANUAL_ENTRY_PAST_30DAY$^3$,SUM_PHONE_ORDER_PAST_30DAY$^0$,SUM_PHONE_ORDER_PAST_30DAY$^1$,SUM_PHONE_ORDER_PAST_30DAY$^2$,SUM_PHONE_ORDER_PAST_30DAY$^3$,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^0$,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^1$,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^2$,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY$^3$,SUM_SWIPE_PAST_30DAY$^0$,SUM_SWIPE_PAST_30DAY$^1$,SUM_SWIPE_PAST_30DAY$^2$,SUM_SWIPE_PAST_30DAY$^3$,SUM_TRAVEL_ONLY_PAST_30DAY$^0$,SUM_TRAVEL_ONLY_PAST_30DAY$^1$,SUM_TRAVEL_ONLY_PAST_30DAY$^2$,SUM_TRAVEL_ONLY_PAST_30DAY$^3$,SUM_WEEKEND_PAST_30DAY$^0$,SUM_WEEKEND_PAST_30DAY$^1$,SUM_WEEKEND_PAST_30DAY$^2$,SUM_WEEKEND_PAST_30DAY$^3$'
    # tg_filted = 'AVAIL_CRDT,AMOUNT,CREDIT_LIMIT,CARD_NOT_PRESENT,FLAG_LX,FLAG_ATM,FLAG_AUTO,FLAG_CASH,FLAG_LS,FLAG_DISCOUNT,FLAG_RECREA,FLAG_ELCTRNCS,FLAG_REG_AMT,FLAG_FASTFOOD,FLAG_GAS,FLAG_HIGH_AMT,FLAG_HIGH_RECREA,FLAG_INTERNET,FLAG_INTERNATIONAL,FLAG_JEWELRY,FLAG_LOW_AMT,FLAG_MANUAL_ENTRY,FLAG_PHONE_ORDER,FLAG_PURCHASE_EXCLUDING_GAS,FLAG_PLANNED,FLAG_RISKY,FLAG_SWIPE,FLAG_TRAVEL_ONLY,FLAG_TRAVEL_AND_ENTERTAINMENT,FLAG_WEEKEND,MEAN_AUTO_PAST_7DAY,MEAN_LS_PAST_7DAY,MEAN_RECREA_PAST_7DAY,MEAN_REG_AMT_PAST_7DAY,MEAN_FASTFOOD_PAST_7DAY,MEAN_HIGH_AMT_PAST_7DAY,MEAN_HIGH_RECREA_PAST_7DAY,MEAN_INTERNET_PAST_7DAY,MEAN_INTERNATIONAL_PAST_7DAY,MEAN_JEWELRY_PAST_7DAY,MEAN_LOW_AMT_PAST_7DAY,MEAN_MANUAL_ENTRY_PAST_7DAY,MEAN_PHONE_ORDER_PAST_7DAY,MEAN_PLANNED_PAST_7DAY,MEAN_SWIPE_PAST_7DAY,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,MEAN_WEEKEND_PAST_7DAY,MAX_CASH_PAST_7DAY,MAX_LS_PAST_7DAY,MAX_RECREA_PAST_7DAY,MAX_HIGH_AMT_PAST_7DAY,MAX_HIGH_RECREA_PAST_7DAY,MAX_INTERNET_PAST_7DAY,MAX_PHONE_ORDER_PAST_7DAY,MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY,MAX_SWIPE_PAST_7DAY,MAX_WEEKEND_PAST_7DAY,SUM_LX_PAST_7DAY,SUM_AUTO_PAST_7DAY,SUM_LS_PAST_7DAY,SUM_RECREA_PAST_7DAY,SUM_GAS_PAST_7DAY,SUM_HIGH_AMT_PAST_7DAY,SUM_INTERNET_PAST_7DAY,SUM_INTERNATIONAL_PAST_7DAY,SUM_LOW_AMT_PAST_7DAY,SUM_MANUAL_ENTRY_PAST_7DAY,SUM_PHONE_ORDER_PAST_7DAY,SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY,SUM_PARTIAL_PAST_7DAY,SUM_PLANNED_PAST_7DAY,SUM_SWIPE_PAST_7DAY,SUM_WEEKEND_PAST_7DAY,COUNT_AUTO_PAST_7DAY,COUNT_ELCTRNCS_PAST_7DAY,COUNT_GAS_PAST_7DAY,COUNT_HIGH_AMT_PAST_7DAY,COUNT_INTERNET_PAST_7DAY,COUNT_LOW_AMT_PAST_7DAY,COUNT_MANUAL_ENTRY_PAST_7DAY,COUNT_PHONE_ORDER_PAST_7DAY,COUNT_PURCHASE_EXCLUDING_GAS_PAST_7DAY,COUNT_SWIPE_PAST_7DAY,COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY,COUNT_WEEKEND_PAST_7DAY,MEAN_AUTO_PAST_30DAY,MEAN_DISCOUNT_PAST_30DAY,MEAN_RECREA_PAST_30DAY,MEAN_ELCTRNCS_PAST_30DAY,MEAN_REG_AMT_PAST_30DAY,MEAN_HIGH_AMT_PAST_30DAY,MEAN_INTERNET_PAST_30DAY,MEAN_LOW_AMT_PAST_30DAY,MEAN_MANUAL_ENTRY_PAST_30DAY,MEAN_PHONE_ORDER_PAST_30DAY,MEAN_PURCHASE_EXCLUDING_GAS_PAST_30DAY,MEAN_PLANNED_PAST_30DAY,MEAN_SWIPE_PAST_30DAY,MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,MEAN_WEEKEND_PAST_30DAY,MAX_AUTO_PAST_30DAY,MAX_LS_PAST_30DAY,MAX_ELCTRNCS_PAST_30DAY,MAX_FASTFOOD_PAST_30DAY,MAX_HIGH_RECREA_PAST_30DAY,MAX_MANUAL_ENTRY_PAST_30DAY,MAX_PHONE_ORDER_PAST_30DAY,MAX_PARTIAL_PAST_30DAY,MAX_RISKY_PAST_30DAY,MAX_WEEKEND_PAST_30DAY,STD_AUTO_PAST_30DAY,STD_LS_PAST_30DAY,STD_RECREA_PAST_30DAY,STD_ELCTRNCS_PAST_30DAY,STD_REG_AMT_PAST_30DAY,STD_HIGH_RECREA_PAST_30DAY,STD_INTERNET_PAST_30DAY,STD_LOW_AMT_PAST_30DAY,STD_MANUAL_ENTRY_PAST_30DAY,STD_PHONE_ORDER_PAST_30DAY,STD_PARTIAL_PAST_30DAY,STD_SWIPE_PAST_30DAY,STD_TRAVEL_ONLY_PAST_30DAY,STD_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,SUM_AUTO_PAST_30DAY,SUM_LS_PAST_30DAY,SUM_DISCOUNT_PAST_30DAY,SUM_RECREA_PAST_30DAY,SUM_ELCTRNCS_PAST_30DAY,SUM_REG_AMT_PAST_30DAY,SUM_FASTFOOD_PAST_30DAY,SUM_GAS_PAST_30DAY,SUM_HIGH_AMT_PAST_30DAY,SUM_HIGH_RECREA_PAST_30DAY,SUM_INTERNET_PAST_30DAY,SUM_INTERNATIONAL_PAST_30DAY,SUM_LOW_AMT_PAST_30DAY,SUM_MANUAL_ENTRY_PAST_30DAY,SUM_PHONE_ORDER_PAST_30DAY,SUM_PURCHASE_EXCLUDING_GAS_PAST_30DAY,SUM_SWIPE_PAST_30DAY,SUM_TRAVEL_ONLY_PAST_30DAY,SUM_WEEKEND_PAST_30DAY,COUNT_AUTO_PAST_30DAY,COUNT_RECREA_PAST_30DAY,COUNT_REG_AMT_PAST_30DAY,COUNT_FASTFOOD_PAST_30DAY,COUNT_GAS_PAST_30DAY,COUNT_HIGH_AMT_PAST_30DAY,COUNT_INTERNET_PAST_30DAY,COUNT_LOW_AMT_PAST_30DAY,COUNT_MANUAL_ENTRY_PAST_30DAY,COUNT_PHONE_ORDER_PAST_30DAY,COUNT_PURCHASE_EXCLUDING_GAS_PAST_30DAY,COUNT_PLANNED_PAST_30DAY,COUNT_SWIPE_PAST_30DAY,COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY,COUNT_WEEKEND_PAST_30DAY'
    
    tags = tg_lg.split(',')

    # tags = ['AMOUNT', 'CARD_NOT_PRESENT', 'FLAG_ATM', 'FLAG_CASH', 'FLAG_DISCOUNT', 'FLAG_ELCTRNCS', 'FLAG_FASTFOOD', 'FLAG_HIGH_AMT', 'FLAG_INTERNET', 'FLAG_INTERNATIONAL', 'FLAG_LOW_AMT', 'FLAG_PHONE_ORDER', 'FLAG_PLANNED', 'FLAG_SWIPE', 'FLAG_TRAVEL_AND_ENTERTAINMENT', 'MEAN_AUTO_PAST_7DAY', 'MEAN_RECREA_PAST_7DAY', 'MEAN_FASTFOOD_PAST_7DAY', 'MEAN_HIGH_RECREA_PAST_7DAY', 'MEAN_INTERNATIONAL_PAST_7DAY', 'MEAN_LOW_AMT_PAST_7DAY', 'MEAN_PHONE_ORDER_PAST_7DAY', 'MEAN_SWIPE_PAST_7DAY', 'MEAN_WEEKEND_PAST_7DAY', 'MAX_LS_PAST_7DAY', 'MAX_HIGH_AMT_PAST_7DAY', 'MAX_INTERNET_PAST_7DAY', 'MAX_PURCHASE_EXCLUDING_GAS_PAST_7DAY', 'MAX_WEEKEND_PAST_7DAY', 'STD_FASTFOOD_PAST_7DAY', 'STD_INTERNET_PAST_7DAY', 'STD_PURCHASE_EXCLUDING_GAS_PAST_7DAY', 'STD_TRAVEL_AND_ENTERTAINMENT_PAST_7DAY', 'SUM_AUTO_PAST_7DAY', 'SUM_RECREA_PAST_7DAY', 'SUM_HIGH_AMT_PAST_7DAY', 'SUM_INTERNATIONAL_PAST_7DAY', 'SUM_MANUAL_ENTRY_PAST_7DAY', 'SUM_PURCHASE_EXCLUDING_GAS_PAST_7DAY', 'SUM_PLANNED_PAST_7DAY', 'SUM_WEEKEND_PAST_7DAY', 'COUNT_ELCTRNCS_PAST_7DAY', 'COUNT_HIGH_AMT_PAST_7DAY', 'COUNT_LOW_AMT_PAST_7DAY', 'COUNT_PHONE_ORDER_PAST_7DAY', 'COUNT_SWIPE_PAST_7DAY', 'COUNT_WEEKEND_PAST_7DAY', 'MEAN_DISCOUNT_PAST_30DAY', 'MEAN_ELCTRNCS_PAST_30DAY', 'MEAN_HIGH_AMT_PAST_30DAY', 'MEAN_LOW_AMT_PAST_30DAY', 'MEAN_PHONE_ORDER_PAST_30DAY', 'MEAN_PLANNED_PAST_30DAY', 'MEAN_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY', 'MAX_AUTO_PAST_30DAY', 'MAX_ELCTRNCS_PAST_30DAY', 'MAX_HIGH_RECREA_PAST_30DAY', 'MAX_PHONE_ORDER_PAST_30DAY', 'MAX_RISKY_PAST_30DAY', 'STD_AUTO_PAST_30DAY', 'STD_RECREA_PAST_30DAY', 'STD_REG_AMT_PAST_30DAY', 'STD_INTERNET_PAST_30DAY', 'STD_MANUAL_ENTRY_PAST_30DAY', 'STD_PARTIAL_PAST_30DAY', 'STD_TRAVEL_ONLY_PAST_30DAY', 'SUM_AUTO_PAST_30DAY', 'SUM_DISCOUNT_PAST_30DAY', 'SUM_ELCTRNCS_PAST_30DAY', 'SUM_FASTFOOD_PAST_30DAY', 'SUM_HIGH_AMT_PAST_30DAY', 'SUM_INTERNET_PAST_30DAY', 'SUM_LOW_AMT_PAST_30DAY', 'SUM_PHONE_ORDER_PAST_30DAY', 'SUM_SWIPE_PAST_30DAY', 'SUM_WEEKEND_PAST_30DAY', 'COUNT_RECREA_PAST_30DAY', 'COUNT_FASTFOOD_PAST_30DAY', 'COUNT_HIGH_AMT_PAST_30DAY', 'COUNT_LOW_AMT_PAST_30DAY', 'COUNT_PHONE_ORDER_PAST_30DAY', 'COUNT_PLANNED_PAST_30DAY', 'COUNT_TRAVEL_AND_ENTERTAINMENT_PAST_30DAY']

   

    fraud = 'FRAUD_FLAG'
    transation_id = 'TRANSACTION_ID'

    print('reading...')
    
    if mode == 'train':
        if not read_new_file:
            x_lst = []
            
            with open('./processed_data/labels_lst.txt', 'r') as f:
                labels_lst = json.loads(f.read())

            with open('./processed_data/y_lst.txt', 'r') as f:
                y_lst = json.loads(f.read())

        else:
            x_lst, labels_lst, y_lst = read_csv_file(path, transation_id, tags, fraud, read_method, mode)
    
    
    else:
        x_lst, labels_lst, y_lst = read_csv_file(path, transation_id, tags, fraud,read_method, mode)


    if read_method != 'read_processed_datalst':
        print('normalizing...')
        labels_lst_normalized = normalize(labels_lst)

    else:
        labels_lst_normalized = standarlize(labels_lst)


    print('reformating...')
    x_vec_lst = reformat(labels_lst_normalized)



    


    print(f'{mode}ing...')

    # for i in trange(80, 90, 2):
    #     print('threshold:---------', i/100)
    #     model(x_vec_lst, y_lst, mode, model_code, model_type, x_lst, tags, i/100)
    model(x_vec_lst, y_lst, mode, model_code, model_type, x_lst, tags, threshold)
        

