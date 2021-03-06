from pickle import TRUE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tree_class_aug as tr
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('heart.csv')


df['nor_press'] = df['trtbps'] / 120

np.random.seed(0)

def data_aug(df):
    
    dfc = df.copy()

    for out in dfc['output'].unique():
        ha = dfc[dfc['output'] == out]
        trtbs_std = ha['trtbps'].std()
        chol_std = ha['chol'].std()
        thalachh_std = ha['thalachh'].std()
        age_std = ha['age'].std()
            
        for j in dfc[dfc['output'] == out].index:
            if np.random.randint(2) == 1:
                dfc['trtbps'].values[j] +=trtbs_std/10
            else:
                dfc['trtbps'].values[j] -= trtbs_std/10

            if np.random.randint(2) == 1:
                dfc['chol'].values[j] += chol_std/10
            else:
                dfc['chol'].values[j] -= chol_std/10

            if np.random.randint(2) == 1:
                dfc['thalachh'].values[j] += thalachh_std/10
            else:
                dfc['thalachh'].values[j] += thalachh_std/10
            
            if np.random.randint(2) ==1:
                dfc['age'].values[j] += age_std/10
            else:
                dfc['age'].values[j] -= age_std/10
    
    return dfc
cdf = data_aug(df)


X,y = df.drop(['output'], axis=1), df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
extra_sample = cdf.sample(cdf.shape[0] // 5)
X_train = pd.concat([X_train, extra_sample.drop(['output'], axis=1 ) ])
y_train = pd.concat([y_train, extra_sample['output'] ])

tree_classifiers = tr.tree_classifiers()

def model_results():
    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in tree_classifiers.items():
        start_time = time.time()        
        model.fit(X_train,y_train)
        pred =model.predict(X_test)

        total_time = time.time() - start_time

        results = results.append({"Model":    model_name,
                                "Accuracy": accuracy_score(y_test, pred)*100,
                                "Bal Acc.": balanced_accuracy_score(y_test, pred)*100,
                                "Time":     total_time},
                                ignore_index=True)
                                
    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
    return results_ord
mod_result = model_results()
print(mod_result)


# mod = RandomForestClassifier(random_state=0, max_depth=4, n_estimators=200)
# mod.fit(X_train, y_train)
# prediction1 = mod.predict(X_test)
# print('Confusion matrix',confusion_matrix(y_test,prediction1))
# print('Classification report:', classification_report(y_test, prediction1))
# from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, recall_score
# import matplotlib.pyplot as plt
# plot_confusion_matrix(tree_classifiers["Extra Trees"], X_test, y_test,cmap="RdPu")
# plt.show()