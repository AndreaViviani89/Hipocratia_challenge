from http.client import UnimplementedFileMode
from pickle import TRUE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tree_classifier as tr
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score

df = pd.read_csv('heart.csv')
X,y = df.drop(['output'], axis=1), df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 ,shuffle=True)
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
mod_res = model_results()
print(mod_res)
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, recall_score
import matplotlib.pyplot as plt
plot_confusion_matrix(tree_classifiers["Extra Trees"], X_test, y_test,cmap="RdPu")
plt.show()

  
