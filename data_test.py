from ast import Or
from multiprocessing import Pipe
import numpy as np
import pandas as pd
from scipy import rand
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


data = pd.read_csv(r'data\heart.csv') 
# (data.corr()['output'].sort_values().plot.barh())
(data.corr()['output'].abs().sort_values().plot.barh())
# plt.show()
# exng       -0.436757
# oldpeak    -0.430696
# caa        -0.391724
# thall      -0.344029
# sex        -0.280937
# age        -0.225439
# trtbps     -0.144931
# chol       -0.085239
# fbs        -0.028046
# restecg     0.137230
# slp         0.345877
# thalachh    0.421741
# cp          0.433798
# output      1.000000


# print(data.isnull().sum()) - no missing data

# Build a data enhancer

def data_enhance(data):
    org_data = data
    for sex in data['sex'].unique():
        sex_data = org_data[org_data['sex']==sex]
        age_std = sex_data['age'].std()
        trtbps_std = sex_data['trtbps'].std()
        chol_std = sex_data['chol'].std()
        thalachh_std = sex_data['thalachh'].std()
        oldpeak_std = sex_data['oldpeak'].std()
        for i in org_data[org_data['sex']==sex].index:
            if np.random.randint(2) == 1:
                org_data['age'].values[i] += age_std/10
                org_data['trtbps'].values[i] += trtbps_std/10
                org_data['chol'].values[i] += chol_std/10
                org_data['thalachh'].values[i] += thalachh_std/10
                org_data['oldpeak'].values[i] += oldpeak_std/10
            else:
                org_data['age'].values[i] -= age_std/10
                org_data['trtbps'].values[i] -= trtbps_std/10
                org_data['chol'].values[i] -= chol_std/10
                org_data['thalachh'].values[i] -= thalachh_std/10
                org_data['oldpeak'].values[i] -= oldpeak_std/10

    return org_data

gen = data_enhance(data)
x = data.drop(['output'], axis=1) # features - train and val data
y = data['output'] # target

num_vals = ['age', 'trtbps','thalachh', 'chol', 'oldpeak']
cat_vals = ['sex', 'cp', 'exng', 'slp', 'caa', 'thall', 'restecg', 'fbs']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Add enhanced data to 20% of the orig data
enhanced_sample = gen.sample(gen.shape[0] // 5)
x_train = pd.concat([x_train, enhanced_sample.drop(['output'], axis=1 ) ])
y_train = pd.concat([y_train, enhanced_sample['output'] ])


# print(x_train)
# print(y_train)

# Make pipelines and transform
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))])

tree_pipe = ColumnTransformer([('num', num_pipeline, num_vals), ('cat', cat_pipeline, cat_vals)], remainder='passthrough')

# Different classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0, n_estimators=100),
    "Ada Boost": AdaBoostClassifier(random_state=0, n_estimators=100),
    "Extra Trees": ExtraTreesClassifier(random_state=0, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0, n_estimators=100),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0, n_estimators=100),
    "Catboost": CatBoostClassifier(random_state=0, n_estimators=100),
    "Logistic Regression": LogisticRegression(random_state=0)
}

classifiers = {name: make_pipeline(tree_pipe, model) for name, model in classifiers.items()}

# Results df
results = pd.DataFrame({'Model': [], "Accuracy Score": [], "Balanced Accuracy score": [], "Time": []})



for model_name, model in classifiers.items():
    start_time = time.time()

    model.fit(x_train, y_train)

    predics = model.predict(x_val)
    total_time = time.time() - start_time
    


    results = results.append({"Model": model_name,
                            "Accuracy Score": accuracy_score(y_val, predics)*100,
                            "Balanced Accuracy score": balanced_accuracy_score(y_val, predics)*100,
                            "Time": total_time}, ignore_index=True)

results_order = results.sort_values(by=['Accuracy Score'], ascending=False, ignore_index=True)

# print(results_order)

def predictor(features):

    best_model = classifiers.get("Extra Trees")

    best_model.fit(x_train, y_train)

    preds = best_model.predict(features)
    return preds


# STD
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0          Extra Trees       93.442623                93.355120  0.106715
1        Random Forest       90.163934                90.032680  0.180518
2            Ada Boost       90.163934                90.413943  0.082954
3    Gradient Boosting       88.524590                88.562092  0.082810
4             Catboost       88.524590                88.562092  2.056365
5              XGBoost       86.885246                87.091503  0.111737
6             LightGBM       86.885246                87.091503  0.083491
7  Logistic Regression       86.885246                86.328976  0.022185
8        Decision Tree       77.049180                77.505447  0.020820
"""

# MEAN
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0          Extra Trees       93.442623                93.736383  0.109719
1        Random Forest       91.803279                91.503268  0.185502
2    Gradient Boosting       91.803279                91.884532  0.082811
3             Catboost       91.803279                91.503268  2.230707
4        Decision Tree       88.524590                88.180828  0.012965
5              XGBoost       88.524590                88.180828  0.114693
6             LightGBM       88.524590                88.180828  0.075797
7  Logistic Regression       86.885246                86.328976  0.019947
8            Ada Boost       85.245902                85.239651  0.076826
"""
# MEDIAN
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0            Ada Boost       91.803279                91.503268  0.088764
1        Random Forest       86.885246                87.091503  0.178659
2          Extra Trees       86.885246                86.710240  0.107744
3              XGBoost       86.885246                86.710240  0.112697
4             Catboost       86.885246                86.710240  2.147449
5    Gradient Boosting       85.245902                85.239651  0.087765
6             LightGBM       85.245902                85.239651  0.076795
7  Logistic Regression       83.606557                83.006536  0.017951
8        Decision Tree       78.688525                78.976035  0.012965
"""

# Without enhancement
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0          Extra Trees       85.245902                84.477124  0.112150
1  Logistic Regression       85.245902                84.477124  0.030931
2            Ada Boost       83.606557                83.006536  0.078821
3             Catboost       83.606557                82.625272  2.205500
4        Random Forest       80.327869                79.302832  0.189494
5    Gradient Boosting       78.688525                78.213508  0.082811
6              XGBoost       78.688525                77.832244  0.111701
7        Decision Tree       75.409836                76.416122  0.013963
8             LightGBM       75.409836                74.891068  0.070812
"""