import numpy    as np
import pandas   as pd
import seaborn  as sns
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer




# Dataset columns:

"""
1. age: The person's age in years
2. sex: The persons sex (1 = male, 0 = female)
3. cp: chest pain type
— Value 0: asymptomatic
— Value 1: atypical angina
— Value 2: non-anginal pain
— Value 3: typical angina
4. trestbps: The persons resting blood pressure (mm Hg on admission to the hospital)
5. chol: The person's cholesterol measurement in mg/dl
6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
7. restecg: resting electrocardiographic results
— Value 0: showing probable or definite left ventricular hypertrophy by Estes' criteria
— Value 1: normal
— Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
8. thalach: The person's maximum heart rate achieved
9. exang: Exercise induced angina (1 = yes; 0 = no)
10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
11. slope: the slope of the peak exercise ST segment — 0: downsloping; 1: flat; 2: upsloping
0: downsloping; 1: flat; 2: upsloping
12. ca: The number of major vessels (0 - 3)
13. thal: A blood disorder called thalassemia Value 0: NULL (dropped from the dataset previously
- Value 1: fixed defect (no blood flow in some part of the heart)
- Value 2: normal blood flow
- Value 3: reversible defect (a blood flow is observed but it is not normal)
14. target: Heart disease (1 = no, 0 = yes)

"""




# read data
data = pd.read_csv("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\09. Challenge\\data\\heart.csv")
#print("Data Shape: {}".format(data.shape))

np.random.seed(0)


#visualizing Null values if it exists 
plt.figure(figsize = (22,10))
plt.xticks(size = 20,color ='grey')
plt.tick_params(size = 12,color ='grey')
plt.title('Finding Null Values Using Heatmap\n',color ='grey',size = 30)
sns.heatmap(data.isnull(),
            yticklabels = False,
            cbar = False,
            cmap = 'PuBu_r',
            )
plt.show()




# plot correlation
data.corr()['output'].abs().sort_values().plot.barh()
plt.show()


# correlation matrix values
"""
exng       -0.436757
oldpeak    -0.430696
caa        -0.391724
thall      -0.344029
sex        -0.280937
age        -0.225439
trtbps     -0.144931
chol       -0.085239
fbs        -0.028046
restecg     0.137230
slp         0.345877
thalachh    0.421741
cp          0.433798
output      1.000000
"""




# Take a look at nulls 0 nulls
#print(data.isnull().sum()) 



# feature generation and feature selection
#data['oldpeak'] = data['oldpeak'].apply(lambda x: 1 if x > 1.6 else 0)

data.drop('fbs', axis = 1, inplace = True)
data.drop('restecg', axis = 1, inplace = True)
#print(data.sample(10))




# Build a data enhancer
def data_enhance(data):

    org_data = data.copy()

    for sex in data['sex'].unique():
        sex_data = org_data[org_data['sex'] == sex]
        age_std1 = sex_data['chol'].std()
        age_std = sex_data['age'].std()
        trtbps_std = sex_data['trtbps'].std()
        thalachh_std = sex_data['thalachh'].std()
        oldpeak_std = sex_data['oldpeak'].std()


        for i in sex_data.index:

            if np.random.randint(2) == 1:
                org_data['age'].values[i] += age_std/10
                org_data['trtbps'].values[i] += trtbps_std/10
                org_data['thalachh'].values[i] += thalachh_std/10
                org_data['oldpeak'].values[i] += oldpeak_std/10
                org_data['chol'].values[i] += age_std1/10
            else:
                org_data['age'].values[i] -= age_std/10
                org_data['trtbps'].values[i] -= trtbps_std/10
                org_data['thalachh'].values[i] -= thalachh_std/10
                org_data['oldpeak'].values[i] -= oldpeak_std/10
                org_data['chol'].values[i] -= age_std1/10

    return org_data



gen = data_enhance(data)
x = data.drop(['output'], axis = 1) # features - train and val data
y = data['output'] # target



# numerical and categorical data
num_vals = ['age', 'trtbps', 'chol', 'thalachh','oldpeak']
cat_vals = ['sex', 'cp', 'exng', 'slp', 'caa', 'thall']



# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0)




# Add enhanced data to 20% of the orig data
enhanced_sample = gen.sample(gen.shape[0] // 5)
x_train = pd.concat([x_train, enhanced_sample.drop(['output'], axis = 1 ) ])
y_train = pd.concat([y_train, enhanced_sample['output'] ])


# print(x_train)
# print(y_train)




# Make pipelines and transform
num_pipeline = Pipeline([('scaler', StandardScaler())])
#cat_pipeline = Pipeline([('ordinal', OrdinalEncoder())])

tree_pipe = ColumnTransformer([
                                ('num', num_pipeline, num_vals),
                                ], 
                                remainder ='passthrough')




# Different classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "Ada Boost": AdaBoostClassifier(random_state=0),
    "Extra Trees": ExtraTreesClassifier(random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0),
    "Catboost": CatBoostClassifier(random_state=0),
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
                            "Time": total_time}, ignore_index = True)

results_order = results.sort_values(by = ['Accuracy Score'], ascending = False, ignore_index = True)

print(results_order)





# final model
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


# after add feature

"""
                Model  Accuracy Score  Balanced Accuracy score      Time
0        Random Forest       91.803279                91.884532  0.255021
1    Gradient Boosting       91.803279                91.884532  0.137404
2             Catboost       91.803279                91.884532  3.175197
3            Ada Boost       90.163934                90.032680  0.147018
4  Logistic Regression       90.163934                90.032680  0.091564
5          Extra Trees       88.524590                88.562092  0.207016
6             LightGBM       88.524590                88.943355  0.102990
7              XGBoost       86.885246                87.472767  0.156011
8        Decision Tree       83.606557                83.387800  0.033862
"""



# after feature selection

"""
                Model  Accuracy Score  Balanced Accuracy score      Time
0  Logistic Regression       91.803279                91.884532  0.039007
1          Extra Trees       90.163934                90.413943  0.156297
2    Gradient Boosting       90.163934                90.795207  0.109625
3             LightGBM       90.163934                90.795207  0.044105
4             Catboost       90.163934                90.795207  2.726169
5        Random Forest       86.885246                87.091503  0.219036
6              XGBoost       86.885246                87.854031  0.113341
7            Ada Boost       85.245902                86.383442  0.109190
8        Decision Tree       80.327869                81.590414  0.015628
"""