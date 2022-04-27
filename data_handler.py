import numpy    as np
import pandas   as pd
import seaborn  as sns
import matplotlib.pyplot as plt
import time
import joblib


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier





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
#plt.show()

plt.figure(figsize = (10, 8))
sns.countplot(x ='output', data = data, hue = "output")
plt.title("Distribution of the target variable", fontsize = 20)
#plt.show()


# plot correlation
data.corr()['output'].abs().sort_values().plot.barh()
#plt.show()


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



# select best feature
data = data.copy()
x = data.drop(['output'], axis = 1) # features - train and val data
y = data['output'] # target


# apply SelectKBest class to extract top best features
bestfeatures = SelectKBest(score_func = chi2, k = 10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns) 
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Specs', 'Score']  
#print(featureScores.nlargest(15, 'Score'))  



"""
Specs       Score
7   thalachh  188.320472
13    output  138.000000
9    oldpeak   72.644253
11       caa   66.440765
2         cp   62.598098
8       exng   38.914377
4       chol   23.936394
0        age   23.286624
3     trtbps   14.823925
10       slp    9.804095
1        sex    7.576835
12     thall    5.791853
6    restecg    2.978271
5        fbs    0.202934
"""




# feature generation and feature selection
data['nor_press'] = data['trtbps'] / 120
data.drop('fbs', axis = 1, inplace = True)
data.drop(164)
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
num_vals = ['age', 'trtbps', 'chol', 'nor_press', 'thalachh', 'oldpeak']
cat_vals = ['sex', 'cp', 'exng', 'restecg', 'slp', 'caa', 'thall']



# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0)

df = x_val.sample(n=6)
# store data
df.to_csv('data_test.csv', index=False)




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

#print(results_order)





# final model
best_model = classifiers.get("Random Forest")

best_model.fit(x_train, y_train)

preds = best_model.predict(x_val)
    
    
    


# Saving model
joblib.dump(best_model, 'best_model.joblib')




# create test data

df = data.sample(n=6)
df = df.drop('output', axis=1)
df.to_csv('test.csv')



