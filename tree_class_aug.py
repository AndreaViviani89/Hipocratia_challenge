from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier

from sklearn import impute
from sklearn import pipeline
from sklearn import compose
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from pandas import MultiIndex, Int64Index

def tree_classifiers():
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Skl GBM": GradientBoostingClassifier(),
    "Skl HistGBM":HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier()}
#criterion='entropy', max_depth=4, n_estimators=91,random_state=42
    num_vars = ['age','trtbps','chol','thalachh','oldpeak','nor_press']
    cat_vars = ['sex', 'cp','exng','fbs','restecg','exng','slp', 'caa', 'thall']


    num_4_treeModels = pipeline.Pipeline(steps=[('scaler', StandardScaler())])
    #cat_4_treeModels = pipeline.Pipeline(steps=[('ordinal', OrdinalEncoder())])

    tree_prepro = compose.ColumnTransformer(transformers=[
        ('num', num_4_treeModels,num_vars),
        #('cat', cat_4_treeModels, cat_vars),
    ], remainder='passthrough') 

    tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
    return tree_classifiers
