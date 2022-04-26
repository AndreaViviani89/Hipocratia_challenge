from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting 
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier

from sklearn import impute
from sklearn import pipeline
from sklearn import compose
from sklearn.preprocessing import OrdinalEncoder


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

    num_vars = ['age','trtbps','chol','thalachh','oldpeak']
    cat_vars = ['sex', 'cp', 'fbs', 'restecg','exng','slp', 'caa', 'thall']


    num_4_treeModels = pipeline.Pipeline(steps=[
        ('imputer', impute.SimpleImputer(strategy='constant', fill_value=-9999)),])
    cat_4_treeModels = pipeline.Pipeline(steps=[('ordinal', OrdinalEncoder())])

    tree_prepro = compose.ColumnTransformer(transformers=[
        ('num', num_4_treeModels,num_vars),
        ('cat', cat_4_treeModels, cat_vars),
    ], remainder='drop') 

    tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
    return tree_classifiers
