
import pandas as pd
import joblib
from joblib import load


m = load("C:\\Users\\ritth\\code\\Strive\\Hipocratia_challenge\\image\\best_model.joblib")
def get_inputs():

    input_features = []

    age = int(input("How old are you? \n"))
    sex = int(input("Gender? 0 for Female, 1 for Male \n"))
    cp = int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n"))
    trtbps = int(input("Resting blood pressure in mm Hg \n"))
    chol = int(input("Serum cholestrol in mg/dl \n"))
    fbs = int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n"))
    restecg = int(input("Resting ecg? (0,1,2) \n"))
    thalachh = int(input("Maximum Heart Rate achieved? \n"))
    exng = int(input("Exercise Induced Angina? 0 for no, 1 for yes \n"))
    oldpeak = float(input("Old Peak? ST Depression induced by exercise relative to rest \n"))
    slp = int(input("Slope of the peak? (0,1,2) \n"))
    caa = int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n"))
    thall = int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n"))

    input_features.append([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])

    return pd.DataFrame(input_features, columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])




pred = m(get_inputs())
if pred == 1:
    print("You may have risk of heart attack")
else:
    print("No risk of heart attack")
    


















# parser = argparse.ArgumentParser()
# for item in features:
#     parser.add_argument(item, type=float, help=item)

# args = parser.parse_args()
# x_features = [int(input("How old are you? \n")),int(input("Gender? 0 for Female, 1 for Male \n")),int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n")),int(input("Resting blood pressure in mm Hg \n")),int(input("Serum cholestrol in mg/dl \n")),int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n")),int(input("Resting ecg? (0,1,2) \n")),int(input("Maximum Heart Rate achieved? \n")),int(input("Exercise Induced Angina? 0 for no, 1 for yes \n")),float(input("Old Peak? ST Depression induced by exercise relative to rest \n")),int(input("Slope of the peak? (0,1,2) \n")),int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n")),int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n")) ]

