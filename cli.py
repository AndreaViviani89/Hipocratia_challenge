import joblib
from data_handler import predictor
import pandas as pd
from joblib import load
import argparse




# def get_inputs():

#     input_features = []
#     age = int(input("How old are you? \n"))
#     sex = int(input("Gender? 0 for Female, 1 for Male \n"))
#     cp = int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n"))
#     trtbps = int(input("Resting blood pressure in mm Hg \n"))
#     chol = int(input("Serum cholestrol in mg/dl \n"))
#     fbs = int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n"))
#     restecg = int(input("Resting ecg? (0,1,2) \n"))
#     thalachh = int(input("Maximum Heart Rate achieved? \n"))
#     exng = int(input("Exercise Induced Angina? 0 for no, 1 for yes \n"))
#     oldpeak = float(input("Old Peak? ST Depression induced by exercise relative to rest - Float number \n"))
#     slp = int(input("Slope of the peak? (0,1,2) \n"))
#     caa = int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n"))
#     thall = int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n"))

#     input_features.append([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])

#     return pd.DataFrame(input_features, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])




# pred = predictor(get_inputs())
# if pred == 1:
#     print("You may have risk of heart attack")
# else:
#     print("No risk of heart attack")
    


# Remember
# we need to run the cli file from the terminal


# Test with argparse


model = joblib.load('best_model.joblib')

arg_parse = argparse.ArgumentParser(description= 'Predict an Heart Attack')

arg_parse.add_argument('age', type=int, help='How old are you?')
arg_parse.add_argument('sex', type=int, choices=[0, 1] , help='Gender? 0 for Female, 1 for Male')
arg_parse.add_argument('cp', type=int, choices=[1, 2, 3, 4] , help='Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain')
arg_parse.add_argument('trtbps', type=int, help='Resting blood pressure in mm Hg')
arg_parse.add_argument('chol', type=int, help='Serum cholestrol in mg/dl')
arg_parse.add_argument('fbs', type=int, choices=[0, 1] , help='Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl')
arg_parse.add_argument('restecg', type=int, choices=[0, 1, 2] , help='Resting ecg? (0,1,2)')
arg_parse.add_argument('thalachh', type=int, help='Maximum Heart Rate achieved?')
arg_parse.add_argument('exng', type=int, choices=[0, 1] , help='Exercise Induced Angina? 0 for no, 1 for yes')
arg_parse.add_argument('oldpeak', type=float, help='Old Peak? ST Depression induced by exercise relative to rest - Float number')
arg_parse.add_argument('slp', type=int, choices=[0, 1, 2] , help='Slope of the peak? (0,1,2)')
arg_parse.add_argument('caa', type=int, choices=[0, 1, 2, 3] , help='Number of colored vessels during Floroscopy? (0,1,2,3)')
arg_parse.add_argument('thall', type=int, choices=[0, 1, 2, 3] , help='3 = normal; 6 = fixed defect; 7 = reversable defect')
args = arg_parse.parse_args()







# parser = argparse.ArgumentParser()
# for item in features:
#     parser.add_argument(item, type=float, help=item)

# args = parser.parse_args()
# x_features = [int(input("How old are you? \n")),int(input("Gender? 0 for Female, 1 for Male \n")),int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n")),int(input("Resting blood pressure in mm Hg \n")),int(input("Serum cholestrol in mg/dl \n")),int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n")),int(input("Resting ecg? (0,1,2) \n")),int(input("Maximum Heart Rate achieved? \n")),int(input("Exercise Induced Angina? 0 for no, 1 for yes \n")),float(input("Old Peak? ST Depression induced by exercise relative to rest \n")),int(input("Slope of the peak? (0,1,2) \n")),int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n")),int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n")) ]

