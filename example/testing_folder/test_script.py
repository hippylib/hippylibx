import numpy as np
import pickle

with open('outputs.pickle', 'rb') as f:
    data = pickle.load(f)

xvals = data["xvals"]
arr_1 = data['arr_1']
arr_2 = data['arr_2']
value = data['sym_Hessian_value']

if(value > 1e-10):
    print("1")
    exit

#values from arr_1
relevant_values_1, relevant_values_2 = arr_1[10:], arr_2[10:]

x =  xvals[10:]

data = relevant_values_1
y = data
coefficients = np.polyfit(np.log(x), np.log(y), 1)
slope_1 = coefficients[0]

data = relevant_values_2
y = data
coefficients = np.polyfit(np.log(x), np.log(y), 1)
slope_2 = coefficients[0]


if(np.abs(slope_1 - 1) > 1e-1 or np.abs(slope_2 - 1) > 1e-1):
    print("1")
    exit

print("0")
