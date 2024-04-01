#to find the optimum value of the ratio of prior parameters gamma and delta
# to best satisfy the Morozov discrepancy principle.

#input: inputs to the run_inversion function, threshold to meet (0.5 * volume of domain)
#output: gamma and delta that will best satisfy the principle for a certain value of 
# ratio
import sys
import os

sys.path.append(os.path.abspath("../"))

import hippylibX as hpx
import numpy as np

sys.path.append(os.path.abspath("../example"))

from example import poisson_dirichlet_example #choose script as needed

nx, ny = 64, 64
noise_variance = 1e-4
threshold = 0.5 #0.5 * volume of domain
base_gamma = 0.1
base_delta = 1

lamba_candidates = [0]*4
lamba_candidates[0] = 1e-2
for i in range(1,len(lamba_candidates)):
    lamba_candidates[i] = 10 * lamba_candidates[i-1]

difference = []

for i in range(len(lamba_candidates)):
    gamma, delta = lamba_candidates[i] * base_gamma, lamba_candidates[i] *  base_delta 
    prior_param = {"gamma": gamma, "delta": delta}
    difference.append(np.abs(threshold - poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, prior_param)) )
    print(i,":",difference)
    if(i > 0 and difference[i] > difference[i-1]):
        chosen_index = i - 1
        break

# print(chosen_index) #should be 1 or 2
# lambda_c = lamba_candidates[1]
# print(poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, {"gamma":base_gamma*lambda_c, "delta": base_delta*lambda_c}))

# lambda_c = lamba_candidates[2]
# print(poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, {"gamma":base_gamma*lambda_c, "delta": base_delta*lambda_c}))


base_lambda = lamba_candidates[chosen_index]
gamma, delta = base_lambda * base_gamma, base_lambda * base_delta
prior_param = {"gamma": gamma, "delta":delta} 

results = [] #to store abs(threshold - function_return) while looping through successive values of lambda

if( poisson_dirichlet_example.run_inversion(nx,ny,noise_variance, prior_param) < threshold ):
    #loop through lambda *= 2
    ind = 0
    while True:
        gamma, delta = base_lambda * base_gamma, base_lambda * base_delta
        prior_param = {"gamma": gamma, "delta":delta} 
        results.append (np.abs(threshold - poisson_dirichlet_example.run_inversion(nx,ny,noise_variance, prior_param) ) )
        print(ind,":",results,":", poisson_dirichlet_example.run_inversion(nx,ny,noise_variance, prior_param))
        if(ind > 0 and results[ind] > results[ind-1]):
            break
        base_lambda *= 2
        ind += 1
else:
    #loop through lambda /= 2
    ind = 0
    while True:
        gamma, delta = base_lambda * base_gamma, base_lambda * base_delta
        prior_param = {"gamma": gamma, "delta":delta} 
        results.append (np.abs(threshold - poisson_dirichlet_example.run_inversion(nx,ny,noise_variance, prior_param) ) )
        print(ind,":",results,":", poisson_dirichlet_example.run_inversion(nx,ny,noise_variance, prior_param))
        if(ind > 0 and results[ind] > results[ind-1]):
            break
        base_lambda /= 2
        ind += 1

print(base_lambda,base_gamma*base_lambda, base_delta*base_lambda) #chosen values
