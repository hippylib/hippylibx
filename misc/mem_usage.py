##Don't push this to github

import matplotlib.pyplot as plt

import time
import os
import psutil

import poisson_example_copy

def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
 
 
def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_full_info()
    return mem_info.uss

 
 
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        return (mem_after - mem_before)/1e6
    return wrapper

nx = 64
ny = 64
noise_variance = 1e-4
prior_param = {"gamma": 0.1, "delta": 1.}

mem_usage = []


value = profile(poisson_example_copy.run_inversion)

num_calls = 100

for _ in range(num_calls):
    mem_usage.append(value(nx, ny, noise_variance, prior_param))

plt.plot(mem_usage)


# # print(mem_usage)
plt.xlabel('Iteration')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Iterations for poisson example')
plt.savefig("memory_usage_plot_poisson.png")
plt.show()

    


# print(value.wrapper)