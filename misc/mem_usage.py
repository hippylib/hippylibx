# import sfsi_toy_gaussian
import matplotlib.pyplot as plt
# from memory_profiler import memory_usage
# num_calls = 100
# mem_usage = []
# nx = 64
# ny = 64
# noise_variance = 1e-6
# prior_param = {"gamma": 0.05, "delta": 1.}

# for _ in range(num_calls):
#     # sfsi_toy_gaussian.run_inversion(nx, ny, noise_variance, prior_param)
#     # sfsi_toy_gaussian.run_inversion(nx, ny, noise_variance, prior_param)
    
#     # mem_usage.append(sfsi_toy_gaussian.run_inversion.memory_usage[0])
#     # mem_usage.append(memory_usage(sfsi_toy_gaussian.run_inversion,(nx, ny, noise_variance, prior_param),{}) ) 

#     usage = memory_usage((sfsi_toy_gaussian.run_inversion, (nx, ny, noise_variance, prior_param), {}))
#     mem_usage.append(usage[-1] - usage[0])
#     # mem_usage.append(memory_usage((sfsi_toy_gaussian.run_inversion, (nx, ny, noise_variance, prior_param), {}))[0])
#     # print(  memory_usage((sfsi_toy_gaussian.run_inversion, (nx, ny, noise_variance, prior_param), {}))  )



# plt.plot(mem_usage)


# # print(mem_usage)
# plt.xlabel('Iteration')
# plt.ylabel('Memory Usage (MB)')
# plt.title('Memory Usage Over Iterations')
# plt.savefig("memory_usage_plot_3.png")
# plt.show()

#########################################################################
# import tracemalloc
# import sfsi_toy_gaussian
# nx = 64
# ny = 64
# noise_variance = 1e-6
# prior_param = {"gamma": 0.05, "delta": 1.}


# tracemalloc.start()

# snapshot_1 = tracemalloc.take_snapshot()
# sfsi_toy_gaussian.run_inversion(nx, ny, noise_variance, prior_param)
# snapshot_2 = tracemalloc.take_snapshot()

# top_stats = snapshot_2.compare_to(snapshot_1, 'lineno')
# print("[ Top 10 differences ]")
# for stat in top_stats[:10]:
#     print(stat)


import time
import os
import psutil
import sfsi_toy_gaussian
  
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
        # print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
        #     func.__name__,
        #     mem_before, mem_after, mem_after - mem_before,
        #     elapsed_time))
        return (mem_after - mem_before)/1e6
    return wrapper

nx = 64
ny = 64
noise_variance = 1e-6
prior_param = {"gamma": 0.05, "delta": 1.}

mem_usage = []


value = profile(sfsi_toy_gaussian.run_inversion)

num_calls = 100

for _ in range(num_calls):
    mem_usage.append(value(nx, ny, noise_variance, prior_param))

plt.plot(mem_usage)


# # print(mem_usage)
plt.xlabel('Iteration')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Iterations')
plt.savefig("memory_usage_plot_3.png")
plt.show()

    


# print(value.wrapper)