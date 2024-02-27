import sfsi_toy_gaussian
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
num_calls = 1
mem_usage = []
nx = 64
ny = 64
noise_variance = 1e-6
prior_param = {"gamma": 0.05, "delta": 1.}

for _ in range(num_calls):
    # sfsi_toy_gaussian.run_inversion(nx, ny, noise_variance, prior_param)
    # sfsi_toy_gaussian.run_inversion(nx, ny, noise_variance, prior_param)
    
    # mem_usage.append(sfsi_toy_gaussian.run_inversion.memory_usage[0])
    # mem_usage.append(memory_usage(sfsi_toy_gaussian.run_inversion,(nx, ny, noise_variance, prior_param),{}) ) 

    mem_usage.append(memory_usage((sfsi_toy_gaussian.run_inversion, (nx, ny, noise_variance, prior_param), {}))[0])
    # print(  memory_usage((sfsi_toy_gaussian.run_inversion, (nx, ny, noise_variance, prior_param), {}))  )


plt.plot(mem_usage)


print(mem_usage)
plt.xlabel('Iteration')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Iterations')
plt.savefig("memory_usage_plot_2.png")
plt.show()


