import matplotlib.pyplot as plt
import re


with open("0gpu.gradvar.txt") as f:
    lines = [ln.strip() for ln in f.readlines()]

    steps_array = []
    var_array = []
    std_array = []
    for ln in lines:
        if ".ckpt" in ln:
            steps = re.match(r"^.*from .*step_(\d+).progress.*$", ln).group(1)
            steps_array.append(int(steps))
        if "mean_of_var" in ln:
            var = re.match(r"^.* mean_of_var=([.\d]*)$", ln).group(1)
            var_array.append(float(var))
        if "mean_of_std" in ln:
            std = re.match(r"^.* mean_of_std=([.\d]*)$", ln).group(1)
            std_array.append(float(std))

    plt.plot(steps_array, std_array)
    plt.show()
