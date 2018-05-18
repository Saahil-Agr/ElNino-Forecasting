import os
import numpy as np



data_dir = "img"
filenames = os.listdir(data_dir)
filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.npy')]

max = -float("inf")
min = float("inf")
for f in filenames:
    test_file = np.load(f)
    local_max = np.max(test_file)
    local_min = np.min(test_file)
    if local_max > max:
        max = local_max

    if local_min < min:
        min = local_min

print(max, min)

out_range = (0,255)
for f in filenames:
    file = np.load(f)
    file_scaled = out_range[0] + (file - min) * (out_range[1] - out_range[0]) / (max - min)
    f_name = f.split('.')[0].split('/')[1] + '_scaled'
    f_path = os.path.join("data",f_name)
    np.save(f_path,file_scaled)
