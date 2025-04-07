import numpy as np
import pandas as pd

np.random.seed(1234)

warm_up = 100
T = 30

time_factors1 = np.zeros(warm_up + T)
time_factors2 = np.zeros(warm_up + T)

for t in range(1, warm_up + T):
    time_factors1[t] = 0.4 * time_factors1[t-1] + np.random.normal(0, 1)
    time_factors2[t] = 0.6 * time_factors2[t-1] + np.random.normal(0, 1)

time_factors1 = time_factors1[warm_up:]
time_factors2 = time_factors2[warm_up:]

df = pd.DataFrame({'time': np.arange(T),
                   'time_factors1': time_factors1,
                   'time_factors2': time_factors2})

df.to_excel(f'simu2_time_factors.xlsx', index=False)