import numpy as np
import pandas as pd

T_pre = 20
T_post = 10
first_treat = [20, 22, 24, 26, 28, 9999]
T = 30
T_obs = T_pre + T_post
N = 100
seed = 1234
reps_num = 50
groups = range(len(first_treat))


# time fixed-effects
time_fixed_effects = 0.1 * np.arange(T)

time_factors = pd.read_excel("simu1_time_factors.xlsx", engine='openpyxl')
time_factors1 = time_factors["time_factors1"].to_numpy()
time_factors2 = time_factors["time_factors2"].to_numpy()


def generate_data(g):
    unit_fixed_effects = np.random.uniform(5.0+1.0*g, 7.0+1.0*g, size=N)
    unit_factors1 = np.random.uniform(1.0 + 0.2*g, 1.2 + 0.2*g, size=N)
    unit_factors2 = np.random.uniform(1.0 + 0.2*g, 1.2 + 0.2*g, size=N)
    y0 = unit_fixed_effects[:,None] + time_fixed_effects[None,:] + unit_factors1[:,None] @ time_factors1[None,:] + unit_factors2[:,None] @ time_factors2[None,:] + np.random.normal(0, 1, (N,T))

    data = np.concatenate([
        np.full((N, T, 1), g),
        np.full((N, T, 1), first_treat[g]),
        np.arange(T)[None,:,None].repeat(N,axis=0),
        y0[:,:,None]
    ], axis=-1)

    return data



np.random.seed(seed)

for rep in range(reps_num):
    data = np.array([generate_data(g) for g in groups])
    data = np.reshape(data[:,:,T-T_obs:,:], (len(groups) * N, T_obs, -1))
    shuffle_idx = np.random.permutation(len(groups) * N)
    data = data[shuffle_idx,:,:]
    idx = np.arange(len(groups) * N)
    data = np.concatenate([idx[:,None,None].repeat(T_obs, axis=1), data], axis=-1)
    data = np.reshape(data, (len(groups) * N * T_obs, -1))
    df = pd.DataFrame(data, columns=['unit', 'group', 'first_treat', 'time', 'y0'])
    df['d'] = (df['time'] - df['first_treat']).apply(lambda x: 1 if x>=0 else 0)
    df['effect'] = df['d'] * (0.05 * df['first_treat'] + 0.01 * df['time'])
    df['y'] = df['y0'] + df['effect']
    df.to_excel(f'T_pre={T_pre}/simu1 rep={rep}.xlsx', index=False)

