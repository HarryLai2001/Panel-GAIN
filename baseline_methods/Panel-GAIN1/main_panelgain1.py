import sys
import time
import torch
from toolz import first

from data_loader import load_data
from panel_gain1 import Generator, Discriminator
from trainer_panelgain1 import Trainer

import pandas as pd
import numpy as np
import random



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # %% simu_model1
    reps_num = 50
    T_pre = 20
    T_post = 10
    T_obs = T_pre + T_post

    reps_pred_err = list()
    reps_gt_est = list()

    start = time.time()
    for rep in range(reps_num):
        print('rep:', rep)

        set_seed(1234)

        df, data_loader, norm_parameters = load_data(f"../../data/simu_model1/T_pre={T_pre}/simu1 rep={rep}.xlsx", batch_size=128)

        generator = Generator(t_dim=T_obs, h_dims=[16, 8, 8, 4], num_groups=6, embed_dim=3)
        discriminator = Discriminator(t_dim=T_obs, h_dims=[16, 8, 8, 4])

        optim_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

        trainer = Trainer(data_loader=data_loader,
                          norm_parameters=norm_parameters,
                          generator=generator,
                          discriminator=discriminator,
                          optim_g=optim_g,
                          optim_d=optim_d,
                          hint_rate=0.9,
                          coef=1000.0,
                          n_epoches=5000)

        trainer.train()
        df_pred = trainer.predict()
        df['pred_y0'] = df_pred['pred_y0']
        df['y_renorm'] = df_pred['y_renorm']
        df['time_to_treat'] = (df['time'] - df['first_treat']).apply(lambda x: x if x > -100 else -9999)

        pred_err = np.sqrt((df['d'] * (df['pred_y0'].to_numpy() - df['y0'].to_numpy()) ** 2).sum() / df['d'].sum())
        reps_pred_err.append(pred_err)

        gt_est = df.groupby(['first_treat', 'time']).agg(y_mean=('y', np.mean),
                                                         y0_mean=('y0', np.mean),
                                                         pred_y0_mean=('pred_y0', np.mean),
                                                         eff=('effect', first))
        gt_est['est_eff'] = gt_est['y_mean'] - gt_est['pred_y0_mean']
        gt_est['err'] = abs(gt_est['est_eff'] - gt_est['eff'])
        reps_gt_est.append(gt_est)

        print(pred_err)

    pred_err = pd.DataFrame({
        "rep": [i + 1 for i in range(reps_num)],
        "pred_err": reps_pred_err,
    })

    gt_est_df = pd.DataFrame(pd.concat(reps_gt_est, keys=[i + 1 for i in range(reps_num)]))
    gt_est_df.reset_index(inplace=True)
    gt_est_df.columns = ['rep', 'first_treat', 'time', 'y_mean', 'y0_mean', 'pred_y0_mean', 'eff', 'est_eff', 'err']

    with pd.ExcelWriter(f"../../result/simu_model1/Panel-GAIN1/simu1 T_pre={T_pre} reps_res.xlsx", engine='openpyxl') as writer:
        pred_err.to_excel(writer, sheet_name="pred_err", index=False)
        gt_est_df.to_excel(writer, sheet_name="gt_est", index=False)

    end = time.time()
    print(f"执行时间：{end - start}s")






    # %% simu_model2
    reps_num = 50
    T_pre = 20
    T_post = 10
    T_obs = T_pre + T_post

    reps_pred_err = list()
    reps_gt_est = list()

    start = time.time()
    for rep in range(reps_num):
        print('rep:', rep)

        set_seed(1234)

        df, data_loader, norm_parameters = load_data(f"../../data/simu_model2/T_pre={T_pre}/simu2 rep={rep}.xlsx", batch_size=128)

        generator = Generator(t_dim=T_obs, h_dims=[16, 8, 8, 4], num_groups=6, embed_dim=3)
        discriminator = Discriminator(t_dim=T_obs, h_dims=[16, 8, 8, 4])

        optim_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

        trainer = Trainer(data_loader=data_loader,
                          norm_parameters=norm_parameters,
                          generator=generator,
                          discriminator=discriminator,
                          optim_g=optim_g,
                          optim_d=optim_d,
                          hint_rate=0.9,
                          coef=1000.0,
                          n_epoches=5000)

        trainer.train()
        df_pred = trainer.predict()
        df['pred_y0'] = df_pred['pred_y0']
        df['y_renorm'] = df_pred['y_renorm']
        df['time_to_treat'] = (df['time'] - df['first_treat']).apply(lambda x: x if x > -100 else -9999)

        pred_err = np.sqrt((df['d'] * (df['pred_y0'].to_numpy() - df['y0'].to_numpy()) ** 2).sum() / df['d'].sum())
        reps_pred_err.append(pred_err)

        gt_est = df.groupby(['first_treat', 'time']).agg(y_mean=('y', np.mean),
                                                         y0_mean=('y0', np.mean),
                                                         pred_y0_mean=('pred_y0', np.mean),
                                                         eff=('effect', first))
        gt_est['est_eff'] = gt_est['y_mean'] - gt_est['pred_y0_mean']
        gt_est['err'] = abs(gt_est['est_eff'] - gt_est['eff'])
        reps_gt_est.append(gt_est)

        print(pred_err)

    pred_err = pd.DataFrame({
        "rep": [i + 1 for i in range(reps_num)],
        "pred_err": reps_pred_err,
    })

    gt_est_df = pd.DataFrame(pd.concat(reps_gt_est, keys=[i + 1 for i in range(reps_num)]))
    gt_est_df.reset_index(inplace=True)
    gt_est_df.columns = ['rep', 'first_treat', 'time', 'y_mean', 'y0_mean', 'pred_y0_mean', 'eff', 'est_eff', 'err']

    with pd.ExcelWriter(f"../../result/simu_model2/Panel-GAIN1/simu2 T_pre={T_pre} reps_res.xlsx", engine='openpyxl') as writer:
        pred_err.to_excel(writer, sheet_name="pred_err", index=False)
        gt_est_df.to_excel(writer, sheet_name="gt_est", index=False)

    end = time.time()
    print(f"执行时间：{end - start}s")