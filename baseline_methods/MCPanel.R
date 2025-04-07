library(openxlsx)
library(MCPanel)
library(tidyverse)
library(glue)
library(data.table)


####################################### simu1 ######################################

T_pre = 20
reps_num = 50

reps_pred_err = vector(mode = "numeric", length = reps_num)
reps_gt_est = list()


for(rep in 0:(reps_num-1)) {
  panel = read.xlsx(glue("../data/simu_model1/T_pre={T_pre}/simu1 rep={rep}.xlsx"))
  panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$time - panel$first_treat, -9999)

  y = pivot_wider(data = panel,
                  id_cols = "unit",
                  names_from = "time",
                  values_from = "y")

  d = pivot_wider(data = panel,
                  id_cols = "unit",
                  names_from = "time",
                  values_from = "d")

  mask = 1 - as.matrix(d[,-1])
  y_obs = as.matrix(y[,-1]) * mask

  mcpanel = mcnnm_cv(y_obs, mask)

  T_obs = ncol(y_obs)
  N = nrow(y_obs)
  pred = replicate(T_obs, mcpanel$u) + t(replicate(N, mcpanel$v)) + mcpanel$L

  pred = as.data.frame(pred)
  colnames(pred) = colnames(y_obs)
  pred$unit = y$unit

  pred = pivot_longer(data = pred,
                      cols = !unit,
                      names_to = "time",
                      values_to = "pred_y0")
  pred$time = as.numeric(pred$time)

  panel = left_join(panel, pred, by = c("unit", "time"))


  reps_pred_err[rep + 1] = (sum(panel$d * (panel$pred_y0 - panel$y0) ^ 2) / sum(panel$d)) ^ 0.5

  gt_est = panel %>%
    group_by(first_treat, time) %>%
    summarise(y_mean = mean(y),
              y0_mean = mean(y0),
              pred_y0_mean = mean(pred_y0),
              eff = first(effect))
  gt_est$est_eff = gt_est$y_mean - gt_est$pred_y0_mean
  gt_est$err = abs(gt_est$est_eff - gt_est$eff)

  reps_gt_est[[rep + 1]] = gt_est
}


pred_err = data.frame(rep = c(1:reps_num),
                      pred_err = reps_pred_err)

gt_est = rbindlist(reps_gt_est, idcol = "rep")


write.xlsx(list(pred_err = pred_err,
                gt_est = gt_est),
           glue("../result/simu_model1/MCPanel/simu1 T_pre={T_pre} reps_res.xlsx"))







####################################### simu2 #####################################

T_pre = 20
reps_num = 50

reps_pred_err = vector(mode = "numeric", length = reps_num)
reps_gt_est = list()


for(rep in 0:(reps_num-1)) {
  panel = read.xlsx(glue("../data/simu_model2/T_pre={T_pre}/simu2 rep={rep}.xlsx"))
  panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$time - panel$first_treat, -9999)

  y = pivot_wider(data = panel,
                  id_cols = "unit",
                  names_from = "time",
                  values_from = "y")

  d = pivot_wider(data = panel,
                  id_cols = "unit",
                  names_from = "time",
                  values_from = "d")

  mask = 1 - as.matrix(d[,-1])
  y_obs = as.matrix(y[,-1]) * mask

  mcpanel = mcnnm_cv(y_obs, mask)

  T_obs = ncol(y_obs)
  N = nrow(y_obs)
  pred = replicate(T_obs, mcpanel$u) + t(replicate(N, mcpanel$v)) + mcpanel$L

  pred = as.data.frame(pred)
  colnames(pred) = colnames(y_obs)
  pred$unit = y$unit

  pred = pivot_longer(data = pred,
                      cols = !unit,
                      names_to = "time",
                      values_to = "pred_y0")
  pred$time = as.numeric(pred$time)

  panel = left_join(panel, pred, by = c("unit", "time"))


  reps_pred_err[rep + 1] = (sum(panel$d * (panel$pred_y0 - panel$y0) ^ 2) / sum(panel$d)) ^ 0.5

  gt_est = panel %>%
    group_by(first_treat, time) %>%
    summarise(y_mean = mean(y),
              y0_mean = mean(y0),
              pred_y0_mean = mean(pred_y0),
              eff = first(effect))
  gt_est$est_eff = gt_est$y_mean - gt_est$pred_y0_mean
  gt_est$err = abs(gt_est$est_eff - gt_est$eff)

  reps_gt_est[[rep + 1]] = gt_est
}


pred_err = data.frame(rep = c(1:reps_num),
                      pred_err = reps_pred_err)

gt_est = rbindlist(reps_gt_est, idcol = "rep")


write.xlsx(list(pred_err = pred_err,
                gt_est = gt_est),
           glue("../result/simu_model2/MCPanel/simu2 T_pre={T_pre} reps_res.xlsx"))





##################################### 环境信息披露 ###########################################
# options(scipen = 999)
# 
# panel = read.xlsx("../data/环境信息披露/eids_dta.xlsx")
# panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$year - panel$first_treat, -9999)
# 
# y = pivot_wider(data = panel,
#                 id_cols = "code",
#                 names_from = "year",
#                 values_from = "perfdi")
# 
# d = pivot_wider(data = panel,
#                 id_cols = "code",
#                 names_from = "year",
#                 values_from = "d")
# 
# mask = 1 - as.matrix(d[,-1])
# y_obs = as.matrix(y[,-1]) * mask
# 
# mcpanel = mcnnm_cv(y_obs, mask)
# 
# T_obs = ncol(y_obs)
# N = nrow(y_obs)
# pred = replicate(T_obs, mcpanel$u) + t(replicate(N, mcpanel$v)) + mcpanel$L
# 
# pred = as.data.frame(pred)
# colnames(pred) = colnames(y_obs)
# pred$code = y$code
# 
# pred = pivot_longer(data = pred,
#                     cols = !code,
#                     names_to = "year",
#                     values_to = "pred_y0")
# pred$year = as.numeric(pred$year)
# 
# panel = left_join(panel, pred, by = c("code", "year"))
# 
# 
# gt_est = panel %>%
#   group_by(first_treat, year) %>%
#   summarise(y_mean = mean(perfdi),
#             pred_y0_mean = mean(pred_y0))
# gt_est$est_eff = gt_est$y_mean - gt_est$pred_y0_mean
# 
# 
# write.xlsx(list(pred = panel,
#                 gt_est = gt_est),
#            "../result/环境信息披露/res_MCPanel.xlsx")






##################################### 低碳城市试点 ###########################################

panel = read.xlsx("../data/低碳城市试点/co2_dta.xlsx")
panel$time_to_treat = ifelse(panel$policy != 9999, panel$年度 - panel$policy, -9999)

y = pivot_wider(data = panel,
                id_cols = "id",
                names_from = "年度",
                values_from = "co2")

d = pivot_wider(data = panel,
                id_cols = "id",
                names_from = "年度",
                values_from = "lccpost")

mask = 1 - as.matrix(d[,-1])
y_obs = as.matrix(y[,-1]) * mask

mcpanel = mcnnm_cv(y_obs, mask)

T_obs = ncol(y_obs)
N = nrow(y_obs)
pred = replicate(T_obs, mcpanel$u) + t(replicate(N, mcpanel$v)) + mcpanel$L

pred = as.data.frame(pred)
colnames(pred) = colnames(y_obs)
pred$id = y$id

pred = pivot_longer(data = pred,
                    cols = !id,
                    names_to = "年度",
                    values_to = "pred_y0")
pred$年度 = as.numeric(pred$年度)

panel = left_join(panel, pred, by = c("id", "年度"))


gt_est = panel %>%
  group_by(policy, 年度) %>%
  summarise(y_mean = mean(co2),
            pred_y0_mean = mean(pred_y0))
gt_est$est_eff = gt_est$y_mean - gt_est$pred_y0_mean


write.xlsx(list(pred = panel,
                gt_est = gt_est),
           "../result/低碳城市试点/res_MCPanel.xlsx")