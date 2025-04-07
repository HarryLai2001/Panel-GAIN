library(openxlsx)
library(fixest)
library(stringr)
library(tidyverse)
library(glue)
library(data.table)


####################################### simu1 #######################################

T_pre = 20
reps_num = 50

reps_pred_err = vector(mode = "numeric", length = reps_num)
reps_gt_est = list()

for(rep in 0:(reps_num-1)) {
  panel = read.xlsx(glue("../data/simu_model1/T_pre={T_pre}/simu1 rep={rep}.xlsx"))
  panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$time - panel$first_treat, -9999)


  twfe = feols(fml = y ~ sunab(first_treat, time, no_agg = TRUE) | unit + time,
               data = panel)


  twfe_res = twfe$coeftable
  twfe_res$first_treat = as.numeric(sub(".*cohort::(\\d+).*", "\\1", rownames(twfe_res)))
  twfe_res$time_to_treat = as.numeric(sub(".*time::(-?\\d+).*", "\\1", rownames(twfe_res)))
  twfe_res = twfe_res %>%
    transmute(first_treat, time_to_treat, est_eff = Estimate)

  fe_unit = as.data.frame(fixef(twfe)$unit)
  fe_unit$unit = as.numeric(rownames(fe_unit))
  colnames(fe_unit)[1] = "fe_unit"
  panel = left_join(panel, fe_unit, by = c("unit"))

  fe_time = as.data.frame(fixef(twfe)$time)
  fe_time$time = as.numeric(rownames(fe_time))
  colnames(fe_time)[1] = "fe_time"
  panel = left_join(panel, fe_time, by = c("time"))
  panel$pred_y0 = panel$fe_unit + panel$fe_time

  reps_pred_err[rep + 1] = (sum(panel$d * (panel$pred_y0 - panel$y0) ^ 2) / sum(panel$d)) ^ 0.5


  gt_est = panel %>%
    group_by(first_treat, time) %>%
    summarise(time_to_treat = first(time_to_treat),
              y_mean = mean(y),
              y0_mean = mean(y0),
              pred_y0_mean = mean(pred_y0),
              eff = first(effect)) %>%
    left_join(twfe_res, by = c("first_treat", "time_to_treat"))
  gt_est[which(is.na(gt_est$est_eff)), "est_eff"] = 0
  gt_est$err = abs(gt_est$est_eff - gt_est$eff)

  reps_gt_est[[rep + 1]] = gt_est
}


pred_err = data.frame(rep = c(1:reps_num),
                      pred_err = reps_pred_err)

gt_est = rbindlist(reps_gt_est, idcol = "rep")


write.xlsx(list(pred_err = pred_err,
                gt_est = gt_est),
           glue("../result/simu_model1/TWFE/simu1 T_pre={T_pre} reps_res.xlsx"))







####################################### simu2 #####################################

T_pre = 20
reps_num = 50

reps_pred_err = vector(mode = "numeric", length = reps_num)
reps_gt_est = list()

for(rep in 0:(reps_num-1)) {
  panel = read.xlsx(glue("../data/simu_model2/T_pre={T_pre}/simu2 rep={rep}.xlsx"))
  panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$time - panel$first_treat, -9999)


  twfe = feols(fml = y ~ sunab(first_treat, time, no_agg = TRUE) | unit + time,
               data = panel)


  twfe_res = twfe$coeftable
  twfe_res$first_treat = as.numeric(sub(".*cohort::(\\d+).*", "\\1", rownames(twfe_res)))
  twfe_res$time_to_treat = as.numeric(sub(".*time::(-?\\d+).*", "\\1", rownames(twfe_res)))
  twfe_res = twfe_res %>%
    transmute(first_treat, time_to_treat, est_eff = Estimate)

  fe_unit = as.data.frame(fixef(twfe)$unit)
  fe_unit$unit = as.numeric(rownames(fe_unit))
  colnames(fe_unit)[1] = "fe_unit"
  panel = left_join(panel, fe_unit, by = c("unit"))

  fe_time = as.data.frame(fixef(twfe)$time)
  fe_time$time = as.numeric(rownames(fe_time))
  colnames(fe_time)[1] = "fe_time"
  panel = left_join(panel, fe_time, by = c("time"))
  panel$pred_y0 = panel$fe_unit + panel$fe_time

  reps_pred_err[rep + 1] = (sum(panel$d * (panel$pred_y0 - panel$y0) ^ 2) / sum(panel$d)) ^ 0.5

  gt_est = panel %>%
    group_by(first_treat, time) %>%
    summarise(time_to_treat = first(time_to_treat),
              y_mean = mean(y),
              y0_mean = mean(y0),
              pred_y0_mean = mean(pred_y0),
              eff = first(effect)) %>%
    left_join(twfe_res, by = c("first_treat", "time_to_treat"))
  gt_est[which(is.na(gt_est$est_eff)), "est_eff"] = 0
  gt_est$err = abs(gt_est$est_eff - gt_est$eff)

  reps_gt_est[[rep + 1]] = gt_est
}


pred_err = data.frame(rep = c(1:reps_num),
                      pred_err = reps_pred_err)

gt_est = rbindlist(reps_gt_est, idcol = "rep")


write.xlsx(list(pred_err = pred_err,
                gt_est = gt_est),
           glue("../result/simu_model2/TWFE/simu2 T_pre={T_pre} reps_res.xlsx"))




##################################### 环境信息披露 ###########################################
# options(scipen = 999)
# 
# panel = read.xlsx("../data/环境信息披露/eids_dta.xlsx")
# panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$year - panel$first_treat, -9999)
# 
# 
# twfe = feols(perfdi ~ sunab(first_treat, year, no_agg = TRUE) | code + year,
#              data = panel)
# 
# 
# twfe_res = twfe$coeftable
# twfe_res$first_treat = as.numeric(sub(".*cohort::(\\d+).*", "\\1", rownames(twfe_res)))
# twfe_res$time_to_treat = as.numeric(sub(".*year::(-?\\d+).*", "\\1", rownames(twfe_res)))
# twfe_res = twfe_res %>%
#   transmute(first_treat, time_to_treat, est_eff = Estimate)
# 
# fe_unit = as.data.frame(fixef(twfe)$code)
# fe_unit$code = rownames(fe_unit)
# colnames(fe_unit)[1] = "fe_unit"
# panel = left_join(panel, fe_unit, by = "code")
# 
# fe_time = as.data.frame(fixef(twfe)$year)
# fe_time$year = as.numeric(rownames(fe_time))
# colnames(fe_time)[1] = "fe_time"
# panel = left_join(panel, fe_time, by = "year")
# panel$pred_y0 = panel$fe_unit + panel$fe_time
# 
# 
# gt_est = panel %>%
#   group_by(first_treat, year) %>%
#   summarise(time_to_treat = first(time_to_treat),
#             y_mean = mean(perfdi),
#             pred_y0_mean = mean(pred_y0)) %>%
#   left_join(twfe_res, by = c("first_treat", "time_to_treat"))
# gt_est[which(is.na(gt_est$est_eff)), "est_eff"] = 0
# 
# 
# write.xlsx(list(pred = panel,
#                 gt_est = gt_est),
#            "../result/环境信息披露/res_TWFE.xlsx")






##################################### 低碳城市试点 ###########################################

panel = read.xlsx("../data/低碳城市试点/co2_dta.xlsx")
panel$time_to_treat = ifelse(panel$policy != 9999, panel$年度 - panel$policy, -9999)


twfe = feols(co2 ~ sunab(policy, 年度, no_agg = TRUE) | id + 年度,
             data = panel)


twfe_res = twfe$coeftable
twfe_res$policy = as.numeric(sub(".*cohort::(\\d+).*", "\\1", rownames(twfe_res)))
twfe_res$time_to_treat = as.numeric(sub(".*年度::(-?\\d+).*", "\\1", rownames(twfe_res)))
twfe_res = twfe_res %>%
  transmute(policy, time_to_treat, est_eff = Estimate)

fe_unit = as.data.frame(fixef(twfe)$id)
fe_unit$id = as.numeric(rownames(fe_unit))
colnames(fe_unit)[1] = "fe_unit"
panel = left_join(panel, fe_unit, by = c("id"))

fe_time = as.data.frame(fixef(twfe)$年度)
fe_time$年度 = as.numeric(rownames(fe_time))
colnames(fe_time)[1] = "fe_time"
panel = left_join(panel, fe_time, by = c("年度"))
panel$pred_y0 = panel$fe_unit + panel$fe_time


gt_est = panel %>%
  group_by(policy, 年度) %>%
  summarise(time_to_treat = first(time_to_treat),
            y_mean = mean(co2),
            pred_y0_mean = mean(pred_y0)) %>%
  left_join(twfe_res, by = c("policy", "time_to_treat"))
gt_est[which(is.na(gt_est$est_eff)), "est_eff"] = 0


write.xlsx(list(pred = panel,
                gt_est = gt_est),
           "../result/低碳城市试点/res_TWFE.xlsx")