library(openxlsx)
library(scpi)
library(stringr)
library(tidyverse)
library(glue)
library(data.table)


####################################### simu1 #######################################

T_pre = 20
reps_num = 50

reps_pred_err = vector(mode = "numeric", length = reps_num)
reps_gt_est = list()

start_time = Sys.time()

for(rep in 0:(reps_num-1)) {
  panel = read.xlsx(glue("../data/simu_model1/T_pre={T_pre}/simu1 rep={rep}.xlsx"))
  panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$time - panel$first_treat, -9999)


  sc = scdataMulti(df = panel,
                   id.var = "unit",
                   time.var = "time",
                   outcome.var = "y",
                   treatment.var = "d")

  sc_est = scest(sc)
  pred_y0 = as.data.frame(rbind(sc_est$est.results$Y.pre.fit, sc_est$est.results$Y.post.fit))
  pred_y0$unit = as.numeric(sub("(\\d+)\\..*", "\\1", rownames(pred_y0)))
  pred_y0$time = as.numeric(sub(".*\\.(\\d+)", "\\1", rownames(pred_y0)))

  panel = left_join(panel, pred_y0, by = c("unit", "time"))
  panel$pred_y0 = ifelse(is.na(panel$V1), panel$y, panel$V1)

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
           glue("../result/simu_model1/SC/simu1 T_pre={T_pre} reps_res.xlsx"))


end_time = Sys.time()
print(end_time - start_time)
  
  
  
  
  
  
####################################### simu2 #######################################

T_pre = 20
reps_num = 50

reps_pred_err = vector(mode = "numeric", length = reps_num)
reps_gt_est = list()

start_time = Sys.time()

for(rep in 0:(reps_num-1)) {
  panel = read.xlsx(glue("../data/simu_model2/T_pre={T_pre}/simu2 rep={rep}.xlsx"))
  panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$time - panel$first_treat, -9999)


  sc = scdataMulti(df = panel,
                   id.var = "unit",
                   time.var = "time",
                   outcome.var = "y",
                   treatment.var = "d")

  sc_est = scest(sc)
  pred_y0 = as.data.frame(rbind(sc_est$est.results$Y.pre.fit, sc_est$est.results$Y.post.fit))
  pred_y0$unit = as.numeric(sub("(\\d+)\\..*", "\\1", rownames(pred_y0)))
  pred_y0$time = as.numeric(sub(".*\\.(\\d+)", "\\1", rownames(pred_y0)))

  panel = left_join(panel, pred_y0, by = c("unit", "time"))
  panel$pred_y0 = ifelse(is.na(panel$V1), panel$y, panel$V1)

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
           glue("../result/simu_model2/SC/simu2 T_pre={T_pre} reps_res.xlsx"))


end_time = Sys.time()
print(end_time - start_time)






##################################### 环境信息披露 ###########################################

# panel = read.xlsx("../data/环境信息披露/eids_dta.xlsx")
# panel$time_to_treat = ifelse(panel$first_treat != 9999, panel$year - panel$first_treat, -9999)
# 
# 
# sc = scdataMulti(df = panel,
#                  id.var = "code",
#                  time.var = "year",
#                  outcome.var = "perfdi",
#                  treatment.var = "d")
# 
# sc_est = scest(sc)
# pred_y0 = as.data.frame(rbind(sc_est$est.results$Y.pre.fit, sc_est$est.results$Y.post.fit))
# pred_y0$code = sub("(\\d+)\\..*", "\\1", rownames(pred_y0))
# pred_y0$year = as.numeric(sub(".*\\.(\\d+)", "\\1", rownames(pred_y0)))
# 
# panel = left_join(panel, pred_y0, by = c("code", "year"))
# panel$pred_y0 = ifelse(is.na(panel$V1), panel$perfdi, panel$V1)
# 
# 
# gt_est = panel %>%
#   group_by(first_treat, year) %>%
#   summarise(y_mean = mean(perfdi),
#             pred_y0_mean = mean(pred_y0))
# gt_est$est_eff = gt_est$y_mean - gt_est$pred_y0_mean
# 
# write.xlsx(list(pred = panel,
#                 gt_est = gt_est),
#            "../result/环境信息披露/res_SC.xlsx")




##################################### 低碳城市试点 ###########################################

panel = read.xlsx("../data/低碳城市试点/co2_dta.xlsx")
panel$time_to_treat = ifelse(panel$policy != 9999, panel$年度 - panel$policy, -9999)

sc = scdataMulti(df = panel,
                 id.var = "id",
                 time.var = "年度",
                 outcome.var = "co2",
                 treatment.var = "lccpost")

sc_est = scest(sc)
pred_y0 = as.data.frame(rbind(sc_est$est.results$Y.pre.fit, sc_est$est.results$Y.post.fit))
pred_y0$id = as.numeric(sub("(\\d+)\\..*", "\\1", rownames(pred_y0)))
pred_y0$年度 = as.numeric(sub(".*\\.(\\d+)", "\\1", rownames(pred_y0)))

panel = left_join(panel, pred_y0, by = c("id", "年度"))

panel$pred_y0 = ifelse(is.na(panel$V1), panel$co2, panel$V1)


gt_est = panel %>%
  group_by(policy, 年度) %>%
  summarise(y_mean = mean(co2),
            pred_y0_mean = mean(pred_y0))
gt_est$est_eff = gt_est$y_mean - gt_est$pred_y0_mean

write.xlsx(list(pred = panel,
                gt_est = gt_est),
           "../result/低碳城市试点/res_SC.xlsx")