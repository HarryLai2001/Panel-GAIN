library(tidyverse)
library(openxlsx)
library(glue)
library(data.table)
library(ggplot2)
library(cowplot)
library(ggsci)
library(latex2exp)


T_pre = 20
methods = c("TWFE", "SC", "MCPanel", "GAIN", "Panel-GAIN1", "Panel-GAIN2", "Panel-GAIN")
cmp_res_ls = list()
pred_err_ls = list()
gt_est_ls = list()


for(method in methods) {
  pred = read.xlsx(glue("{method}/simu2 T_pre={T_pre} reps_res.xlsx"),
                   sheet = "pred_err")
  pred_err_ls[[method]] = pred
  pred_err_mean = mean(pred$pred_err)
  pred_err_std = sd(pred$pred_err)
  
  
  gt_est = read.xlsx(glue("{method}/simu2 T_pre={T_pre} reps_res.xlsx"),
                     sheet = "gt_est") %>%
    group_by(first_treat, time) %>%
    summarise(mean = mean(err),
              std = sd(err))
  gt_est_ls[[method]] = gt_est
  
  
  cmp_res = data.frame(pred_err_mean, pred_err_std)
  cmp_res_ls[[method]] = cmp_res
}



### summary table
cmp_res = rbindlist(cmp_res_ls, idcol = "method")
write.xlsx(cmp_res, glue("simu2 cmp_res T_pre={T_pre}.xlsx"))


### group-time treatment effect estimation error plot
gt_est = rbindlist(gt_est_ls, idcol = "method")
gt_est_ls = split(gt_est, gt_est$first_treat)
plots = list()
for(g in names(gt_est_ls)) {
  df = gt_est_ls[[g]]
  df$method = factor(df$method,
                     levels = c("TWFE", "SC", "MCPanel", "GAIN", "Panel-GAIN1", "Panel-GAIN2", "Panel-GAIN"))
  
  p = ggplot(df, aes(x = time, y = mean, group = method, color = method)) + 
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std), size = 0.2, width = 0.2) + 
    geom_line(size = 0.5) +
    geom_point(size = 1.0, shape = 15) + 
    geom_vline(xintercept = as.numeric(g) - 0.5, size = 0.5, linetype = 'dashed', color = 'gray50') + 
    scale_y_continuous(name = TeX("\\textbf{bias}$(ATT(g,t))$", italic = TRUE), limits = c(0, 10), breaks = seq(0, 10, 1)) + 
    scale_x_continuous(name = "时期", limits = c(20-T_pre, 29), breaks = seq(0, 29, 1)) + 
    labs(title = ifelse(g != '9999', 
                        TeX(glue("$G_i = {g}$"), italic = TRUE), 
                        TeX(glue("$G_i = \\infty$"), italic = TRUE))) + 
    scale_color_manual(name="方法",
                       values = c("#BD6263","#8EA325","#A9D179","#84CAC0","#F5AE6B","#BCB8D3","#4387B5")) + 
    theme_bw() + 
    theme(panel.grid=element_blank(),
          plot.title = element_text(hjust = 0.5),
          legend.title = element_text(size=8),
          legend.key.size = unit(5, 'mm'),
          legend.text = element_text(size=8),
          legend.position = c(0.18, 0.8),
          legend.spacing.y = unit(3, 'mm'),
          legend.background=element_rect(fill=rgb(1,1,1,alpha=0.5),colour='gray50')) +
    guides(color = guide_legend(ncol = 2, byrow = TRUE))
  
  plots[[g]] = p
}


plot_grid(plotlist = plots, ncol = 2)


ggsave(glue("simu2 gt_eff_err T_pre={T_pre}.png"), width = 14, height = 12)