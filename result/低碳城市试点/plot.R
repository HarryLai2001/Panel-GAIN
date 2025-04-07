library(openxlsx)
library(tidyverse)
library(ggplot2)
library(glue)
library(cowplot)
library(latex2exp)

df = read.xlsx("res_all.xlsx") %>%
  pivot_longer(cols = -c("policy", "年度", "time_to_treat"))

groups = split(df, df$policy)

plots = list()
for(g in names(groups)) {
  df = groups[[g]]
  df$name = factor(df$name,
                   levels = c("TWFE", "SC", "MCPanel", "PanelGAIN"))
  
  p = ggplot(df, aes(x = 年度, y = value, group = name, color = name)) +
    geom_line(size = 0.5) +
    geom_point(size = 1.0, shape = 15) +
    geom_vline(xintercept = as.numeric(g) - 0.5, size = 0.5, linetype = 'dashed', color = 'gray50') + 
    geom_hline(yintercept = 0, size = 0.5, linetype = 'dashed', color = 'gray50') + 
    scale_y_continuous(name = "组别-时间平均干预效应", limits = c(-0.5, 0.5), breaks = seq(-0.5, 0.5, 0.1)) + 
    scale_x_continuous(name = "年度", limits = c(2007, 2019), breaks = seq(2007, 2019, 1)) + 
    labs(title = ifelse(g != '9999', 
                        TeX(glue("$G_i = {g}$"), italic = TRUE), 
                        TeX(glue("$G_i = \\infty$"), italic = TRUE))) + 
    scale_color_manual(values = c("#BD6263","#84CAC0","#F5AE6B","#BCB8D3","#4387B5"),
                       name = "方法",
                       labels = c(TeX("\\textrm{TWFE} $\\hat{E}[y_0]$", italic = TRUE), 
                                  TeX("\\textrm{SC} $\\hat{E}[y_0]$", italic = TRUE), 
                                  TeX("\\textrm{MCPanel} $\\hat{E}[y_0]$", italic = TRUE),
                                  TeX("\\textrm{Panel-GAIN} $\\hat{E}[y_0]$", italic = TRUE))) + 
    theme_bw() + 
    theme(panel.grid=element_blank(),
          plot.title = element_text(hjust = 0.5),
          legend.title = element_text(size=8),
          legend.key.size = unit(5, 'mm'),
          legend.text = element_text(size=8),
          legend.position = c(0.12, 0.8),
          legend.spacing.y = unit(3, 'mm'),
          legend.background=element_rect(fill=rgb(1,1,1,alpha=0.5),colour='gray50'))
  
  plots[[g]] = p
}

plot_grid(plotlist = plots, ncol = 2)


ggsave(glue("plot.png"), width = 14, height = 8)