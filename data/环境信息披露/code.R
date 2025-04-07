library(haven)
library(tidyverse)
library(ggplot2)
library(openxlsx)
library(ggsci)
library(fixest)


dta = read_dta("数据1.dta") %>%
  transmute(code, 
            year, 
            first_treat = ifelse(treat == 1, 2008, 9999),
            group = treat,
            d = treat * time,
            perfdi) %>%
  na.omit()

select_id = dta %>%
  select(code, year, perfdi) %>%
  pivot_wider(id_cols = code,
              names_from = year,
              values_from = perfdi) %>%
  na.omit()

dta = dta[which(dta$code %in% select_id$code), ]

count = dta %>%
  group_by(code) %>%
  summarise(first_treat = first(first_treat)) %>%
  group_by(first_treat) %>%
  summarise(n = n())

summ = dta %>%
  group_by(first_treat, year) %>%
  summarise(y_mean = mean(perfdi))
summ$first_treat = factor(summ$first_treat,
                          levels = c(2008, 9999),
                          labels = c("2008", "never"))

ggplot(summ, aes(x = year, y = y_mean, group = first_treat, colour = first_treat)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept=2008-0.5, color = 'gray', linetype = 'dashed', size = 0.3) + 
  guides(color=guide_legend(title = "组别")) + 
  scale_y_continuous(name = "perfdi", limits = c(2, 8), breaks = seq(2, 8, 1)) + 
  scale_x_continuous(name = "年份", limits = c(2004, 2013), breaks = seq(2004, 2013, 1)) + 
  scale_color_aaas() + 
  theme_bw() + 
  theme(panel.grid=element_blank())
ggsave('fdi_avg.png', width = 8, height = 4)


dym_test = twfe = feols(perfdi ~ sunab(first_treat, year, no_agg = FALSE) | code + year,
                        data = dta)
dym_test_res = as.data.frame(dym_test$coeftable)
dym_test_res$rel_time = as.numeric(sub("year::(-?\\d+).*", "\\1", rownames(dym_test_res)))
dym_test_res[nrow(dym_test_res)+1, c("Estimate", "rel_time")] = c(0, -1)

ggplot(dym_test_res, aes(x = rel_time, y = Estimate)) +
  geom_errorbar(aes(ymin = Estimate-`Std. Error`, ymax = Estimate+`Std. Error`), width = 0.2, colour = "gray50") + 
  geom_vline(xintercept=0, color = 'gray', linetype = 'dashed', size = 0.3) + 
  geom_hline(yintercept=0.0, color = 'gray', linetype = 'dashed', size = 0.3) + 
  geom_line(colour = "steelblue4") +
  geom_point(size=2, colour = "steelblue4") +
  geom_point(aes(x=-1, y=0), colour="brown4", size=2) + 
  scale_y_continuous(name = "动态效应", limits = c(-1, 1), breaks = seq(-1, 1, 0.2)) + 
  scale_x_continuous(name = "政策开始实施的相对年份", limits = c(-4, 5), breaks = seq(-4, 5, 1)) + 
  scale_color_aaas() + 
  theme_bw() + 
  theme(panel.grid=element_blank())
ggsave('fdi_dym_eff.png', width = 8, height = 4)



write.xlsx(dta, "eids_dta.xlsx")