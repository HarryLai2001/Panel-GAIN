library(haven)
library(tidyverse)
library(ggplot2)
library(openxlsx)
library(ggsci)
library(fixest)


dta = read_dta("数据-城市.dta")
dta = dta[order(dta$id, dta$年度),]

dta$policy = case_when(dta$省份 %in% c("辽宁省", 
                                     "湖北省", 
                                     "陕西省", 
                                     "云南省") 
                       | dta$城市 %in% c("天津市",
                                       "重庆市",
                                       "深圳市",
                                       "厦门市",
                                       "杭州市",
                                       "南昌市",
                                       "贵阳市",
                                       "保定市") ~ 2010,
                       dta$省份 %in% c("海南省")
                       | dta$城市 %in% c("北京市", 
                                       "上海市",
                                       "石家庄市",
                                       "秦皇岛市",
                                       "晋城市",
                                       "呼伦贝尔市",
                                       "吉林市",
                                       "苏州市",
                                       "淮安市",
                                       "镇江市",
                                       "宁波市",
                                       "温州市",
                                       "池州市",
                                       "南平市",
                                       "景德镇市",
                                       "赣州市",
                                       "青岛市",
                                       "济源市",
                                       "广元市",
                                       "遵义市",
                                       "金昌市",
                                       "乌鲁木齐市") ~ 2013,
                       dta$城市 %in% c("乌海市",
                                     "南京市",
                                     "常州市",
                                     "嘉兴市",
                                     "金华市",
                                     "衢州市",
                                     "合肥市",
                                     "淮北市",
                                     "黄山市",
                                     "六安市",
                                     "宣城市",
                                     "三明市",
                                     "共青城市",
                                     "吉安市",
                                     "抚州市",
                                     "济南市",
                                     "烟台市",
                                     "潍坊市",
                                     "长沙市",
                                     "株洲市",
                                     "湘潭市",
                                     "郴州市",
                                     "成都市",
                                     "普洱市",
                                     "拉萨市",
                                     "兰州市",
                                     "敦煌市",
                                     "西宁市",
                                     "银川市",
                                     "吴忠市",
                                     "昌吉市",
                                     "伊宁市",
                                     "和田市",
                                     "第一师阿拉尔市") ~ 2017,
                       TRUE ~ 9999)

dta$lccpost = ifelse(dta$年度 >= dta$policy, 1, 0)



sta = dta %>%
  group_by(id) %>%
  summarise(first_treat = first(policy)) %>%
  group_by(first_treat) %>%
  summarise(n = n())

summ = dta %>%
  group_by(policy, 年度) %>%
  summarise(y_mean = mean(co2))
summ$policy = factor(summ$policy,
                     levels = c(2010, 2013, 2017, 9999),
                     labels = c("2010", "2013", "2017", "never"))

ggplot(summ, aes(x = 年度, y = y_mean, group = policy, colour = policy)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept=2010-0.5, color = 'gray', linetype = 'dashed', size = 0.3) + 
  guides(color=guide_legend(title = "组别")) + 
  scale_y_continuous(name = "co2", limits = c(11.5, 13), breaks = seq(11.5, 13, 0.1)) +
  scale_x_continuous(name = "年份", limits = c(2007, 2019), breaks = seq(2007, 2019, 1)) +
  scale_color_aaas() + 
  theme_bw() + 
  theme(panel.grid=element_blank())
ggsave('co2_avg.png', width = 8, height = 4)


dym_test = twfe = feols(co2 ~ sunab(policy, 年度, no_agg = FALSE) | id + 年度,
                        data = dta)
dym_test_res = as.data.frame(dym_test$coeftable)
dym_test_res$rel_time = as.numeric(sub("年度::(-?\\d+).*", "\\1", rownames(dym_test_res)))

ggplot(dym_test_res, aes(x = rel_time, y = Estimate)) +
  geom_errorbar(aes(ymin = Estimate-`Std. Error`, ymax = Estimate+`Std. Error`), width = 0.2, colour = "gray50") + 
  geom_vline(xintercept=0, color = 'gray', linetype = 'dashed', size = 0.3) + 
  geom_hline(yintercept=0.0, color = 'gray', linetype = 'dashed', size = 0.3) + 
  geom_line(colour = "steelblue4") +
  geom_point(size=2, colour = "steelblue4") +
  geom_point(aes(x=-1, y=0), colour="brown4", size=2) + 
  scale_y_continuous(name = "动态效应", limits = c(-0.5, 0.5), breaks = seq(-0.5, 0.5, 0.1)) + 
  scale_x_continuous(name = "政策开始实施的相对年份", limits = c(-10, 9), breaks = seq(-10, 9, 1)) + 
  scale_color_aaas() + 
  theme_bw() + 
  theme(panel.grid=element_blank())
ggsave('co2_dym_eff.png', width = 8, height = 4)

dta$group = case_when(dta$policy == 2010 ~ 0,
                      dta$policy == 2013 ~ 1,
                      dta$policy == 2017 ~ 2,
                      TRUE ~ 3)

write.xlsx(dta, "co2_dta.xlsx")