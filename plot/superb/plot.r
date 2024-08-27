# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)

# read data
df = read.csv('superb.csv')

# subdataframes for each task
df_asr = df[df$superb_task == "asr",]
df_asr_zh = df[df$superb_task == "asr-ood-zh",]
df_asv = df[df$superb_task == "asv",]
df_ic = df[df$superb_task == "ic",]
df_er = df[df$superb_task == "er",]
df_pr = df[df$superb_task == "pr",]

# x_axis for every plot
x_axis = scale_x_continuous(
  name="steps",
  labels = unit_format(unit = "k", scale = 1e-3),
  limits=c(0, 400000),
  expand = c(0.02, 0.02)  # Set expand to include only the bottom limit
)

# legend order
custom_legend_order <- c(
  "87.5 sec",
  "150 sec",
  "5 min",
  "10 min",
  "20 min",
  "40 min",
  "80 min"
)

# functions for determing breaks in logscale
breaks_5log10 <- function(x) {
  low <- floor(log10(min(x)/5))
  high <- ceiling(log10(max(x)/5))
  
  5 * 10^(seq.int(low, high))
}

breaks_log10 <- function(x) {
  low <- floor(log10(min(x)))
  high <- ceiling(log10(max(x)))
  
  10^(seq.int(low, high))
}

# set line width
lw = 0.5


# per 
pr = (
  ggplot(df_pr)
  + aes(
    hours_seen, 
    metric.value, 
    color=batch.size,
    shape=batch.size
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='observed hours during SSL', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='PER',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 100, by = 10),
    limits = c(0, 50)
  )
  + scale_colour_colorblind(
    name='batch size',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='batch size',
    breaks=custom_legend_order,
    values=seq(0,6)
  )
  + ggtitle('phoneneme recog. (en)')
)

# asr
asr = (
  ggplot(df_asr)
  + aes(
    hours_seen, 
    metric.value, 
    color=batch.size,
    shape=batch.size
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='observed hours during SSL', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='WER',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 100, by = 10),
    limits = c(0, 30)
  )
  + scale_colour_colorblind(
    name='batch size',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='batch size',
    breaks=custom_legend_order,
    values=seq(0,6)
  )
  + ggtitle('ASR (en)')
)

# asr-zh
asr_zh = (
  ggplot(df_asr_zh)
  + aes(
    hours_seen, 
    metric.value, 
    color=batch.size,
    shape=batch.size
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='observed hours during SSL', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='CER',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 100, by = 10),
    limits = c(20, 40)
  )
  + scale_colour_colorblind(
    name='batch size',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='batch size',
    breaks=custom_legend_order,
    values=seq(0,6)
  )
  + ggtitle('OOD ASR (mandarin)')
)

# speaker verification
asv = (
  ggplot(df_asv)
  + aes(
    hours_seen, 
    metric.value, 
    color=batch.size,
    shape=batch.size
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='observed hours during SSL', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='EER',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 15, by = 5),
    limits = c(0, 15)
  )
  + scale_colour_colorblind(
    name='batch size',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='batch size',
    breaks=custom_legend_order,
    values=seq(0,6)
  )
  + ggtitle('speaker recognition')
)


# emotion
er = (
  ggplot(df_er)
  + aes(
    hours_seen, 
    metric.value, 
    color=batch.size,
    shape=batch.size
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='observed hours during SSL', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='accuracy',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 100, by = 2),
    limits = c(55, 65)
  )
  + scale_colour_colorblind(
    name='batch size',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='batch size',
    breaks=custom_legend_order,
    values=seq(0,6)
  )
  + ggtitle('emotion recognition')
)

# intent classification
ic = (
  ggplot(df_ic)
  + aes(
    hours_seen, 
    metric.value, 
    color=batch.size,
    shape=batch.size
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='observed hours during SSL', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='accuracy',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 100, by = 10),
    limits = c(70, 100)
  )
  + scale_colour_colorblind(
    name='batch size',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='batch size',
    breaks=custom_legend_order,
    values=seq(0,6)
  )
  + ggtitle('intent classification')
)


# subplots
g = (
  (pr | asr | asr_zh)
  / (asv | er | ic)
  + plot_layout(guides = "collect")
  & theme(
    legend.direction = "horizontal",
    legend.position = "bottom",
    plot.margin = unit(c(0.1,0.5,0.1,0.1), "cm")
  )
)
 
g

# save
ggsave(
  file="superb.pdf",
  plot=g,
  device = cairo_pdf,
  width = 200,
  height = 120,
  units = "mm"
)