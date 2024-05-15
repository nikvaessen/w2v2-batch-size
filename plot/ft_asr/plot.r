# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)

# read data
df = read.csv('ft_asr_lm.csv')

# process data

# only use test set results
df = subset(df, !(eval_dataset %in% c('dev-clean', 'dev-other')))

# change lm true/false to label for facet grid
df$lm = as.logical(df$lm)
df = df %>%
  mutate(lm = ifelse(lm == FALSE, 'letter decoding', '4-gram word decoding'))

# clip WER values to 1 and make percentages
df <- df %>%
  mutate(value = pmax(0, pmin(1, value)),  # Clip values between 0 and 1
         value = value * 100)              # Multiply values by 100

# change order of rows in facet grid
df$lm = factor(df$lm, levels = c("letter decoding", "4-gram word decoding"))

# x-axis for all graphs
x_axis = scale_x_discrete(
  name='self-supervised learning batch size',
  limits=c('scratch','0gpu', '1gpu', '2gpu', '4gpu', '8gpu', '16gpu', '32gpu'),
  labels=c('scratch', '87.5 s', '150 s', '5 min', '10 min', '20 min', '40 min', '80 min')
)

# legend for graphs
legend_breaks <- c(
  "ls10m", "ls1h", "ls10h", "ls100h", 'ls960h'
)
legend_labels = c('10 min', '1 hour', '10 hours', '100 hours', '960 hours')

# remove rows where column lm is equal to '4-gram word decoding'
df <- subset(df, lm != '4-gram word decoding')


# plot
g = (
  ggplot(df)
  + aes(
    batch.size, 
    value, 
    color=ft_dataset,
    shape=ft_dataset,
    group=ft_dataset
  )
  + geom_point()
  + geom_line()
  + x_axis
  + scale_y_continuous(name='WER after fine-tuning on labeled data',
                       labels = scales::percent_format(scale = 1),
                       breaks = seq(0, 100, by = 20))
  + scale_colour_colorblind(
    name='amount of labels for fine-tuning',
    breaks=legend_breaks,
    labels=legend_labels,
  )
  + scale_shape(
    name='amount of labels for fine-tuning',
    breaks=legend_breaks, 
    labels=legend_labels, 
  )
)

# change order of 
g = (
  g
  + facet_grid(cols = vars(eval_dataset)) 
  + theme(legend.position = "bottom")
)


g

ggsave(
  file="ft_plot_a4.pdf",
  plot=g,
  device = cairo_pdf,
  width = 210,
  height = 297/2-10,
  units = "mm"
)


