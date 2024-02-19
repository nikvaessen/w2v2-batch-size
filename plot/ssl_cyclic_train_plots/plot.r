# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)

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

# set line width
lw = 0.5

df = read_csv('csv/cb1_sim_avg.csv')

# codebook similarity
cb1_avg = (
  ggplot(read_csv('csv/cb1_sim_avg.csv')) 
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + scale_y_continuous(limits = c(0.5, 0.8))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Average similarity cb1", tag="G", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Similarity')
)
cb1_min = (
  ggplot(read_csv('csv/cb1_sim_min.csv')) 
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + scale_y_continuous(limits = c(0.1, 0.8))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Minimum similarity cb1", tag="H", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Similarity')
)
cb1_max = (
  ggplot(read_csv('csv/cb1_sim_max.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + scale_y_continuous(limits = c(0.8, 1.0))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Maximum similarity cb1", tag="I", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Similarity')
)

cb2_avg = (
  ggplot(read_csv('csv/cb2_sim_avg.csv')) 
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + scale_y_continuous(limits = c(0.5, 0.8))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Average similarity cb2", tag="J", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Similarity')
)
cb2_min = (
  ggplot(read_csv('csv/cb2_sim_min.csv')) 
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + scale_y_continuous(limits = c(0.1, 0.8))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Minimum similarity cb2", tag="K", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Similarity')
)
cb2_max = (
  ggplot(read_csv('csv/cb2_sim_max.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + scale_y_continuous(limits = c(0.8, 1.0))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Maximum similarity cb2", tag="L", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Similarity')
)


# losses
loss_c = (
  ggplot(read_csv('csv/val_loss_contrastive.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  # + scale_y_continuous(limits = c(4000, 10000))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Contrastive loss", tag="A", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('loss')
)

loss_d = (
  ggplot(read_csv('csv/val_loss_diversity.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  # + scale_y_continuous(limits = c(0, 70))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "Diversity loss", tag="B", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('loss')
)
loss_l2 = (
  ggplot(read_csv('csv/val_loss_l2.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  # + scale_y_continuous(limits = c(0, 8))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + labs(title = "L2-penalty loss", tag="C", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('loss')
)

# perplexity
perplex_cb1 = (
  ggplot(read_csv('csv/val_perplexity_cb1.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + coord_cartesian(ylim = c(0, 225))
  # + scale_y_continuous(limits = c(0, 225))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide="none")
  + labs(title = "Perplexity cb1", tag="E", color = "batch size", linetype = "batch size", shape = "batch size") 
  + ylab('Perplexity')
)
perplex_cb2 = (
  ggplot(read_csv('csv/val_perplexity_cb2.csv'))
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + x_axis 
  + coord_cartesian(ylim = c(0, 225))
  # + scale_y_continuous(limits = c(-50, 225))
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide="none")
  + labs(title = "Perplexity cb2", tag="F", color = "batch size", linetype = "batch size")
  + ylab('Perplexity')
)

# accuracy
acc = (
  ggplot(read_csv('csv/val_acc.csv')) 
  + aes(
    step, 
    value, 
    color=`batch size`,
    #linetype=`batch size`
  )
  + geom_line(linewidth=lw) 
  + coord_cartesian(ylim = c(0, 0.6))
  + x_axis 
  # + scale_linetype(breaks=custom_legend_order, guide=guide_legend(nrow = 1))
  + scale_color_colorblind(breaks=custom_legend_order, guide="none")
  + labs(title = "Accuracy", tag="D", color = "batch size", linetype = "batch size") 
  + ylab('Accuracy')
)

# subplots
g = (
  (loss_c | loss_d | loss_l2)
  / (acc | perplex_cb1 | perplex_cb2)
  / (cb1_avg | cb1_min | cb1_max)
  / (cb2_avg | cb2_min | cb2_max)
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
  file="train_plot_a4.pdf",
  plot=g,
  device = cairo_pdf,
  width = 210,
  height = 297/2 + 50,
  units = "mm"
)
