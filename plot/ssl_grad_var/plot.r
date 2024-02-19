# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)
library(dplyr)

# Concatenate the list of data frames into a single data frame
df = read_csv("merged.csv")

print(df)

custom_legend_order <- c(
  "87.5 sec",
  "150 sec",
  "5 min",
  "10 min",
  "20 min",
  "40 min",
  "80 min"
)

x_axis = scale_x_continuous(
  name="steps",
  labels = unit_format(unit = "k", scale = 1e-3),
  limits=c(0, 400000),
  expand = c(0.02, 0.02)  # Set expand to include only the bottom limit
)

g = (
  ggplot(df)
  + aes(
      train_step, 
      average_variance, 
      color=batch_size, 
     #linetype=batch_size
    )
  + geom_line()
  #+ geom_point()
  + x_axis
  + scale_y_continuous(limits = c(0, 0.30))
  + scale_color_colorblind(
      name="batch size", 
      breaks=custom_legend_order, 
      #guide=guide_legend(nrow = 3)
    )
  + scale_linetype_manual(
      name="batch size",
      breaks=custom_legend_order, 
      values=c(6, 5, 7, 3, 4, 2, 1)
    )
#  + scale_shape_manual(
#      name='batch size',
#      breaks=custom_legend_order,
#      values=seq(0,6)
#    )
  + ylab("average std of gradient")
     
)

#g = g + theme(legend.position = "bottom")

g

ggsave(
  file="gradient_std.pdf",
  plot=g,
  device = cairo_pdf,
  width = 100,
  height = 297/2-70,
  units = "mm"
)



