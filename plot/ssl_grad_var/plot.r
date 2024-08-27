# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)
library(dplyr)

# Concatenate the list of data frames into a single data frame
df = read_csv("merged1.csv")
df$dw <- as.character(df$dw)

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
      step,
      std,
      color=bs,
      linetype=dw
    )
  + geom_line()
  + x_axis
  + scale_y_continuous(limits = c(0, 0.30))
  + scale_color_colorblind(
      name="batch size",
      breaks=custom_legend_order,
      guide=guide_legend(nrow = 1)
    )
  + scale_linetype_manual(
      name="diversity loss weight",
      breaks=c("0.05", "0.1"),
      values=c(2, 1)
    )
 # + scale_shape_manual(
 #     name='batch size',
 #     breaks=custom_legend_order,
 #     values=seq(0,6)
 #   )
  + ylab("average std of gradient")
  + labs(title="Average standard deviation of gradient of each parameter (n=10)")

)

g = g + theme(legend.position = "bottom")
g


ggsave(
  file="gradient_std.pdf",
  plot=g,
  device = cairo_pdf,
  width = 210,
  height = 297/2-60,
  units = "mm"
)
