# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)

# read data
df = read.csv("merged.csv")


# clip WER values to 1 and make percentages
df <- df %>%
  mutate(wer.clean = pmax(0, pmin(1, wer.clean)),  # Clip values between 0 and 1
         wer.clean = wer.clean * 100)              # Multiply values by 100

df <- df %>%
  mutate(wer.other = pmax(0, pmin(1, wer.other)),  # Clip values between 0 and 1
         wer.other = wer.other * 100)              # Multiply values by 100

# make hours seen in k
df$hours.seen <- df$hours.seen / 1000

# legend for graphs
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


# plot
g = (
  ggplot(df)
  + aes(
    hours.seen, 
    wer.clean, 
    color=batch.size.label,
    shape=batch.size.label
  )
  + geom_point()
  + geom_line()
  + annotation_logticks(sides='b')
  + scale_x_log10(
    name='Observed data in hours during self-supervision', 
    labels = scales::label_comma(suffix = ' k'),
    breaks = breaks_log10,
    minor_breaks = breaks_5log10,
  )
  + scale_y_continuous(
    name='WER after fine-tuning on labeled data',
    labels = scales::percent_format(scale = 1),
    breaks = seq(0, 100, by = 20),
    limits = c(0, NA)
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
  + labs(title="WER after fine-tuning when having seen a certain amount of data during SSL")
)


# change order of 
g = (
  g
  + facet_grid(cols = vars(ft_dataset)) 
  & theme(
    legend.direction = "horizontal",
    legend.position = "bottom",
    plot.margin = unit(c(0.1,0.5,0.1,0.1), "cm")
  )
)


g

ggsave(
  file="merged.pdf",
  device = cairo_pdf,
  width = 210,
  height = 297/2-60,
  units = "mm"
)


