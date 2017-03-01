#t-sne

#Load Required Packages
library(feather)
library(data.table)
library(dplyr)
library(dtplyr)
library(Rtsne)
library(ggplot2)
library(RColorBrewer)

#Read in feather file - from nmf-tsne.py
tsne_embedding = read_feather('tsne_embedding.feather') %>% 
  as.data.table()

#Recode topics
tsne_embedding = tsne_embedding %>%
  mutate(topic = recode(topic, `0` = "Generic", `1` = "Tea", `2` = "Coffee", `3` = "Pets", `4` = "Shopping", `5` = "Chocolate"))

#tsne plot
ggplot(tsne_embedding, aes(x = x, y = y, colour = topic)) +
  geom_point(size=1) +
  guides(colour = guide_legend(title = "Topic", override.aes = list(size = 2))) + #Increase legend dot size
  xlab("") + ylab("") + #Remove axis labels
  ggtitle("t-SNE 2D Embedding of Text Topics") +
  theme_light(base_size=10) +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        axis.line = element_blank(),
        panel.border = element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  scale_colour_brewer(palette = "Set1")

ggsave("eda/tsne-topics.jpg", plot = last_plot(), dpi = 1000, width = 10)
