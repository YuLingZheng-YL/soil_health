abundance <- read.delim('abundance.txt', sep = '\t', check.names = FALSE)
library(vegan)
library(ggplot2)
abundance <- t(abundance)
abundance.distance <- vegdist(abundance, method = 'bray')
df_nmds <- metaMDS(abundance.distance, k = 2)
summary(df_nmds)
df_nmds_stress <- df_nmds$stress
df_nmds_stress
stressplot(df_nmds)
df_points <- as.data.frame(df_nmds$points)
df_points$samples <- row.names(df_points)
names(df_points)[1:2] <- c('NMDS1', 'NMDS2')
head(df_points)
p <- ggplot(df_points,aes(x=NMDS1, y=NMDS2))+
  geom_point(size=3)+
  theme_bw()
p
