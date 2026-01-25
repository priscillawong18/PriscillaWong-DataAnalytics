library(readr)
library(EnvStats)
library(nortest)

# set working directory (relative path)
setwd("~/Data Analytics/lab1/")

# read data
epi.data <- read_csv("epi_results_2024_pop_gdp.csv")

WWC <- epi.data$WWC.new
WWT <- epi.data$WWT.new

# Variable summaries
summary(WWC)
summary(WWT)

# Variable boxplot
boxplot(WWC, WWT, names = c("WWC","WWT"))

# Histograms with overlayed theoretical probability distributions

hist(WWC, prob = TRUE, main = "WWC Histogram")
lines(density(WWC, bw = "SJ"))
rug(WWC)

x1 <- seq(min(WWC, na.rm=TRUE), max(WWC, na.rm=TRUE), 1)
d1 <- dnorm(x1, mean = mean(WWC, na.rm=TRUE), sd = sd(WWC, na.rm=TRUE))
hist(WWC, prob=TRUE, main="WWC with Normal Curve")
lines(x1, d1)

hist(WWT, prob = TRUE, main = "WWT Histogram")
lines(density(WWT, bw = "SJ"))
rug(WWT)

x2 <- seq(min(WWT, na.rm=TRUE), max(WWT, na.rm=TRUE), 1)
d2 <- dnorm(x2, mean = mean(WWT, na.rm=TRUE), sd = sd(WWT, na.rm=TRUE))
hist(WWT, prob=TRUE, main="WWT with Normal Curve")
lines(x2, d2)

# ECDF plots
plot(ecdf(WWC), do.points=FALSE, verticals=TRUE)
plot(ecdf(WWT), do.points=FALSE, verticals=TRUE)

# QQ plots of each variable against the normal distribution
qqnorm(WWC, main = "QQ Plot: WWC"); qqline(WWC)
qqnorm(WWT, main = "QQ Plot: WWT"); qqline(WWT)

# QQ plot of the 2 variables against each other
qqplot(WWC, WWT, xlab = "WWC Quantiles", ylab = "WWT Quantiles", main = "QQ Plot: WWC vs WWT")

# Normality statistical tests for each variable

WWC.NAs <- is.na(WWC)
WWC.complete <- WWC[!WWC.NAs]

WWT.NAs <- is.na(WWT)
WWT.complete <- WWT[!WWT.NAs]

shapiro.test(WWC.complete)
shapiro.test(WWT.complete)

ad.test(WWC.complete)
ad.test(WWT.complete)

# A statistical test for whether the variables having identical distribution
ks.test(WWC.complete, WWT.complete)

wilcox.test(WWC.complete, WWT.complete)

