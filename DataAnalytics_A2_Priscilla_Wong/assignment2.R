library(readr)
library(ggplot2)
library(class)
library(dplyr)

# Import and Remove NA Data
epi_data <- read_csv("~/Data Analytics/DataAnalytics_A2_Priscilla_Wong/epi_results_2024_pop_gdp.csv")
epi_clean <- epi_data %>% 
  filter(!is.na(EPI.new), !is.na(population), !is.na(gdp), !is.na(ECO.new))

## Variable Distributions

# 1.1. Histogram of var with density lines overlayed
hist(epi_clean$EPI.new, prob = TRUE, col = "yellow", 
     main = "Histogram of EPI.new", 
     xlab = "EPI Score (New)")
lines(density(epi_clean$EPI.new, na.rm = TRUE), col = "red", lwd = 2)

# 1.2. Boxplots of var (one for each region, multiple box plots in one figure)

#par(mar = c(10, 4, 4, 2))
boxplot(EPI.new ~ region,
        data = epi_clean,
        col = "lightblue",
        las = 2,
        main = "EPI.new Distribution by Region",
        xlab = "Region",
        ylab = "EPI.new")

# Two Subsets
apac <- epi_clean %>% filter(region == "Asia-Pacific")
latam <- epi_clean %>% filter(region == "Latin America & Caribbean")

# 1.3. Histograms of the vars for each region
hist(apac$EPI.new, main = "EPI.new: APAC", xlab = "Score", col = "orange")
hist(latam$EPI.new, main = "EPI.new: LATAM", xlab = "Score", col = "green")

# 1.4. QQ plot for the variable between the 2 subsets
qqplot(apac$EPI.new, latam$EPI.new,
       xlab = "APAC Quantiles", ylab = "LATAM Quantiles", 
       main = "QQ Plot: APAC vs LATAM")

## Linear Models

# 3.1. Population and GDP against EPI.new

ggplot(apac, aes(log10(population), EPI.new)) +
  geom_point() +
  geom_smooth(method="lm") +
  labs(title="APAC: EPI vs log(Population)")

ggplot(apac, aes(log10(gdp), EPI.new)) +
  geom_point() +
  geom_smooth(method="lm") +
  labs(title="APAC: EPI vs log(GDP)")

ggplot(latam, aes(log10(population), EPI.new)) +
  geom_point() +
  geom_smooth(method="lm") +
  labs(title="LATAM: EPI vs log(Population)")

ggplot(latam, aes(log10(gdp), EPI.new)) +
  geom_point() +
  geom_smooth(method="lm") +
  labs(title="LATAM: EPI vs log(GDP)")

# 3.2. Fit 2 Linear Models

# APAC Models and Residual Plots
apac_gdp <- lm(EPI.new ~ log10(gdp), data = apac)
apac_pop <- lm(EPI.new ~ log10(population), data = apac)

summary(apac_gdp)
summary(apac_pop)

plot(apac_gdp$residuals, main="APAC GDP Residuals")
abline(h=0, col="red")

plot(apac_pop$residuals, main="APAC Population Residuals")
abline(h=0, col="red")

# LATAM Models and Residual Plots
latam_gdp <- lm(EPI.new ~ log10(gdp), data = latam)
latam_pop <- lm(EPI.new ~ log10(population), data = latam)

summary(latam_gdp)
summary(latam_pop)

plot(latam_gdp$residuals, main="LATAM GDP Residuals")
abline(h=0, col="red")

plot(latam_pop$residuals, main="LATAM Population Residuals")
abline(h=0, col="red")

# 3.3. Compare Models
compare_models <- function(data_subset, region_name) {
  m_gdp <- lm(EPI.new ~ log10(gdp), data = data_subset)
  m_pop <- lm(EPI.new ~ log10(population), data = data_subset)
  
  cat(region_name, "Model Fit Comparison\n")
  cat("GDP model R-squared:", summary(m_gdp)$r.squared, "\n")
  cat("Population model R-squared:", summary(m_pop)$r.squared, "\n")
}

compare_models(apac, "APAC")
compare_models(latam, "LATAM")

## Classification (kNN)

# Derive subset with 3 regions
knn_data <- epi_clean %>%
  filter(region %in% c("Eastern Europe", "Asia-Pacific", "Latin America & Caribbean")) %>%
  select(region, population, gdp, EPI.new, ECO.new) %>%
  na.omit()

# 4.1. Train kNN model using "region" as the class with 3 input variables

# Log transform to reduce skew
knn_data$log_pop <- log10(knn_data$population)
knn_data$log_gdp <- log10(knn_data$gdp)

# Features for model 1
features1 <- knn_data %>%
  select(log_pop, log_gdp, EPI.new)

# Scale features
features1_scaled <- scale(features1)
labels <- knn_data$region

# Train split
set.seed(1)
train_index <- sample(1:nrow(features1_scaled), 0.7*nrow(features1_scaled))
train_x <- features1_scaled[train_index, ]
test_x  <- features1_scaled[-train_index, ]
train_y <- labels[train_index]
test_y  <- labels[-train_index]

# Get optimal k
accuracy <- c()
k_values <- seq(1, 25, 2)

for (k in k_values) {
  pred <- knn(train_x, test_x, cl = train_y, k = k)
  acc <- sum(pred == test_y) / length(test_y)
  accuracy <- c(accuracy, acc)
}

plot(k_values, accuracy, type="b", pch=19,
     xlab="k", ylab="Accuracy", main="k Optimization")

best_k <- k_values[which.max(accuracy)]
best_k

# Train best and confusion matrix
pred1 <- knn(train_x, test_x, cl = train_y, k = best_k)

conf_matrix1 <- table(Predicted = pred1, Actual = test_y)
conf_matrix1

accuracy1 <- sum(pred1 == test_y) / length(test_y)
accuracy1

# 4.2. Train another model with a 3rd variable

features2 <- knn_data %>%
  select(log_pop, log_gdp, ECO.new)

features2_scaled <- scale(features2)

train_x2 <- features2_scaled[train_index, ]
test_x2  <- features2_scaled[-train_index, ]

pred2 <- knn(train_x2, test_x2, cl = train_y, k = best_k)

conf_matrix2 <- table(Predicted = pred2, Actual = test_y)
conf_matrix2

accuracy2 <- sum(pred2 == test_y) / length(test_y)
accuracy2
