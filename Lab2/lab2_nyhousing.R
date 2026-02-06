library("ggplot2")
library("readr")

## Read Dataset
NY_House_Dataset <- read_csv("~/Data Analytics/lab2/NY-House-Dataset.csv")
dataset <- NY_House_Dataset

## Data Cleaning

# Filter data by removing extreme price outliers
dataset <- dataset[dataset$PRICE<195000000,]

# Remove missing values in variables
dataset <- dataset[complete.cases(dataset[, c("PRICE", "PROPERTYSQFT", "BEDS", "BATH")]), ]

# Transform variables
dataset$logPRICE <- log10(dataset$PRICE)
dataset$logSQFT <- log10(dataset$PROPERTYSQFT)

## MODEL 1: Price and PropertySqft
lmod1 <- lm(logPRICE ~ logSQFT, data = dataset) 
summary(lmod1)

# Plot most significant variable vs Price
ggplot(dataset, aes(x = logSQFT, y = logPRICE)) +
  geom_point() +
  stat_smooth(method = "lm", col = "red") +
  labs(title = "Model 1: log(PRICE) vs log(PROPERTYSQFT)")

# Residual plot
ggplot(lmod1, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted (Model 1)")

## MODEL 2: Price and PropertySqrt + Beds
lmod2 <- lm(logPRICE ~ logSQFT + BEDS, data = dataset)
summary(lmod2)

# Most significant variable (typically logSQFT)
ggplot(dataset, aes(x = logSQFT, y = logPRICE)) +
  geom_point() +
  stat_smooth(method = "lm", col = "red") +
  labs(title = "Model 2: log(PRICE) vs log(PROPERTYSQFT)")

# Residual plot
ggplot(lmod2, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted (Model 2)")

## MODEL 3: Price and PropertySqrt + Beds + Bath
lmod3 <- lm(logPRICE ~ logSQFT + BEDS + BATH, data = dataset)
summary(lmod3)

# Most significant variable
ggplot(dataset, aes(x = logSQFT, y = logPRICE)) +
  geom_point() +
  stat_smooth(method = "lm", col = "red") +
  labs(title = "Model 3: log(PRICE) vs log(PROPERTYSQFT)")


# Residual plot
ggplot(lmod3, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted (Model 3)")



### THE END ###

