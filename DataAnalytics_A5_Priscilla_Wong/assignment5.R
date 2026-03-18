library(readr)
library(ggplot2)
library(class)
library(dplyr)
library(randomForest)
library(caret)
library(rpart)

# Import and Remove NA Data
nycdf <- read_csv("~/Data Analytics/DataAnalytics_A5_Priscilla_Wong/NYC_Citywide_Annualized_Calendar_Sales_Update_20241107.csv", show_col_types = FALSE)

# Subset for Queens
queens <- nycdf %>% filter(BOROUGH == "QUEENS")

# Convert important variables to numeric
queens$`SALE PRICE` <- as.numeric(queens$`SALE PRICE`)
queens$`GROSS SQUARE FEET` <- as.numeric(queens$`GROSS SQUARE FEET`)
queens$`LAND SQUARE FEET` <- as.numeric(queens$`LAND SQUARE FEET`)
queens$`YEAR BUILT` <- as.numeric(queens$`YEAR BUILT`)

# Remove bad values (0 price or missing)
queens <- queens %>% filter(`SALE PRICE` > 0)

## 1a. Exploratory Data Analysis

# Variable Distributions (Plotting 4 variables)
# Sale Price Distribution
ggplot(queens, aes(x = `SALE PRICE`)) +
  geom_histogram(bins = 50, fill = "blue", color = "black") +
  scale_x_log10() +
  labs(title = "Queens: Distribution of Sale Price (Log Scale)",
       x = "Sale Price",
       y = "Count")

# Gross Square Feet Distribution
ggplot(queens, aes(x = `GROSS SQUARE FEET`)) +
  geom_histogram(bins = 40, fill = "green", color = "black") +
  scale_x_log10() +
  labs(title = "Queens: Distribution of Gross Square Feet",
       x = "Gross Square Feet",
       y = "Count")

# Land Square Feet Distribution
ggplot(queens, aes(x = `LAND SQUARE FEET`)) +
  geom_histogram(bins = 40, fill = "red", color = "black") +
  scale_x_log10() +
  labs(title = "Queens: Distribution of Land Square Feet",
       x = "Land Square Feet",
       y = "Count")

# Year Built Distribution
ggplot(queens %>% filter(`YEAR BUILT` > 0),
       aes(x = `YEAR BUILT`)) +
  geom_histogram(bins = 40, fill = "purple", color = "black") +
  labs(title = "Queens: Distribution of Year Built",
       x = "Year Built",
       y = "Count")

# Outlier Identification Plot for Sale Price
ggplot(queens, aes(x = `SALE PRICE`)) +
  geom_boxplot(fill = "orange", outlier.color = "red") +
  scale_x_log10() +
  labs(title = "Queens: Boxplot of Sale Price (Outliers)",
       x = "Sale Price") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

## 1b. Regression Analysis

# Data Cleaning

queens_clean <- queens %>%
  filter(!is.na(`SALE PRICE`), 
         !is.na(`GROSS SQUARE FEET`), 
         !is.na(`LAND SQUARE FEET`), 
         !is.na(`YEAR BUILT`)) %>%
  filter(`SALE PRICE` > 10000) %>% 
  filter(`SALE PRICE` < quantile(`SALE PRICE`, 0.99, na.rm = TRUE))

queens_clean <- queens_clean %>%
  rename(
    SALE_PRICE = `SALE PRICE`,
    GROSS_SQUARE_FEET = `GROSS SQUARE FEET`,
    LAND_SQUARE_FEET = `LAND SQUARE FEET`,
    YEAR_BUILT = `YEAR BUILT`
  )

set.seed(123) 
train_indices <- sample(seq_len(nrow(queens_clean)), size = 0.8 * nrow(queens_clean))
train_data <- queens_clean[train_indices, ]
test_data <- queens_clean[-train_indices, ]

# Model 1: Multiple Linear Regression
lm_model <- lm(SALE_PRICE ~ GROSS_SQUARE_FEET + LAND_SQUARE_FEET + YEAR_BUILT, data = train_data)

# Predict on test data
lm_preds <- predict(lm_model, newdata = test_data)

# Calculate R-squared manually to evaluate
tss <- sum((test_data$SALE_PRICE - mean(test_data$SALE_PRICE))^2)
rss_lm <- sum((test_data$SALE_PRICE - lm_preds)^2)
r2_lm <- 1 - (rss_lm / tss)
cat("Linear Regression R-squared:", round(r2_lm, 4), "\n")

# Model 2: Random Forest Regressor
rf_model <- randomForest(SALE_PRICE ~ GROSS_SQUARE_FEET + LAND_SQUARE_FEET + YEAR_BUILT, 
                         data = train_data, ntree = 100)
# Predict on test data
rf_preds <- predict(rf_model, newdata = test_data)

# Calculate R-squared
rss_rf <- sum((test_data$SALE_PRICE - rf_preds)^2)
r2_rf <- 1 - (rss_rf / tss)
cat("Random Forest R-squared:", round(r2_rf, 4), "\n")


## 2. Classification Analysis

# Convert total units to numeric
queens$`TOTAL UNITS` <- as.numeric(queens$`TOTAL UNITS`)

# Queens neighborhoods
target <- c("LONG ISLAND CITY", "FLUSHING-NORTH", "BAYSIDE")

# Subset the data, rename columns to avoid space errors, and remove NAs
queens_class <- queens %>%
  filter(NEIGHBORHOOD %in% target) %>%
  rename(
    SALE_PRICE = `SALE PRICE`,
    GROSS_SQUARE_FEET = `GROSS SQUARE FEET`,
    TOTAL_UNITS = `TOTAL UNITS`
  ) %>%
  filter(!is.na(SALE_PRICE), !is.na(GROSS_SQUARE_FEET), !is.na(TOTAL_UNITS)) %>%
  filter(SALE_PRICE > 10000)

# Convert the target into a Factor
queens_class$NEIGHBORHOOD <- as.factor(droplevels(as.factor(queens_class$NEIGHBORHOOD)))

# Simple One-Round Cross Validation
set.seed(42)
train_idx_c <- sample(seq_len(nrow(queens_class)), size = 0.8 * nrow(queens_class))
train_c <- queens_class[train_idx_c, ]
test_c <- queens_class[-train_idx_c, ]

# Model 1: Random Forest Classifier
rf_class <- randomForest(NEIGHBORHOOD ~ SALE_PRICE + GROSS_SQUARE_FEET + TOTAL_UNITS, 
                         data = train_c, ntree = 100)
rf_preds <- predict(rf_class, newdata = test_c)

# Model 2: k-Nearest Neighbors (kNN)
train_x_scaled <- scale(train_c[, c("SALE_PRICE", "GROSS_SQUARE_FEET", "TOTAL_UNITS")])
test_x_scaled <- scale(test_c[, c("SALE_PRICE", "GROSS_SQUARE_FEET", "TOTAL_UNITS")],
                       center = attr(train_x_scaled, "scaled:center"),
                       scale = attr(train_x_scaled, "scaled:scale"))

knn_preds <- knn(train = train_x_scaled, test = test_x_scaled, cl = train_c$NEIGHBORHOOD, k = 5)

# Model 3: Decision Tree Classifier
dt_class <- rpart(NEIGHBORHOOD ~ SALE_PRICE + GROSS_SQUARE_FEET + TOTAL_UNITS, 
                  data = train_c, method = "class")
dt_preds <- predict(dt_class, newdata = test_c, type = "class")

# Evaluate Results
cat("\nRandom Forest Evaluation: \n")
print(confusionMatrix(rf_preds, test_c$NEIGHBORHOOD, mode = "prec_recall"))

cat("\nkNN Evaluation: \n")
print(confusionMatrix(knn_preds, test_c$NEIGHBORHOOD, mode = "prec_recall"))

cat("\nDecision Tree Evaluation: \n")
print(confusionMatrix(dt_preds, test_c$NEIGHBORHOOD, mode = "prec_recall"))

