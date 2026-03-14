##########################################
### Principal Component Analysis (PCA) ###
##########################################

## load libraries
library(ggplot2)
library(ggfortify)
library(GGally)
library(e1071)
library(class)
library(psych)
library(readr)
library(caret)

## set working directory so that files can be referenced without the full path
setwd("~/Data Analytics/lab4/")

## read dataset
wine <- read_csv("wine.data", col_names = FALSE)

## set column names
names(wine) <- c("Type","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid Phenols","Proanthocyanins","Color Intensity","Hue","Od280/od315 of diluted wines","Proline")

## change the data type of the "Type" column from character to factor
wine$Type <- as.factor(wine$Type)

## visualize variables
pairs.panels(wine[,-1],gap = 0,bg = c("red", "yellow", "blue")[wine$Type],pch=21)
ggpairs(wine, ggplot2::aes(colour = Type))

###
X <- wine[,-1]
Y <- wine$Type
###

# Compute the PCs and plot the dataset using the 1st and 2nd PCs
principal_components <- princomp(X, cor = TRUE)
summary(principal_components)

autoplot(principal_components, data = wine, colour = 'Type',
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3, scale = 0)

# Identify the variables that contribute the most to the 1st PC
pc1_loadings <- principal_components$loadings[,1]
top_contributors <- sort(abs(pc1_loadings), decreasing = TRUE)
print("Top contributors to PC1:")
print(top_contributors)


## SET UP TRAINING AND TESTING DATA
set.seed(123) # For reproducibility
train_index <- sample(1:nrow(wine), 0.7 * nrow(wine))

train_data <- wine[train_index, ]
test_data  <- wine[-train_index, ]

train_Y <- train_data$Type
test_Y  <- test_data$Type


# MODEL 1: Train kNN using a subset (4) of the original variables
subset_features <- c("Flavanoids", "Total phenols", "Od280/od315 of diluted wines", "Proanthocyanins")

train_X_subset <- train_data[, subset_features]
test_X_subset  <- test_data[, subset_features]

train_X_subset_scaled <- scale(train_X_subset)
test_X_subset_scaled <- scale(test_X_subset, 
                              center = attr(train_X_subset_scaled, "scaled:center"), 
                              scale = attr(train_X_subset_scaled, "scaled:scale"))

# Predict using kNN (k = 5)
knn_pred_subset <- knn(train = train_X_subset_scaled, 
                       test = test_X_subset_scaled, 
                       cl = train_Y, k = 5)


# MODEL 2: Train kNN using the data projected onto the first 2 PCs
# Create a dataframe of the PCA scores
pca_scores <- as.data.frame(principal_components$scores)

# Split the PCA scores using the exact same indices as before
train_X_pca <- pca_scores[train_index, c("Comp.1", "Comp.2")]
test_X_pca  <- pca_scores[-train_index, c("Comp.1", "Comp.2")]

# Predict using kNN on the PCs (k = 5)
knn_pred_pca <- knn(train = train_X_pca, 
                    test = test_X_pca, 
                    cl = train_Y, k = 5)


## Compare the 2 models
# Evaluate the Subset Model
print("Model 1: kNN on 4 Original Variables")
cm_subset <- confusionMatrix(data = knn_pred_subset, reference = test_Y, mode = "prec_recall")
print(cm_subset$table) # Contingency Table
print(cm_subset$byClass[, c("Precision", "Recall", "F1")])

# Evaluate the PCA Model
print("Model 2: kNN on First 2 Principal Components")
cm_pca <- confusionMatrix(data = knn_pred_pca, reference = test_Y, mode = "prec_recall")
print(cm_pca$table) # Contingency Table
print(cm_pca$byClass[, c("Precision", "Recall", "F1")])