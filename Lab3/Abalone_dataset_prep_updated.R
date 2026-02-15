################################################
#### Evaluating Classification & CLustering ####
################################################

library("caret")
library(GGally)
library(psych)
library(class)
library(cluster)

## read data
abalone <- read.csv("~/Data Analytics/lab3/abalone.data", header=FALSE)

## rename columns
colnames(abalone) <- c("sex", "length", 'diameter', 'height', 'whole_weight', 'shucked_wieght', 'viscera_wieght', 'shell_weight', 'rings' ) 

## derive age group based in number of rings
abalone$age.group <- cut(abalone$rings, br=c(0,8,11,35), labels = c("young", 'adult', 'old'))

## take copy removing sex and rings
abalone.sub <- abalone[,c(2:8,10)]

## convert class labels to strings
abalone.sub$age.group <- as.character(abalone.sub$age.group)

## convert back to factor
abalone.sub$age.group <- as.factor(abalone.sub$age.group)

## split train/test
train.indexes <- sample(4177,0.7*4177)

train <- abalone.sub[train.indexes,]
test <- abalone.sub[-train.indexes,]

## separate x (features) & y (class labels)
X <- train[,1:7] 
Y <- train[,8]

set.seed(123)

# EXERCISE 1

# Train and evaluate 2 kNN models
# Model A: Using physical dimensions (Length, Diameter, Height)
# Model B: Using weights (Whole, Shucked, Viscera, Shell)

# Model A
knn_a <- knn(train = train[, 1:3], test = test[, 1:3], cl = train$age.group, k = 5)

# Model B
knn_b <- knn(train = train[, 4:7], test = test[, 4:7], cl = train$age.group, k = 5)

# Compare models using contingency tables
cat("Model A (Dimensions) Confusion Matrix:\n")
table_A <- table(Predicted = knn_a, Actual = test$age.group)
print(table_A)

cat("\nModel B (Weights) Confusion Matrix:\n")
table_B <- table(Predicted = knn_b, Actual = test$age.group)
print(table_B)

# Find optimal K
accuracy_list <- c()
k_values <- seq(1, 50, by = 2)

for (k in k_values) {
  pred <- knn(train = train[, 4:7], test = test[, 4:7], cl = train$age.group, k = k)
  acc <- sum(pred == test$age.group) / nrow(test)
  accuracy_list <- c(accuracy_list, acc)
}

# Plot Accuracy vs K
plot(k_values, accuracy_list, type="b", pch=19, col="blue", 
     xlab="Number of Neighbors (k)", ylab="Accuracy", main="Optimization of K")

opt_k <- k_values[which.max(accuracy_list)]
print(paste("The optimal value for K is:", opt_k))


# EXERCISE 2

# Prepare the data
cluster_data <- scale(abalone.sub[, 4:7]) 

set.seed(123)

# Find optimal K for K-means (Elbow Method)
wss <- sapply(1:10, function(k){kmeans(cluster_data, centers=k, nstart=20)$tot.withinss})
plot(1:10, wss, type="b", pch=19, main="Elbow Method for K-means", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Train Models
km_res  <- kmeans(cluster_data, centers = 3, nstart = 25)
pam_res <- pam(cluster_data, k = 3)

# Silhouette Plots
par(mfrow=c(1,2)) # Side-by-side plots

sil_km <- silhouette(km_res$cluster, dist(cluster_data))
plot(sil_km, main="K-means Silhouette", col="red", border=NA)

sil_pam <- silhouette(pam_res$clustering, dist(cluster_data))
plot(sil_pam, main="PAM Silhouette", col="green", border=NA)

## EOF ##
