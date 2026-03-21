library(readr)
library(e1071)
library(randomForest)

setwd("~/Data Analytics/lab5/")
wine <- read_csv("wine.data", col_names = FALSE)

head(wine)

# set column names
names(wine) <- c("Type","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium",
                 "Total phenols","Flavanoids","Nonflavanoid Phenols","Proanthocyanins",
                 "Color Intensity","Hue","Od280/od315 of diluted wines","Proline")

# change the data type of the "Type" column from character to factor
wine$Type <- as.factor(wine$Type)

# subset
subset_features <- c("Type", "Flavanoids", "Total phenols", "Od280/od315 of diluted wines", "Proanthocyanins")
wine_subset <- wine[, subset_features]
names(wine_subset) <- make.names(names(wine_subset))

# split train/test
N <- nrow(wine_subset)
set.seed(123)
train.indexes <- sample(N, 0.7 * N)

train <- wine_subset[train.indexes,]
test <- wine_subset[-train.indexes,]


## train SVM model
tune.linear <- tune.svm(Type ~ ., data = train, kernel = 'linear', cost = c(0.01, 0.1, 1, 10, 100))
best.cost.linear <- tune.linear$best.parameters$cost

# Train the best model
svm.mod0 <- svm(Type ~ ., data = train, kernel = 'linear', cost = best.cost.linear)

# predict for test data
test.pred.0 <- predict(svm.mod0, test)

# confusion matrix and metric calculation
cm0 = as.matrix(table(Actual = test$Type, Predicted = test.pred.0))

n = sum(cm0) # number of instances
diagv = diag(cm0) # number of correctly classified instances per class 
rowsums = apply(cm0, 1, sum) # number of instances per class
colsums = apply(cm0, 2, sum) # number of predictions per class

accuracy <- sum(diagv)/n
recall = diagv / rowsums 
precision = diagv / colsums
f1 = 2 * precision * recall / (precision + recall) 

svm.mod0.res <- data.frame(model="linear", precision, recall, f1)
results <- svm.mod0.res


## train SVM model
tune.radial <- tune.svm(Type ~ ., data = train, kernel = 'radial', 
                        cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1))
best.cost.radial <- tune.radial$best.parameters$cost
best.gamma.radial <- tune.radial$best.parameters$gamma

# Train the best model
svm.mod1 <- svm(Type ~ ., data = train, kernel = 'radial', 
                cost = best.cost.radial, gamma = best.gamma.radial)

# predict for test data
test.pred.1 <- predict(svm.mod1, test)

# confusion matrix and metric calculation
cm1 = as.matrix(table(Actual = test$Type, Predicted = test.pred.1))

n = sum(cm1) 
diagv = diag(cm1) 
rowsums = apply(cm1, 1, sum) 
colsums = apply(cm1, 2, sum) 

accuracy <- sum(diagv)/n
recall = diagv / rowsums 
precision = diagv / colsums
f1 = 2 * precision * recall / (precision + recall) 

svm.mod1.res <- data.frame(model="radial", precision, recall, f1)
results <- rbind(results, svm.mod1.res)


## Random Forest
# Train model using 100 trees
rf.mod <- randomForest(Type ~ ., data = train, ntree = 100)

# predict for test data
test.pred.rf <- predict(rf.mod, test)

# confusion matrix and metric calculation
cm.rf = as.matrix(table(Actual = test$Type, Predicted = test.pred.rf))

n = sum(cm.rf) 
diagv = diag(cm.rf) 
rowsums = apply(cm.rf, 1, sum) 
colsums = apply(cm.rf, 2, sum) 

accuracy <- sum(diagv)/n
recall = diagv / rowsums 
precision = diagv / colsums
f1 = 2 * precision * recall / (precision + recall) 

rf.res <- data.frame(model="random forest", precision, recall, f1)
results <- rbind(results, rf.res)

results
