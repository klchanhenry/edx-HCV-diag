#
# Code objectives and usage summary:-  
# 
# This is a set of R code for Liver Disease Diagnosis via HCV project. 
# 
# By running this code, you will 
# 1. install those library needed automatically. ( can possible takes several minutes )
# 2. download the analysis data from internet ( should be fast as it less than 1MBytes )
# 3. build different kind of machine learning model and measure the performance
# 4. some machine learning method is computational intensive, it will occupy most of
# you CPU time and you may feel your computer respond slowly at the period of time.
# ( likely take 10 to 20 minutes depend on different computer )
# 
# 
# Hardware reference:-
# The computer tested on this code is a Intel PC running 2 Cores 4 Threads CPU with 16GBytes
# of memory. The PC was manufactured at around 2015. If your platform is something similar,
# it should be reasonable good to run and having similar operation experience. 
# 
# IMPORTANT Pre-caution:-
# 1. This code will run computational intensive task. It is highly recommended to
# dedicate a computer running for this instead of running it while you are having 
# some other important task doing at the same time.
# 2. It is highly recommended to run this code on a development machine 
# instead of mission critical machine. By running this code, you accept the risk and
# any unexpected consequent. You are free to examine the code line by lines.
# 3. Finally, if you are not sure what this code will do, you are suggested to review
# and understand instead of rush for execute the code.
# 
#


#
# Obtaining data
#

# Loading various library needed upfront
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")

library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(doParallel)

library(corrplot)
library(RColorBrewer)

library(rpart.plot)
library(rpart)
library(rattle)
library(randomForest)

library(xgboost)

# Load the data set to variable from the UC Irvine Machine Learning Repository
hcv <- fread("https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv")

#
# In case you cannot access the url above due to any reason, you can download the hcvdat0.csv
# file from the github state in the FORWORD section and put it in your current working directory.
# you can check your working directory by getwd(). 
# Then you can uncomment the following line the it will load the data from the local file.
#hcv <- fread("hcvdat0.csv")



#
# Scrubbing data 
#

# Show the current data structure of the data set
str(hcv)
# Show a few row of the data set so that we have some idea of what the data look like.
head(hcv)

# Show the numbner of NA, Not Available, data under each attribute
sapply(hcv, function(x) sum(is.na(x)))

# removing the attribute V1 which only present the row number
hcv$V1 <- NULL

# change the Category and Sex attribute form char to factor
hcv <- as.data.frame(hcv) %>% 
  mutate(Category = as.factor(Category),
         Sex = as.factor(Sex))

# Show the structure of the data set after modification
str(hcv)

# locate the missing value and replace with the mean of respective attribute
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
hcv[] <- lapply(hcv, NA2mean)

# check again for any missing value
sapply(hcv, function(x) sum(is.na(x)))

# fix the seed so that process result related to random is repeatable
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# random create index of the data set into 7 to 3 ratio 
index = createDataPartition(hcv$Category, p = .7, list = F)

# Assign the training set and validation set in the ratio of 7 to 3
train_set = hcv[index, ]
validation_set = hcv[-index, ]




#
# Exploring Data 
#

# showing statistical summary of the training set
summary(train_set)

# create a boxplot chart for lab test attributes
boxplot(x = as.list(train_set[,c(-1,-2,-3)]),
        horizontal=TRUE,las=1,
        log = "x",
        main="Distribution Pattern of   \n Numerical Result Value in Different Laboratory Test Attributes",
        xlab= "Numerical value of test results in repect to each laboratory Test (log scale)")


#calculate correlation between lab test attribute and plot chart
corr_matrix <-cor(train_set[,c(-1,-2,-3)])
corrplot(corr_matrix,type="lower", order="hclust",
         method="color",outline = T,
         addCoef.col = "grey", number.digits = 2,
         number.cex= 7/ncol(corr_matrix),
         title="Correlation between Laboratory Test Attributes",
         mar=c(0,0,2,0))


#
# Modelling Data
#

# show the number of sample under different category in training set
table(train_set$Category)

# Creating artifical balanced data set by upSample
train_set <- upSample(x = train_set[, -1],y = train_set$Category, yname = "Category")

# show the number of sample under different category in training set after upSample
table(train_set$Category)



#
# Modelling Framework
#



# Decision Tree

#
# In this Modelling Framework section, there are computational intensive tasks.
# You are welcome to run the Rmd / R script with a non-critical machine since
# this process may take you around 20 minutes and using most of your machine
# CPU power during this period of time.
#

# As this machine learning method demand for more computing power
# We are creating a parallel socket cluster to utilize most of the CPU core
# One CPU core intentionally reserved in order to keep the system responsive
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

# fix the seed so that process result related to random is repeatable
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# Cross-Validated (10 folds, repeated 3 times)
control_dt <- trainControl(method="repeatedcv", number=10, repeats=3)

# create decision tree model
model_dt <- train(Category ~ . -Age -Sex ,
                  data=train_set,
                  method="rpart",
                  trControl=control_dt,
                  tuneLength=5)

# Stop the parallel socket cluster after use
stopCluster(cl)

# Show the decision tree model built
print(model_dt)

# plot the decision tree
fancyRpartPlot(model_dt$finalModel, sub="Decision Tree model")



# Learning Vector Quantization (LVQ)

# As this machine learning method demand for more computing power
# We are creating a parallel socket cluster to utilize most of the CPU core
# One CPU core intentionally reserved in order to keep the system responsive
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

# fix the seed so that process result related to random is repeatable
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# Cross-Validated (10 folds, repeated 3 times)
control_lvq <- trainControl(method="repeatedcv", number=10, repeats=3)

# create Learning Vector Quantization (LVQ) model
model_lvq <- train(Category ~ . -Age -Sex ,
                   data=train_set,
                   method="lvq",
                   trControl=control_lvq,
                   tuneLength=5)

# Stop the parallel socket cluster after use
stopCluster(cl)

# Show the LVQ model built
print(model_lvq)



# Random Forest

# As this machine learning method demand for more computing power
# We are creating a parallel socket cluster to utilize most of the CPU core
# One CPU core intentionally reserved in order to keep the system responsive
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

# fix the seed so that process result related to random is repeatable
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# Cross-Validated (10 folds, repeated 3 times)
control_rf <- trainControl(method="repeatedcv", number=10, repeats=3)

# create random forest model
model_rf <- train(Category ~ . -Age -Sex ,
                  data=train_set,
                  method="rf",
                  trControl=control_rf,
                  tuneLength=5)

# Stop the parallel socket cluster after use
stopCluster(cl)

# Show the random forest model built
print(model_rf)


# xgBoost model

# As this machine learning method demand for more computing power
# We are creating a parallel socket cluster to utilize most of the CPU core
# One CPU core intentionally reserved in order to keep the system responsive
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

# fix the seed so that process result related to random is repeatable
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# Cross-Validated (10 folds, repeated 3 times)
control_xgb <- trainControl(method="repeatedcv", number=10, repeats=3)

# create xgBoost model
model_xgb <- train(Category ~ . -Age -Sex ,
                   data=train_set, 
                   method="xgbTree",
                   tuneLength=5,
                   trControl=control_xgb)

# Stop the parallel socket cluster after use
stopCluster(cl)

# Show the xgBoost model built
options(max.print = 100) 
print(model_xgb)
options(max.print = 1000) 


#
# RESULT
#

# using the decision tree model we built previously and challenge with the validation set
predict_dt <- predict(model_dt, validation_set, type = "raw")
# formulate confusion matrix for performance review 
cm_dt <- confusionMatrix(predict_dt, validation_set$Category)

# using the LVQ model we built previously and challenge with the validation set
predict_lvq <- predict(model_lvq, validation_set, type = "raw")
# formulate confusion matrix for performance review 
cm_lvq <- confusionMatrix(predict_lvq, validation_set$Category)

# using the random forest model we built previously and challenge with the validation set
predict_rf <- predict(model_rf, validation_set, type = "raw")
# formulate confusion matrix for performance review 
cm_rf <- confusionMatrix(predict_rf, validation_set$Category)

# using the xgboost model we built previously and challenge with the validation set
predict_xgb <- predict(model_xgb, validation_set, type = "raw")
# formulate confusion matrix for performance review 
cm_xgb <- confusionMatrix(predict_xgb, validation_set$Category)

#confusion matrix table for decision tree model
print(cm_dt$table)
print(cm_dt$overall)

#confusion matrix table for LVQ model
print(cm_lvq$table)
print(cm_lvq$overall)

#confusion matrix table for random forest model
print(cm_rf$table)
print(cm_rf$overall)

#confusion matrix table for xgboost model
print(cm_xgb$table)
print(cm_xgb$overall)




