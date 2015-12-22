# tutorial taken from http://www.knowbigdata.com/blog/predicting-income-level-analytics-casestudy-r
library(ggplot2)
library(gridExtra)
library(caret)
library(lattice)
library(plyr)
library(gbm)
library(parallel)

# download and Read the Data

trainFileName = "adult.data" 
testFileName = "adult.test" 

if (!file.exists (trainFileName)) 
  download.file (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
                 destfile = trainFileName) 

if (!file.exists (testFileName)) 
  download.file (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", 
                 destfile = testFileName) 

colNames = c ("age", "workclass", "fnlwgt", "education", "educationnum", "maritalstatus", "occupation", 
              "relationship", "race", "sex", "capitalgain", "capitalloss", "hoursperweek", "nativecountry", 
              "incomelevel") 

train = read.table(trainFileName, 
                   header = FALSE, 
                   sep = ",", 
                   strip.white = TRUE, 
                   col.names = colNames, 
                   na.strings = "?", 
                   stringsAsFactors = TRUE) 

# summary of the data
str(train)

# data cleaning
table(complete.cases (train))

# Summarize all data sets with NAs only 
summary (train)
summary(train$incomelevel)

# remove NAs
myCleanTrain = train [!is.na (train$workclass) & !is.na (train$occupation), ] 
myCleanTrain = myCleanTrain [!is.na (myCleanTrain$nativecountry), ] 

summary(myCleanTrain)

# remove 'fnlwgt' attribute because it has no imact on the income level
myCleanTrain$fnlwgt = NULL

# explore the age attribute
summary(myCleanTrain$age)
help("~")

boxplot (age ~ incomelevel, 
         data = myCleanTrain, 
         main = "Age distribution for different income levels", 
         xlab = "Income Levels", 
         ylab = "Age", 
         col = "salmon")

incomeBelow50K = (myCleanTrain$incomelevel == "<=50K") 
xlimit = c(min (myCleanTrain$age), max (myCleanTrain$age)) 
ylimit = c(0, 1600) 

hist1 = qplot (age, 
               data = myCleanTrain[incomeBelow50K,], 
               margins = TRUE, 
               binwidth = 2, 
               xlim = xlimit, 
               ylim = ylimit, 
               colour = incomelevel) 

hist2 = qplot (age, 
               data = myCleanTrain[!incomeBelow50K,], 
               margins = TRUE, 
               binwidth = 2, 
               xlim = xlimit, 
               ylim = ylimit, 
               colour = incomelevel) 
help("grid")
grid.arrange (hist1, hist2, nrow = 2)

# explore the years of education attribute
summary(myCleanTrain$educationnum)

boxplot (educationnum ~ incomelevel, 
         data = myCleanTrain, 
         main = "Years of Education distribution for different income levels", 
         xlab = "Income Levels", 
         ylab = "Years of Education", 
         col = "blue") 

# explore the capital gain and capital loss attributes
nearZeroVar (myCleanTrain[, c("capitalgain", "capitalloss")], saveMetrics = TRUE)

summary (myCleanTrain[ myCleanTrain$incomelevel == "<=50K",
                       c("capitalgain", "capitalloss")])

summary (myCleanTrain[ myCleanTrain$incomelevel == ">50K", 
                       c("capitalgain", "capitalloss")]) 

# explore the hours per week attribute
summary (myCleanTrain$hoursperweek)

boxplot (hoursperweek ~ incomelevel, 
         data = myCleanTrain, 
         main = "Hours Per Week distribution for different income levels", 
         xlab = "Income Levels",
         ylab = "Hours Per Week", 
         col = "salmon") 

nearZeroVar (myCleanTrain[, "hoursperweek"], saveMetrics = TRUE)

# explore the correlation between continuous attributes and show that they are independent
corMat = cor (myCleanTrain[, c("age", "educationnum", "capitalgain", "capitalloss", "hoursperweek")])
diag (corMat) = 0 #Remove self correlations 
corMat 

# explore the sex attribute
table (myCleanTrain[,c("sex", "incomelevel")])

# explore the workclass, occupation, maritalstatus, relationship and educaton attributes
qplot (incomelevel, data = myCleanTrain, fill = workclass) + facet_grid (. ~ workclass) 
qplot (incomelevel, data = myCleanTrain, fill = occupation) + facet_grid (. ~ occupation)
qplot (incomelevel, data = myCleanTrain, fill = maritalstatus) + facet_grid (. ~ maritalstatus)
qplot (incomelevel, data = myCleanTrain, fill = relationship) + facet_grid (. ~ relationship) 

# modify the levels to be ordinal 
myCleanTrain$education = ordered (myCleanTrain$education, levels (myCleanTrain$education) [c(14, 4:7, 1:3, 12, 15, 8:9, 16, 10, 13, 11)]) 

print (levels (myCleanTrain$education)) 

qplot (incomelevel, data = myCleanTrain, fill = education) + facet_grid (. ~ education)

# build the model using the boosting algorithm
set.seed (32323) 
trCtrl = trainControl (method = "cv", number = 10) 
boostFit = train (incomelevel ~ age + workclass + education + educationnum + maritalstatus + occupation + relationship + race + capitalgain + capitalloss + hoursperweek + nativecountry, trControl = trCtrl, method = "gbm", data = myCleanTrain, verbose = FALSE)

confusionMatrix (myCleanTrain$incomelevel, predict (boostFit, myCleanTrain))

# validate the model

# first read the test data
test = read.table(testFileName, 
                   header = FALSE, 
                   sep = ",", 
                   skip=1,
                   strip.white = TRUE, 
                   col.names = colNames, 
                   na.strings = "?", 
                   stringsAsFactors = TRUE) 

summary(test)

# remove NAs values
myCleanTest = test [!is.na (test$workclass) & !is.na (test$occupation), ] 
myCleanTest = myCleanTest [!is.na (myCleanTest$nativecountry), ] 

# remove the fnlwgt attribute
myCleanTest$fnlwgt = NULL

# modify the levels to be ordinal 
myCleanTest$education = ordered (myCleanTest$education, levels (myCleanTest$education) [c(14, 4:7, 1:3, 12, 15, 8:9, 16, 10, 13, 11)]) 

str(myCleanTrain)
str(myCleanTest)

# and finally check the model on the test data
myCleanTest$predicted = predict (boostFit, myCleanTest) 
confusionMatrix (myCleanTest$incomelevel, myCleanTest$predicted)
