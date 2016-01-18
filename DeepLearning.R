library(h2o)
library(mlbench)
library(caret)

# start a local H2O cluster
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

# load the breast cancer dataset from mlbench
# 
# Format of the data:
# A data frame with 699 observations on 11 variables, one being a character variable, 9 being ordered
# or nominal, and 1 target class.
# [,1]      Id                            Sample code number
# [,2]      Cl.thickness           Clump Thickness
# [,3]      Cell.size                 Uniformity of Cell Size
# [,4]      Cell.shape              Uniformity of Cell Shape
# [,5]      Marg.adhesion       Marginal Adhesion
# [,6]      Epith.c.size            Single Epithelial Cell Size
# [,7]      Bare.nuclei             Bare Nuclei
# [,8]      Bl.cromatin            Bland Chromatin
# [,9]      Normal.nucleoli     Normal Nucleoli
# [,10]     Mitoses                  Mitoses
# [,11]     Class                      Class
data(BreastCancer)

head(BreastCancer)
BreastCancer$Class

# remove the Id column because is not needed for classification
BreastCancer$Id <- NULL

# convert the data to a H2O data frame
pathToData = paste0(normalizePath("~/"), "bcData.csv")
write.table(x = BreastCancer, file = pathToData, row.names = F, col.names = T)
bcData = h2o.importFile(path = pathToData, destination_frame = "bcData")

# split the dataset 80/20 for training/test
set.seed(1111)
y_data = as.matrix(bcData$Class)
folds = createFolds(as.factor(y_data), k=9)
train_rows = as.integer(unlist(folds[1:8]))
test_rows = as.integer(unlist(folds[9:10]))
y_test = as.factor(y_data[test_rows])

# train a model
model = h2o.deeplearning(x = 1:9, # feature rows
                         y = 10, # label
                         training_frame = bcData[train_rows, ],
                         activation = "Tanh",
                         balance_classes = TRUE,
                         hidden = c(100, 100, 100), # hidden layers
                         epochs = 500)

y_predicted = h2o.predict(model, bcData[test_rows, ])$predict
y_predicted = as.factor(as.matrix(y_predicted))

confusionMatrix(y_predicted, y_test)
