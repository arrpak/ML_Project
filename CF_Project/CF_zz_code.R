# install packages
if(!require(readr)) install.packages("readr")
if(!require(recommenderlab)) install.packages("recommenderlab")
if(!require(reshape2)) install.packages("reshape2")

# input data
u.data <- read_delim("~/Documents/R_Project/ML_Project/CF_Project/locked.data", delim = ",", n_max = -1, na = "NA", 
                     col_names = c("user_id", "movie_id", "rating", "timestamp"))
u.data$user_id <- as.factor(u.data$user_id)
u.data$movie_id <- as.factor(u.data$movie_id)

# convert to Sparsity matrix
u.data <- u.data[, -which(colnames(u.data) == 'timestamp')]
g <- acast(u.data, user_id ~ movie_id)

# Check the class of g
class(g)

# Remove NA value if you have
R <- dropNA(g)

# Convert R into "realRatingMatrix" or "binaryRatingMatrix" data structure
# binaryRatingMatrix and realRatingMatrix is a recommenderlab sparse-matrix like data-structure
r <- as(g, "binaryRatingMatrix")

# inspection of data set properties
rowCounts(r) # colCounts
rowMeans(r) # colMeans
rowSums(r) # rowSums
rowSds(r) # rowSds

# show how many video have seen by users and how many people saw each video 
hist(rowCounts(r), breaks = 50)
hist(colCounts(r), breaks = 30, xlim = c(0, 20))

# view r in other possible ways
#as(r, "list")     # A list
#as(r, "matrix")   # A sparse matrx

# normalize the rating matrix if your data is realRatingMatrix
#r_m <- normalize(r)

## Create a recommender object (model)
# They pertain to four different algorithms.
# UBCF: User-based collaborative filtering
# IBCF: Item-based collaborative filtering
# Parameter 'method' decides similarity measure
# Cosine or Jaccard
# minRating : user specified threshold, In the following only items with a rating of 4 or 
#             higher will become a positive rating in the new binary rating matrix
# nn:NearestNeighbors

### Evaluation of a top-N recommender algorithm
## There have three method to split data : 1. split 2. cross-validation 3. boostrap
# scheme, given : set number for withhold items
# (we should make sure every users rating more than given number before set given) 
scheme <- evaluationScheme(r[which(rowCounts(r) > 10)], method = "split", train = 0.9, given = 5)
scheme

## multiple evaluate model with each method
algorithms_binary <- list("randomItmes" = list(name = "RANDOM", param = NULL),
                          "popularItmes" = list(name = "POPULAR", param = NULL),
                          "userbasedCF" = list(name = "UBCF", param = list(method = "Jaccard", nn = 50)))

## n:predict top N value
results <- evaluate(scheme, algorithms_binary, n = c(1, 3, 5, 10, 20, 30)) 
names(results) # results[["user-based CF"]]
plot(results, annotate = c(1, 2, 3), legend = "topleft")

## evaluate model with each method one by one
rec1 <- Recommender(getData(scheme, "train"), method = "POPULAR") 
rec1
rec2 <- Recommender(getData(scheme, "train"), method = "UBCF", param = list(nn = 50, method = "Jaccard")) 
rec2

# predict, known is all items except withhold items, unknow is withhold itmes 
recom1 <- predict(rec1, getData(scheme, "known"), type = "topNList", n = 10)
as(recom1, "matrix")[1:5, 1:20]
recom2 <- predict(rec2, getData(scheme, "known"), type = "topNList", n = 10)
as(recom2, "list")

# compare result
error <- rbind(calcPredictionAccuracy(recom1, getData(scheme, "unknow"), given = 10), 
               calcPredictionAccuracy(recom2, getData(scheme, "unknow"), given = 10))
rownames(error) <- c("POPULAR", "UBCF")
error

# F-measure(F1 score) where an F1 score reaches its best value at 1 and worst score at 0.
F_measure <- rbind(2/(1/error[1, "precision"] + 1/error[1, "recall"]), 
                   2/(1/error[2, "precision"] + 1/error[2, "recall"]))
rownames(F_measure) <- c("POPULAR", "UBCF")
F_measure

## similarity matrix 
similarity(r[10:20, ], r[1:10, ], method = "Jaccard")

