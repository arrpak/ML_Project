library("recommenderlab")

### create a small artificial data set
m <- matrix(sample(c(as.numeric(0:5), NA), 50, replace = TRUE, prob = c(rep(0.4/6, 6), 0.6)), 
            ncol = 10, dimnames = list(user = paste("u", 1:5, sep = ""), item = paste("i", 1:10, sep = "")))

## set data set convert into realRatingMatrix object
r <- as(m, "realRatingMatrix")

## check the r as same as the original matrix m
identical(as(r, "matrix"), m)
head(as(r, "data.frame")) # list or data.frame or matrix

## remove rating bias using normalization
r_m <- ？(r)
r_m

## plot for samll portions data
image(r, main = "Raw Ratings")
image(r_m, main = "Normalied Ratings")

## binarization of daat
r_b <- binarize(r , minRating = 1) # minRating : choose which rating want to convert into 1

## using Jester5k data set coms with recommenderlab
data(Jester5k)
r <- sample(Jester5k, 1000)

# inspection of data set properties
rowCounts(r[1, ]) # colCounts
rowMeans(r[1, ]) # colMeans
rowSums(r[1, ]) # rowSums
rowSds(r[1, ]) # rowSds

# we could understand the distribution of ratings, is positive more or negative more 
hist(getRatings(r), breaks = 100) # getRatings : extracts ratings from data
hist(getRatings(normalize(r)), breaks = 100)
hist(getRatings(normalize(r, method = "Z-score")), breaks = 100)

# show how many users with ratings and who have rated all joks. show average ratings per joks 
hist(rowCounts(r), breaks = 50)
hist(colMeans(r), breaks = 20)

## create a recommender
recommenderRegistry$get_entry_names() # look at registry
recommenderRegistry$get_entries(dataType = "realRatingMatrix") # show each methods in registry

## predict topNList
rec <- Recommender(Jester5k[1:1000], method = "POPULAR")
names(getModel(rec))
recom <- predict(rec, Jester5k[1001:1005], n = 5)
as(recom, "list")

## predict ratings
recom <- predict(rec, Jester5k[1001:1005], n = 5, type = "ratings")
as(recom, "matrix")

### Evaluation of predicted ratings 
# scheme, given : set number for withhold items
# (we should make sure every users rating more than given number before set given) 
e <- evaluationScheme(Jester5k[1:1000], method = "split", train = 0.9, given = 15, goodRating = 5)
e
# model
rec1 <- Recommender(getData(e, "train"), method = "UBCF", param = list(nn = 50, method = "Jaccard")) 
rec1
rec2 <- Recommender(getData(e, "train"), method = "IBCF", param = list(method = "Jaccard")) 
rec2

# predict, known is all items except withhold items, unknow is withhold itmes 
recom1 <- predict(rec1, getData(e, "known"), type = "ratings")
as(recom1, "matrix")[1:5, 1:20]
recom2 <- predict(rec2, getData(e, "known"), type = "ratings")
as(recom2, "matrix")
# compare
error <- rbind(calcPredictionAccuracy(recom1, getData(e, "unknow")), calcPredictionAccuracy(recom2, getData(e, "unknow")))
rownames(error) <- c("UBCF", "IBCF")
error

### Evaluation of a top-N recommender algorithm
scheme <- evaluationScheme(Jester5k[1:1000], method = "cross", k = 10, given = 5, goodRating = 5)
scheme
results <- evaluate(scheme, method = "POPULAR", n = c(1, 3, 5, 10, 15, 20))
results
getConfusionMatrix(results)
avg(results)

plot(results, annotate = TRUE)
plot(results, "prec/rec", annotate = TRUE)

### Comparing recommender algorithms
## realRatingMartix
scheme <- evaluationScheme(Jester5k[1:1000], method = "split", train = 0.9, k = 1, given = 20, goodRating = 5)
scheme
algorithms <- list("random itmes" = list(name = "RANDOM", param = NULL),
                   "popular itmes" = list(name = "POPULAR", param = NULL),
                   "user-based CF" = list(name = "UBCF", param = list(method = "Cosine", nn = 50, minRating = 5)))

results <- evaluate(scheme, algorithms, n = c(1, 3, 5, 10, 15, 20)) 
names(results) # results[["user-based CF"]]
plot(results, annotate = c(1, 2, 3), legend = "topleft")
plot(results, "prec/rec", annotate = 3)

## binaryRatingMatrix
Jester_binary <- binarize(Jester5k, minRating = 5)
Jester_binary <- Jester_binary[rowCounts(Jester_binary)>20] 
scheme_binary <- evaluationScheme(Jester_binary[1:1000], method = "split", train = 0.9, k = 1, given = 20)

algorithms_binary <- list("random itmes" = list(name = "RANDOM", param = NULL),
                          "popular itmes" = list(name = "POPULAR", param = NULL),
                          "user-based CF" = list(name = "UBCF", param = list(method = "Jaccard", nn = 500)))
results_binary <- evaluate(scheme_binary, algorithms_binary, n = c(1, 3, 5, 10, 15, 20))

plot(results_binary, annotate = c(1, 2, 3), legend = "bottomright")
