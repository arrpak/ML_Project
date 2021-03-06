# install packages
if(!require(readr)) install.packages("readr")
if(!require(recommenderlab)) installed.packages("recommenderlab")
if(!require(reshape2)) install.packages("reshape2")

data.germany <- read_delim("~/Documents/R_Project/ML_Project/CF_Project/lastfm-matrix-germany.csv", delim = ",", n_max = -1, na = "NA")
# Drop any column named "user"
data.germany.ibs <- data.germany[, !(names(data.germany) %in% c("user"))]

### item based recommendations
# Create a similarity function to calculate the cosine between two vectors (Collborative filtering algorithm)
getCosine <- function(x, y){
  this.cosine <- sum(x*y)/(sqrt(sum(x*x)))*(sqrt(sum(y*y)))
  return(this.cosine)
}

# Create a placeholder dataframe listing item vs. item (ctreat matrxi for output the item similarity number)
data.germany.ibs.similarity  <- matrix(NA, nrow = ncol(data.germany.ibs),
                                       ncol = ncol(data.germany.ibs),
                                       dimnames = list(colnames(data.germany.ibs),colnames(data.germany.ibs)))

# Lets fill in those empty spaces with cosine similarities
# Loop through the columns
for(i in 1:ncol(data.germany.ibs)) {
  # Loop through the columns for each column
  for(j in 1:ncol(data.germany.ibs)) {
    # Fill in placeholder with cosine similarities
    data.germany.ibs.similarity[i,j] <- getCosine(as.matrix(data.germany.ibs[i]), as.matrix(data.germany.ibs[j]))
  }
}

# Back to dataframe
data.germany.ibs.similarity <- as.data.frame(data.germany.ibs.similarity)

# Get the top 10 neighbours for each
nn <- 10
data.germany.neighbours <- matrix(NA, nrow = ncol(data.germany.ibs.similarity),
                                  ncol = nn+1,
                                  dimnames = list(colnames(data.germany.ibs.similarity)))
# find the Nerest neighbours
for(i in 1:ncol(data.germany.ibs)) 
{
  data.germany.neighbours[i,] <- (t(head(n = nn+1,
                                         rownames(data.germany.ibs.similarity[order(data.germany.ibs.similarity[, i], decreasing = TRUE), ][i]))))
}

### user based recommendations
# Lets make a helper function to calculate the scores
getScore <- function(history, similarities)
{
  x <- sum(history*similarities)/sum(similarities)
  return(x)
}

# A placeholder matrix
holder <- matrix(NA, nrow = nrow(data.germany),
                 ncol = ncol(data.germany)-1,
                 dimnames = list((data.germany$user), colnames(data.germany[-1])))
nn <- 10

# Loop through the users (rows)
for(i in 1:nrow(holder)) {
  # Loops through the products (columns)
  for(j in 1:ncol(holder)) {
    # Get the user's name and th product's name
    # We do this not to conform with vectors sorted differently 
    user <- rownames(holder)[i]
    product <- colnames(holder)[j]
    
    # We do not want to recommend products you have already consumed
    # If you have already consumed it, we store an empty string
    if(as.integer(data.germany[data.germany$user==user, product]) == 1)
    { 
      holder[i,j]<-"" # if value already had, we skip
    } else {
      # We first have to get a product's top 10 neighbours sorted by similarity
      topN <- ((head(n= nn+1,
                     (data.germany.ibs.similarity[order(data.germany.ibs.similarity[, product], decreasing = TRUE), ][product]))))
      topN.names <- as.character(rownames(topN))
      topN.similarities <- as.numeric(topN[, 1])
      
      # Drop the first one because it will always be the same song
      topN.similarities <- topN.similarities[-1]
      topN.names <- topN.names[-1]
      
      # We then get the user's purchase history for those 10 items
      topN.purchases <- data.germany[, c("user", topN.names)]
      topN.userPurchases <- topN.purchases[topN.purchases$user==user, ]
      topN.userPurchases <- as.numeric(topN.userPurchases[!(names(topN.userPurchases) %in% c("user"))])
      
      # We then calculate the score for that product and that user
      holder[i, j] <- getScore(similarities = topN.similarities, history = topN.userPurchases)
      
    } # close else statement
  } # end product for loop   
} # end user for loop

data.germany.user.scores <- holder

# Lets make our recommendations pretty
data.germany.user.scores.holder <- matrix(NA, 
                                          nrow = nrow(data.germany.user.scores),
                                          ncol = 100,
                                          dimnames = list(rownames(data.germany.user.scores)))
for(i in 1:nrow(data.germany.user.scores)) 
{
  data.germany.user.scores.holder[i,] <- names(head(n = 100,
                                                    (data.germany.user.scores[, order(data.germany.user.scores[i, ] ,decreasing = TRUE)])[i,]))
}
