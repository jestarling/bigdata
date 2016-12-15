### SDS 385 - Exercises 04 - Part B
#This code implements stochastic gradient descent
#using adagrad.  Reads in the malicious URL data.

#Jennifer Starling
#18 September 2016

rm(list=ls())	#Cleans workspace.

library(microbenchmark)
library(Rcpp)
library(RcppEigen)
library(Matrix)
library(permute)

#Load C++ file.
sourceCpp(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 R Code/stoch_grad_desc_logit_binom_Eigen.cpp')

#------------------------------------------------------------------
#Read in small data set.

#Read in code.
wdbc = read.csv('/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Course Data/wdbc.csv', header=FALSE)
y = wdbc[,2]

#Convert y values to 1/0's.
Y = rep(0,length(y)); Y[y=='M']=1
X = as.matrix(wdbc[,-c(1,2)])

#Select features to keep, and scale features.
scrub = which(1:ncol(X) %% 3 == 0)
scrub = 11:30
X = X[,-scrub]
X <- scale(X) #Normalize design matrix features.
X = cbind(rep(1,nrow(X)),X)
X = Matrix(X,sparse=T)

Xt = t(X)

#Set up vector of sample sizes.  (All 1 for wdbc data.)
m <- rep(1,nrow(X))	

#------------------------------------------------------------------
#Read in large sparse URL data set.

#READ IN DATA:
Xt_URL <- readRDS(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 R Code/url_Xt.rds')
Y_URL <- readRDS(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 R Code/url_y.rds')

#Set up vector of sample sizes.  (All 1 for wdbc data.)
m_URL <- rep(1,ncol(Xt_URL))

#------------------------------------------------------------------
#Run eigen code on small data set, with and without a penalty.

#Lambda=0 (no penalty)
output_eigen_small <- sparse_sgd_logit(Xt,Y,m,step=5,beta0=rep(0,11),lambda=0,npass=10000)
output_eigen_small$beta_hat

#Lambda=.1 (with penalty - notice the predictors shrink towards 0)
output_eigen_small <- sparse_sgd_logit(Xt,Y,m,step=5,beta0=rep(0,11),lambda=.1,npass=10000)
output_eigen_small$beta_hat

#------------------------------------------------------------------
#Run eigen code on large sparse URL data set, with and without penalty.
#Just one pass through data set.

#With lambda=0.
output_eigen_url <- sparse_sgd_logit(Xt_URL,Y_URL,m_URL,step=1,beta0=rep(0,nrow(Xt_URL)),lambda=0,npass=1)
head(output_eigen_url$beta_hat)
range(output_eigen_url$beta_hat)
length(output_eigen_url$beta_hat)

#With lambda=.1.
output_eigen_url <- sparse_sgd_logit(Xt_URL,Y_URL,m_URL,step=1,beta0=rep(0,nrow(Xt_URL)),lambda=.1,npass=1)
head(output_eigen_url$beta_hat)
range(output_eigen_url$beta_hat)
length(output_eigen_url$beta_hat)

#Microbenchmark.  Note: was 10.7 seconds before adding lazy recursion term, 13.3 seconds after adding.
microbenchmark(
	sparse_sgd_logit(Xt_URL,Y_URL,m_URL,step=1,beta0=rep(0,ncol(Xt_URL)),lambda=0,npass=1),
	times=1,
	unit='s'
)

###################################
###  Test/train prediction:     ###
###################################

#------------------------------------------------------------------
#Set up training and test data.

#Split URL data set into 75% train, 25% test.
train_samps = sample(ncol(Xt_URL)*.75,replace=F)

Xt_URL_tr = Xt_URL[,train_samps]
Xt_URL_te = Xt_URL[,-train_samps]

Y_URL_tr = Y_URL[train_samps]
Y_URL_te = Y_URL[-train_samps]

m_URL_tr = m_URL[train_samps]
m_URL_te = m_URL[-train_samps]

X_URL_te = t(Xt_URL_te)	#Need un-transposed test X for multiplying by beta when predicting.

#Build model on the training data set.  Try lambda=.1
url_tr <- sparse_sgd_logit(Xt_URL_tr,Y_URL_tr,m_URL_tr,step=1,beta0=rep(0,nrow(Xt_URL)),lambda=.1,npass=1)
head(url_tr$beta_hat)
range(url_tr$beta_hat)
length(url_tr$beta_hat)

#Predicted y values for test set.
XB = X_URL_te %*% url_tr$beta_hat
pred_probs = exp(XB) / (1 + exp(XB))
yhat = ifelse(pred_probs>.5,1,0)

#Calculate classification error rate.  (If prob > .5, yhat=1, else yhat=0.)
table(yhat==Y_URL_te)/length(Y_URL_te)

#Sample Results from one of my runs:
#> table(yhat==Y_URL_te)/length(Y_URL_te)
#
#     FALSE       TRUE 
#     0.05525405 0.94474595

#Plot log-likelihood function for the training data.
plot(1:length(url_tr$loglik),url_tr$loglik,type='l',col='blue',
	main='Running Avg Neg Loglhood for Training URL Model',
	xlab='Iteration',ylab='RA Neg Loglhood')
	
#Save plot:
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 LaTeX Files/URL_RAlogl_test.jpg')	
 plot(1:length(url_tr$loglik),url_tr$loglik,type='l',col='blue',
	main='Running Avg Neg Loglhood for Training URL Model',
	xlab='Iteration',ylab='RA Neg Loglhood')
dev.off()
