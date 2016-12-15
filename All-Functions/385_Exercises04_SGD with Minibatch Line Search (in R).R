### SDS 385 - Exercises 04 - Part 
#This code implements stochastic gradient descent with mini-batch 
#backtracking line search to estimate the beta coefficients 
#for binomial logistic regression.

#Jennifer Starling
#17 Sept 2016

### SDS 385 - Exercises 02 - Part C 
#This code implements stochastic gradient descent to estimate the 
#beta coefficients for binomial logistic regression.

#Jennifer Starling
#30 August 2016

rm(list=ls())	#Cleans workspace.
library(microbenchmark)
library(permute)

#PART C:

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

#Set up vector of sample sizes.  (All 1 for wdbc data.)
m <- rep(1,nrow(X))	

#------------------------------------------------------------------
#Binomial Negative Loglikelihood function. 
	#Inputs: Design matrix X, vector of 1/0 vals Y, 
	#   coefficient matrix beta, sample size vector m.
	#Output: Returns value of negative log-likelihood 
	#   function for binomial logistic regression.
logl <- function(X,Y,beta,m){
	w <- 1 / (1 + exp(-X %*% beta))	#Calculate probabilities vector w_i.
	logl <- - sum(Y*log(w+1E-4) + (m-Y)*log(1-w+1E-4)) #Calculate log-likelihood.
		#Adding constant to resolve issues with probabilities near 0 or 1.	
	return(logl)	
}

#------------------------------------------------------------------
#Gradient Function: 
	#Inputs: Vector X (One row of design matrix), vector of 1/0 vals Y, 
	#   coefficient matrix beta, sample size vector m.
	#Output: Returns value of gradient function for binomial 
	#   logistic regression.

gradient <- function(X,Y,beta,m){
	w <- 1 / (1 + exp(-X %*% beta))	#Calculate probabilities vector w_i.
	gradient <- crossprod(X,as.numeric(m*w-Y)) #Calculate the gradient.
	
	return(gradient)
}

#------------------------------------------------------------------
#Line Search Function
	#Inputs:  X = design matrix
	#		  Y = vector of 1/0 response values
	#		  b = vector of betas
	# 		  g = gradient for beta vector
	#		  p = direction vector 
	#         m = sample size vector m
	#  	      maxalpha = The maximum allowed step size.
	#Outputs: alpha = The multiple of the search direction.

linesearch <- function(X,Y,b,gr,p,m,maxalpha=1){
	c <- .01			#A constant, in (0,1)
	alpha <- maxalpha	#The max step size, ie the starting step size.
	rho <- .5				#The multiplier for the step size at each iteration.
	
	while( (logl(X,Y,b + alpha*p,m)) > logl(X,Y,b,m) + c*alpha* (t(gr) %*% p) ) {
		alpha <- rho*alpha
	}
	
	return(alpha)
}

#------------------------------------------------------------------
#Mini-Batch Line Search Function
	#Inputs:  t= Current row of single stochastic observation.
	#		  X = full design matrix
	#		  Y = vector of 1/0 response values
	#		  b = vector of betas
	#         m = sample size vector m
	#  	      maxalpha = The maximum allowed step size.
	#	      batchsize = Number of obs in each mini-batch.
	#Outputs: alpha = The multiple of the search direction.
	#Dependencies: Calls the linesearch function.
	
#Sample a mini-batch of obs for line search.
linesearch_mb <- function(t,X,Y,b,m,maxalpha=1,batchsize=nrow(X)){
	
	Xnew <- matrix(X[t,,drop=F],nrow=1,byrow=T)
		
	#Draw a sample of size batchsize from the rows of X.
	mb_idx <- sample(1:nrow(X),batchsize,replace=F)
		
	#Calculate gradient for each individual obs in minibatch.
	temp_grads <- matrix(0,nrow=batchsize,ncol=ncol(X))
			
	for (j in 1:length(mb_idx)){
		temp_grads[j,] <- gradient(X[mb_idx[j],,drop=F],Y[mb_idx[j]],b,m[mb_idx[j]])
	}
			
	#Calculate average gradient.
	grad_avg <- colMeans(temp_grads)
			
	#Calculate updated backtracking line search step size using avg gradient for minibatch.
	alpha <- linesearch(Xnew,Y[t],b=b,gr=grad_avg,p=-grad_avg,m[t],maxalpha=1)	
	return(alpha)
}	

#------------------------------------------------------------------
#Stochastic Gradient Descent Algorithm:

stoch_gradient_descent_mbls <- function(X,Y,m,maxiter=50000,conv=1E-10,batchsize=1){

	#Set up random iterations through data, up to maxiter.
	npermutes <- ceiling(maxiter/nrow(X))
	obs_order <- as.vector(t(shuffleSet(1:nrow(X),nset=npermutes)))
	
	#Initialize matrix to hold gradients for each iteration.					
	grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 		

	#Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter+1,ncol=ncol(X)) 	

	#Initialize vector to hold full loglikelihood fctn for each iter.
	loglik <- rep(0,maxiter)	
	#Initialize vector to hold loglikelihood for each indiv t obs.	
	loglik_t <- rep(0,maxiter)	
	#Initialize vector to hold running avg for logl for t's.	
	loglik_ra <- rep(0,maxiter)		

	#Initialize values:
	i=1
	t  <- obs_order[i]
	Xnew <- matrix(X[t,,drop=F],nrow=1,byrow=T)
	
	loglik_t[i] <- logl(Xnew,Y[t],betas[i,],m[t])
	loglik_ra[i] <- loglik_t[i]
	grad[1,] <- gradient(Xnew,Y[t],betas[i,],m[t])
	betas[1,] <- 0

	#2. Perform stochastic gradient descent.
	for (i in 2:maxiter){
	
		converged <- 0; 	#Set indicator for convergence status.
		
		#Select one random obs per iter.
		t  <- obs_order[i]
		Xnew <- matrix(X[t,,drop=F],nrow=1,byrow=T)
		
		#Refresh step size using backtracking line search with 
		#new mini-batch for every 100th obs, including first real iteration (i=2).
		if(i==2 | i %% 100 == 0){ 		#(If i mod 100 = 0, obs is a multiple of 100.)
			step <- linesearch_mb(t,X,Y,b=betas[i-1,],m,maxalpha=1,batchsize)
		}
			
		#step <- linesearch(X,Y,b=betas[i-1,],gr=grad[i-1,],p=-grad[i-1,],m,maxalpha=1)
		
		#Set new beta equal to beta - a*gradient(beta).
		betas[i,] <- betas[i-1,] - step * grad[i-1,]
	
		#Calculate fullloglikelihood for each iteration.
		loglik[i] <- logl(X,Y,betas[i,],m)
		
		#Calculate loglikelihood of individual observation t.
		loglik_t[i] <- logl(Xnew,Y[t],betas[i,],m[t])
	
		#Calculate running average of loglikelihood for individual t's.
		loglik_ra[i] <- (loglik_ra[i-1]*(i-1) + loglik_t[i])/i
		
		#Calculate stochastic gradient for beta, using only obs t.
		grad[i,] <- gradient(Xnew,Y[t],betas[i,],m[t])
	
		print(i)
	
		#Check if convergence met:  If yes, exit loop.
		#Note: Not using norm(gradient) like with regular gradient descent.
		#Gradient is too variable in stochastic case.
		#Can run for set iterations, but here, checking for convergence based
		#on iter over iter change in running avg of log-likelihoods.
	
		#Check if convergence met: If yes, exit loop.
		if (abs(loglik_ra[i]-loglik_ra[i-1])/abs(loglik_ra[i-1]+1E-3) < conv ){
			converged=1;
			break;
		}
	
	} #End gradient descent iterations.
	
	#Return function output.
	return(list(beta_hat=betas[i,], 
			iter=i, 
			converged=converged, 
			loglik_full=loglik[1:i],
			loglik_ra = loglik_ra[1:i],
			loglik_t = loglik_t[1:i]))
}


#------------------------------------------------------------------
#1. Fit glm model for comparison. (No intercept: already added to X.)
glm1 = glm(y~X-1, family='binomial') #Fits model, obtains beta values.
beta <- glm1$coefficients

output <- stoch_gradient_descent_mbls(X,Y,m,maxiter=1000000,conv=1E-10,batchsize=100)

#------------------------------------------------------------------
#OUTPUT DATA RESULTS:

beta #GLM estimates
output$beta_hat #Stochastic estimates.


#Plot full log-likelihood function for convergence, and running average for log-likelihoods.
par(mfrow=c(2,1))
plot(1:output$iter,output$loglik_full,type='l',xlab='i',ylab='full neg loglhood')
plot(1:output$iter,output$loglik_ra,type='l',xlab='i',ylab='running avg neg loglhood')

#Save plots:
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 LaTeX Files/Ex04_loglik_and_ravg.jpeg')
par(mfrow=c(2,1))
plot(1:output$iter,output$loglik_full,type='l',xlab='i',ylab='full neg loglhood')
plot(1:output$iter,output$loglik_ra,type='l',xlab='i',ylab='running avg neg loglhood')
dev.off()

