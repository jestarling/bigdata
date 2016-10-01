### SDS 385 - Exercises 04 - Part B
#This code implements stochastic gradient descent
#using adagrad.

#Jennifer Starling
#18 September 2016

rm(list=ls())	#Cleans workspace.
library(microbenchmark)
library(permute)

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
	#	lambda = L2 regularization coefficient, lambda > 0.
	#Output: Returns value of negative log-likelihood 
	#   function for binomial logistic regression.
logl <- function(X,Y,beta,m,lambda=0){
	w <- 1 / (1 + exp(-X %*% beta))	#Calculate probabilities vector w_i.
	logl <- - sum(Y*log(w+1E-4) + (m-Y)*log(1-w+1E-4)) + lambda*sum(beta^2) #Calculate log-likelihood.
		#Adding constant to resolve issues with probabilities near 0 or 1.	
	return(logl)	
}

#------------------------------------------------------------------
#Gradient Function: 
	#Inputs: Vector X (One row of design matrix), vector of 1/0 vals Y, 
	#   coefficient matrix beta, sample size vector m.
	#	lambda = L2 regularization coefficient, lambda > 0.
	#Output: Returns value of gradient function for binomial 
	#   logistic regression.

gradient <- function(X,Y,beta,m,lambda=0){
	w <- 1 / (1 + exp(-X %*% beta))	#Calculate probabilities vector w_i.
	gradient <- crossprod(X,as.numeric(m*w-Y)) + 2*lambda*beta #Calculate the gradient.
	
	return(gradient)
}

#------------------------------------------------------------------
#AdaGrad Stochastic Gradient Descent Algorithm:

	#Inputs:
		#step = master step size.
		#sparse=F indicator if design matrix X is sparse


sgd_adagrad <- function(X,Y,m,step=.01,maxiter=50000,conv=1E-10,lambda=0,sparse=0){

	#If sparse=T, set X as a sparse matrix type.
	if (sparse==1){
		X = Matrix(X,sparse=T)
	}
	
	converged <- 0 #Indicator variable to track convergence status.
	
	#Set up random iterations through data, up to maxiter.
	npermutes <- ceiling(maxiter/nrow(X))
	obs_order <- as.vector(t(shuffleSet(1:nrow(X),nset=npermutes)))
	
	#Initialize matrix to hold gradients for each iteration.					
	grad <- matrix(0,nrow=maxiter,ncol=ncol(X))
	
	#Initialize vectors to hold Adagrad historical and adjusted gradients.
	hist_grad <- rep(0,ncol(X))
	adj_grad <- rep(0,ncol(X))
	
	#Initialize constant for numerical stability in Adagrad calculations.
	epsilon <- 1E-6 		

	#Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter+1,ncol=ncol(X)) 	

	#Initialize log-likelihood vectors.
	loglik <- rep(0,maxiter)	#Full loglhood.
	loglik_t <- rep(0,maxiter)	#Per-obs loglhood.	
	loglik_ra <- rep(0,maxiter)	#Running avg of loglhoods.

	#Initialize values for first iteration:
	i=1
	t  <- obs_order[i]
	Xnew <- matrix(X[t,,drop=F],nrow=1,byrow=T)
	loglik_t[i] <- logl(Xnew,Y[t],betas[i,],m[t],lambda)
	loglik_ra[i] <- loglik_t[i]
	grad[1,] <- gradient(Xnew,Y[t],betas[i,],m[t],lambda)
	betas[1,] <- 0

	#2. Perform stochastic gradient descent.
	for (i in 2:maxiter){
	
		#Select one random obs per iter.
		t  <- obs_order[i]
		Xnew <- matrix(X[t,,drop=F],nrow=1,byrow=T)
		
		#Calculate updated AdaGrad historical and adjusted gradients.
		hist_grad <- hist_grad + grad[i-1,]^2
		adj_grad <- grad[i-1,] / (sqrt(hist_grad) + epsilon)
		
		#Set new beta equal to beta - a*adj_grad(beta_i-1) where 
		#adj_grad is the AdaGrad adjusted gradient.
		betas[i,] <- betas[i-1,] - step * adj_grad
		
		#Calculate updated gradient for beta, using only obs t.
		grad[i,] <- gradient(Xnew,Y[t],betas[i,],m[t])
	
		#Calculate fullloglikelihood for each iteration.
		loglik[i] <- logl(X,Y,betas[i,],m,lambda)
		
		#Calculate loglikelihood of individual observation t.
		loglik_t[i] <- logl(Xnew,Y[t],betas[i,],m[t],lambda)
	
		#Calculate running average of loglikelihood for individual t's.
		loglik_ra[i] <- (loglik_ra[i-1]*(i-1) + loglik_t[i])/i
	
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
			betas=betas, 
			iter=i, 
			converged=converged, 
			loglik_full=loglik[1:i],
			loglik_ra = loglik_ra[1:i],
			loglik_indiv = loglik_t[1:i]))
}


#------------------------------------------------------------------
#OUTPUT ANALYSIS:

#1. Fit glm model for comparison. (No intercept: already added to X.)
glm1 = glm(y~X-1, family='binomial') #Fits model, obtains beta values.
beta <- glm1$coefficients

#2. Run Adagrad algorithm.
output <- sgd_adagrad(X,Y,m,step=5,maxiter=1000000,conv=1E-12,lambda=0,sparse=0)

#3. Output beta estimates for glm vs adagrad.
beta #GLM estimates
output$beta_hat #Stochastic estimates.

#4. Plog full loglhood, running avg loglhood, and convergence of beta estimates.

#Plot full log-likelihood function for convergence, and running average for log-likelihoods.
par(mfrow=c(2,1))
plot(1:output$iter,output$loglik_full,type='l',xlab='i',ylab='full neg loglhood')
plot(1:output$iter,output$loglik_ra,type='l',xlab='i',ylab='running avg neg loglhood')

#Plot the convergence of the beta variables compared to glm.
par(mfrow=c(4,3))
for (j in 1:length(output$beta_hat)){
	plot(1:nrow(output$betas),output$betas[,j],type='l',xlab='iterations',ylab=paste('beta',j))
	abline(h=beta[j],col='red')
}


#------------------------------------------------------------------
#SAVE PLOTS TO JPG FILES:

#Save plots:
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 LaTeX Files/Ex04_adagrad_loglik.jpg')

#Plot full log-likelihood function for convergence, and running average for log-likelihoods.
par(mfrow=c(2,1))
plot(1:output$iter,output$loglik_full,type='l',xlab='i',ylab='full neg loglhood')
plot(1:output$iter,output$loglik_ra,type='l',xlab='i',ylab='running avg neg loglhood')
dev.off()

jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 04 LaTeX Files/Ex04_adagrad_betas.jpg')
par(mfrow=c(4,3))
for (j in 1:length(output$beta_hat)){
	plot(1:nrow(output$betas),output$betas[,j],type='l',xlab='iterations',ylab=paste('beta',j))
	abline(h=beta[j],col='red')
}
dev.off()

