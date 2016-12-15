### SDS 385 - Exercises 06 - Proximal Gradient Descent for LASSO.

#Jennifer Starling
#7 October 2016

rm(list=ls())	#Clean workspace.

library(glmnet)
library(Matrix)

#Read in Diabetes.csv data.
X <- read.csv(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 05 R Code/DiabetesX.csv',header=T)
y <- read.csv(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 05 R Code/DiabetesY.csv',header=F)

#Scale X and y.
X = scale(X)
y = scale(y)

#----------------------------------------------
#LASSO objective function:
#Inputs:
#	X = X matrix (scaled)
#	y = response data (scaled)
#	lambda = a chosen lambda value
#	beta = a vector of beta coefficients.
#Output:
#	Value of the LASSO objective function at specified inputs.
fx <- function(X,y,lambda,beta){
	f = (1/nrow(X)) * (t(y - X %*% beta) %*% (y - X %*% beta))
	g = lambda * sum(abs(beta))
	obj = (f+g)
	return(as.numeric(obj))
}


prox_l1 <- function(x, lambda){

  # Computes the soft thresholding estimator
  # ----------------------------------------
  # Args: 
  #   - x: vector of the observations
  #   - lambda: penalization parameter (threshold)
  # Returns: 
  #   - theta: the soft thresholding estimator
  # ------------------------------------------
  theta <- sign(x) * pmax(rep(0, length(x)), abs(x) - lambda)
  return (theta)
}

#----------------------------------------------
#Gradient for differentiable (non-penalty) part of LASSO objective:
gradient <- function(X,y,beta){
	grad = (2/nrow(X)) * (t(X) %*% X %*% beta - t(X) %*% y )
	return(grad)
}	

#----------------------------------------------
#Proximal Gradient Descent for L1 Norm Function:
#Inputs:
#	X = design matrix
#	y = response vector
#	gamma = step size
#	maxiter = maximum iterations
#	tol = tolerance for convergence
#	lambda = l1 norm penalty constant.
#Output:
#	List including estimated beta values and objective function.

proxGD <- function(X,Y,gamma=.01,maxiter=50,tol=1E-10,lambda=.1){
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#1. Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter,ncol=ncol(X)) 
	betas[1,] <- rep(0,ncol(X))	#Initialize beta vector to 0 to start.
	
	#2. Initialize values for objective function.
	obj <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood fctn.
	obj[1] <- fx(X,y,lambda,betas[1,])
	
	#3. Initialize matrix to hold gradients for each iteration.					
	grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 		
	
	for (i in 2:maxiter){
		#STEP 1: Gradient Step.
		
		#Calc gradient.
		#grad[i-1,] = (2/nrow(X)) * (t(X) %*% X %*% betas[i-1,]  - t(X) %*% y )
		grad[i-1,] = gradient(X,y,betas[i-1,])
		
		#Determine intermediate point.
		z = betas[i-1,] - gamma*grad[i-1,]
		
		#STEP 2: Proximal step.
		betas[i,] = prox_l1(z,gamma*lambda)
		
		#Update objective function.
		obj[i] = fx(X,y,lambda=lambda,beta=betas[i,])
	
		#Check if convergence met: If yes, exit loop.
		if (abs(obj[i]-obj[i-1])/abs(obj[i-1]+1E-3) < tol ){
			converged=1;
			break;
		}
	} #end for loop
	
	return(list(obj=obj, betas=betas, beta_hat=betas[i,], converged=converged, iter=i))
} #end function

#----------------------------------------------
#Accelerated Proximal Gradient Descent for L1 Norm Function:
#(Nesterov)
#Inputs:
#	X = design matrix
#	y = response vector
#	gamma = step size
#	maxiter = maximum iterations
#	tol = tolerance for convergence
#	lambda = l1 norm penalty constant.
#Output:
#	List including estimated beta values and objective function.

accelProxGD <- function(X,Y,gamma=.01,maxiter=50,tol=1E-10,lambda=.1){
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#1. Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter,ncol=ncol(X)) 
	betas[1,] <- rep(0,ncol(X))	#Initialize beta vector to 0 to start.
	
	#2. Initialize values for objective function.
	obj <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood fctn.
	obj[1] <- fx(X,y,lambda,betas[1,])
	
	#3. Initialize matrix to hold gradients for each iteration.					
	grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 
	grad[1,] =  gradient(X,y,betas[1,])
	
	#4. Initialize vectors to hold Nesterov update values.
	z = matrix(0,nrow=maxiter,ncol=ncol(X)) #Use initial z value of zero.
	s = rep(0,maxiter)	
	
	#Set up scalar s terms.  Ok before main loop, as do not depend on other terms' updates.
	for (j in 2:maxiter){
		s[j] = (1 + sqrt(1 + 4*(s[j-1])^2)) / 2
	}	
	
	#Loop through iterations until converged or maxiter met.
	for (i in 2:maxiter){
		
		#STEP 1: Gradient Step.
		
		#Calc gradient.
		grad[i-1,] =  gradient(X,y,z[i-1,])
		
		#Update intermediate u term.
		u = z[i-1,] - gamma * grad[i-1,]
		
		#STEP 2: Proximal step; update betas.
		betas[i,] = prox_l1(u,gamma*lambda)
		#betas[i,] = prox_l1(u,gamma,tau=lambda)
		
		#STEP 3: Nesterov step; update Nesterov momentum z.
		z[i,] = betas[i,] + ((s[i-1]-1)/s[i]) * (betas[i,] - betas[i-1,])
		
		#Update objective function.
		obj[i] = fx(X,y,lambda=lambda,beta=betas[i,])
	
		#Check if convergence met: If yes, exit loop.
		#if (abs(obj[i]-obj[i-1])/abs(obj[i-1]+1E-10) < tol ){
		#	converged=1;
		#	break;
		#}
	} #end for loop
	
	return(list(obj=obj, betas=betas, beta_hat=betas[i,], converged=converged, iter=i,s=s))
} #end function

#----------------------------------------------

#Run proximal gradient descent & accelerated proximal gradient descent.
lam=.01
output <- proxGD(X,y,gamma=.01,maxiter=1000,tol=1E-10,lambda=lam)
outputAccel <- accelProxGD(X,y,gamma=.01,maxiter=1000,tol=1E-10,lambda=lam)

#Iterations to convergence:
print(output$iter)
print(outputAccel$iter)
print(output$converged)
print(outputAccel$converged)

#Compare results to glmnet:
myLasso <- glmnet(X,y,family='gaussian',lambda=lam)	#fit glmnet model.
beta_glmnet <- myLasso$beta							#Save glmnet betas.
cbind(glmnet=beta_glmnet,
	proximal=output$beta_hat,
	accel.prox=round(outputAccel$beta_hat,8)) 	#output comparison


#Plot objective function.
plot(1:output$iter,output$obj[1:output$iter],type='l',log='xy',col='blue',xlab=paste('iter ',1,' to ',output$iter),
	ylab='objective function')
lines(1:outputAccel$iter,outputAccel$obj[1:outputAccel$iter],type='l',col='red',xlab=paste('iter ',1,' to ',outputAccel$iter),
	ylab='objective function')

#Plot convergence of betas.
idx = which(output$beta_hat>0)

#idxPlot = sample(idx,9,replace=F)
for (j in idx){
	plot(1:length(output$betas[,j]),output$betas[,j],xlab='iter',ylab=paste('betahat',j),type='l',col='blue')
	abline(h=beta_glmnet[j],col='red')
}



#----------------------------------------------
#Run accelerated proximal gradient descent.


