### SDS 385 - Exercises 07 - ADMM for LASSO.
### ADMM = Alternating Direction Method of Multipliers

#Jennifer Starling
#18 October 2016

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
	#f = (1/nrow(X)) * (t(y - X %*% beta) %*% (y - X %*% beta))
	f = (1/2) * (t(y - X %*% beta) %*% (y - X %*% beta))
	g = lambda * sum(abs(beta))
	obj = (f+g)
	return(as.numeric(obj))
}

#----------------------------------------------
#Proximal L1 Operator function: (soft thresholding operator)
#Inputs:
#	x = vector of values.
#	lambda = the scaling factor of the l1 norm.
#	t = the step size.

#Output:
#	Value of the soft-thresholding proximal operator.
prox_l1 <- function(x,gamma,tau=1) {
	
	thresh <- gamma*tau
	prox = rep(0,length(x))
	
	idx.1 = which(x < -thresh)
	idx.2 = which(x > thresh)
	idx.3 = which(abs(x) <= thresh)
	
	if (length(idx.1) > 0) prox[idx.1] = x[idx.1] + thresh
	if (length(idx.2) > 0) prox[idx.2] = x[idx.2] - thresh
	if (length(idx.3) > 0) prox[idx.3] = 0

    return(prox)
}

#----------------------------------------------
#ADMM for Lasso:
#Inputs:
#	X = design matrix
#	y = response vector
#	rho = step size
#	maxiter = maximum iterations
#	tol = tolerance for convergence
#	lambda = l1 norm penalty constant.
#Output:
#	List including estimated beta values and objective function.
admmLasso = function(X,Y,rho=.01,lambda=.1,maxiter=1000,tol=1E-10){
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#1. Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter,ncol=ncol(X)) 

	
	#2. Initialize values for objective function.
	obj <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood fctn.
	obj[1] <- fx(X,y,lambda,betas[1,])
	
	#3. Cache matrix inverse, since using fixed step size for each iter.
	inv_cache = solve(t(X) %*% X + rho*diag(1,ncol(betas)))
	
	#4. Initialize values.
	betas[1,] <- rep(0,ncol(X))	#Initialize beta vector to 0 to start.
	u = rep(0,ncol(betas))	#Initialize the lagrangian to all zeros.
	z = rep(0,ncol(betas))	#Initialize z vector to all zeros.

	#4. ADMM looping.
	for (i in 2:maxiter){
		
		#Update betas.
		betas[i,] = inv_cache %*% (t(X) %*% y + rho * (z-u) )
		
		#Update z.
		z = prox_l1(betas[i,] + u,lambda/rho)
		
		#Update u (lagrangian).
		u = u + betas[i,] - z
		
		#Update objective function.
		obj[i] = fx(X,y,lambda=lambda,beta=betas[i,])

		#Convergence check.
		#Check if convergence met: If yes, exit loop.
		if (abs(obj[i]-obj[i-1])/abs(obj[i-1]+1E-3) < tol ){
			converged=1;
			break;
		}
	}
	
	#Return function values.
	return(list(obj=obj, betas=betas, beta_hat=betas[i,], converged=converged, iter=i))
}
#----------------------------------------------

#Run admm for lasso.
output <- admmLasso(X,y,rho=.01,lambda=.01,maxiter=1000,tol=1E-14)

#Iterations to convergence:
print(output$iter)
print(output$converged)

#Plot objective function.
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 07 LaTeX Files/admm_objective.jpg')
plot(1:output$iter,output$obj[1:output$iter],type='l',col='blue',log='xy',
	main='Lasso Objective Function',xlab='iter',ylab='objective')
dev.off()

#Compare results to glmnet:
myLasso <- glmnet(X,y,family='gaussian',alpha=1,lambda=.01)	#Fit lasso glmnet model.
beta_glmnet <- myLasso$beta									#Save glmnet betas.
cbind(glmnet=beta_glmnet,admm=round(output$beta_hat,8))		#Output comparison

