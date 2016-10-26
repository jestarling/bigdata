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
	obj = (1/2) * sum((y - X %*% beta) ^ 2) + lambda * sum(abs(beta))	
	return(obj)
}

#----------------------------------------------
#Proximal L1 Operator function: (soft thresholding operator)
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
#ADMM for Lasso:
#Inputs:
#	X = design matrix (A)
#	y = response vector (b)
#	rho = step size
#	maxiter = maximum iterations
#	eps_abs = primal tolerance for convergence (epsilon_abs)
#	eps_rel = dual tolerance for convergence (epsilon_rel)
#	lambda = l1 norm penalty constant.
#Output:
#	List including estimated beta (x) values and objective function.
#Note: In optimization notation, A=X, b=Y, x=beta (minimizing x).

admmLasso = function(X,Y,rho=1,lambda=.1,maxiter=1000,e_abs=1E-3,e_rel=1E-6){
		
	#Define dimensions n and p.
	n = nrow(X)
	p = ncol(X)
	
	#Rescale lambda to match glmnet results.
	lambda = lambda * n
	
	#Define function Euclidian (l2) norm of a vector.
	l2norm <- function(x) sqrt(sum(x^2))
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#Initialize data structures.
	betas <- matrix(0,nrow=maxiter,ncol=p) 	#holds beta vector for each iteration.
	obj <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood fctn.
	z = matrix(0,nrow=maxiter,ncol=p)	#Initialize z vector to all zeros.
	
	#Initialize values.
	obj[1] <- fx(X,y,lambda,betas[1,])	#Initialize objective.
	betas[1,] <- rep(0,p)	#Initialize beta vector to 0 to start.
	u = rep(0,p)	#Initialize the lagrangian to all zeros.
	
	#Pre-cache matrix inverse and Xty, since using fixed step size for each iter.
	Xty = crossprod(X,y)
	inv = solve(crossprod(X) + diag(rep(rho,p)))
	
	#Initialize residual vectors.
	s = 0	#dual residual
	r = 0	#primal residual

	#ADMM looping.
	for (i in 2:maxiter){
		
		#Update betas.
		betas[i,] = inv %*% (Xty + rho * (z[i,]-u) )
		
		#Update z.
		z[i,] = prox_l1(betas[i,] + u,lambda/rho)
		
		#Update u (lagrangian).
		u = u + betas[i,] - z[i,]
		
		#Update objective function.
		obj[i] = fx(X,y,lambda=lambda,beta=betas[i,])
		
		#--------------------------
		#Convergence check:
		
		#Calculate residuals for iteration i.
		r = betas[i,] - z[i,]
		s = -rho * (z[i,] - z[i-1,])
		
		r.norm = l2norm(r)
		s.norm = l2norm(s)
		
		e.primal = sqrt(p)*e_abs + e_rel * max(l2norm(betas[i,]), l2norm(z[i,])) 
		e.dual =  sqrt(p)*e_abs + e_rel * l2norm(u)
		
		if (r.norm <= e.primal && s.norm <= e.dual){
			converged=1
			break
		}
		#--------------------------
	}
	
	#Return function values.
	return(list(obj=obj, betas=betas, beta_hat=betas[i,], converged=converged, iter=i))
}
#----------------------------------------------

#Run admm for lasso.
output <- admmLasso(X,y,rho=5,lambda=.01,maxiter=1000,e_abs=1E-6,e_rel=1E-2)

#Iterations to convergence:
print(output$iter)
print(output$converged)

#Plot objective function.
#jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 07 LaTeX Files/admm_objective.jpg')
plot(1:output$iter,output$obj[1:output$iter],type='l',col='blue',log='xy',
	main='Lasso Objective Function',xlab='iter',ylab='objective')
#dev.off()

#Compare results to glmnet:
myLasso <- glmnet(X,y,family='gaussian',alpha=1,lambda=.01,intercept=F,standardize=F)	#Fit lasso glmnet model.
beta_glmnet <- myLasso$beta									#Save glmnet betas.
cbind(glmnet=beta_glmnet,admm=round(output$beta_hat,8))		#Output comparison

