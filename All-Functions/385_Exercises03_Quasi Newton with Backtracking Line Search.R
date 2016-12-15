### SDS 385 - Exercises 03 - Backtracking Line Search
#This code implements gradient descent to estimate the 
#beta coefficients for binomial logistic regression.
#It uses backtracking line search to calculate the step size.

#Jennifer Starling
#26 August 2016

library(Matrix)
rm(list=ls())	#Clean workspace.

#Read in code.
wdbc = read.csv('/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Stats Models for Big Data/Course Data/wdbc.csv', header=FALSE)
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
#Function for calculating Euclidean norm of a vector.
norm_vec <- function(x) sqrt(sum(x^2)) 

#------------------------------------------------------------------
#Gradient Function: 
	#Inputs: Design matrix X, vector of 1/0 vals Y, 
	#   coefficient matrix beta, sample size vector m.
	#Output: Returns value of gradient function for binomial 
	#   logistic regression.

gradient <- function(X,Y,beta,m){
	w <- 1 / (1 + exp(-X %*% beta))	#Calculate probabilities vector w_i.
	
	gradient <- array(NA,dim=length(beta))	#Initialize the gradient.
	gradient <- -apply(X*as.numeric(Y-m*w),2,sum) #Calculate the gradient.
	
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
	
	while( (logl(X,Y,b + alpha*p,m)) > logl(X,Y,b,m) + c*alpha*t(gr) %*% p ) {
		alpha <- rho*alpha
	}
	
	return(alpha)
}

#------------------------------------------------------------------
#Quasi Newton with Backtracking Line Search Algorithm:
#Inputs:
#	X: n x p design matrix.
#	Y: response vector length n.
#	m: vector length n.
#	conv: Tolerance level for evaluating convergence.
#	a: Step size.

#Outputs:
#	beta_hat: A vector of estimated beta coefficients.
#	iter: The number of iterations until convergence.
#	converged: 1/0, depending on whether algorithm converged.
#	loglik: Log-likelihood function.

quasi_newton <- function(X,Y,m,maxiter=50000,conv=1*10^-10){
	
	converged <- 0		#Indicator for whether convergence met.
	
	#1. Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter+1,ncol=ncol(X)) 
	betas[1,] <- rep(0,ncol(X))	#Initialize beta vector to 0 to start.
	
	#2. Initialize values for log-likelihood.
	loglik <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood fctn.
	loglik[1] <- logl(X,Y,betas[1,],m)
	
	#3. Initialize matrix to hold gradients for each iteration.					
	grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 		
	grad[1,] <- gradient(X,Y,betas[1,],m)
	
	#4. Initialize list of approximations of Hessian inverse, B. 
	#   (Use identity matrix as initial value.)
	B <- list()
	B[[1]] <- diag(ncol(betas))

	#5. Perform gradient descent.
	for (i in 2:maxiter){
		
		#Compute direction and step size for beta update.
		p <- -B[[i-1]] %*% grad[i-1,]
		alpha <- linesearch(X,Y,b=betas[i-1,],gr=grad[i-1,],p,m,maxalpha=1)
	
		#Update beta values based on step/direction.
		betas[i,] <- betas[i-1,] + alpha*p
		
		#Calculate loglikelihood for each iteration.
		loglik[i] <- logl(X,Y,betas[i,],m)
		
		#Calculate gradient for new betas.
		grad[i,] <- gradient(X,Y,betas[i,],m)
		
		#Update values needed for BFGS Hessian inverse approximation.
		s <- alpha*p
		z <- grad[i,] - grad[i-1,]
		rho <- as.vector(1/(t(z) %*% s))	 #as.vector to make rho a scalar.
		tau <- rho * s %*% t(z)	#Just breaking up the formula a bit for ease.
		I <- diag(ncol(grad))
		
		#BFGS formula for updating approx of H inverse.
		B[[i]] <- (I-tau) %*% B[[i-1]] %*% (I-t(tau)) + rho * s %*% t(s) 
	
		print(i)
	
		#Check if convergence met: If yes, exit loop.
		if (abs(loglik[i]-loglik[i-1])/abs(loglik[i-1]+1E-3) < conv ){
			converged=1;
			break;
		}
	
	} #End gradient descent iterations.
		
	return(list(betas=betas[1:i,],beta_hat=betas[i,], iter=i, converged=converged, loglik=loglik[1:i]))
}

#------------------------------------------------------------------
#Run gradient descent and view results.

#1. Fit glm model for comparison. (No intercept: already added to X.)
glm1 = glm(y~X-1, family='binomial') #Fits model, obtains beta values.
beta <- glm1$coefficients

#2. Call gradient descent function to estimate.
output <- quasi_newton(X,Y,m,maxiter=10000,conv=1*10^-10)

#3. Eyeball values for accuracy & display convergence.
beta				#Glm estimated beta values.
output$beta_hat	#Gradient descent estimated beta values.

#Print whether the algorithm has converged, and the number of iterations.
if(output$converged>0){cat('Algorithm converged in',output$iter, 'iterations.')}
if(output$converged<1){cat('Algorithm did not converge. Ran for max iterations.')}

#4. Plot the convergence of the beta variables compared to glm.
par(mfrow=c(4,3))
for (j in 1:length(output$beta_hat)){
	plot(1:nrow(output$betas),output$betas[,j],type='l',xlab='iterations',ylab=paste('beta',j))
	abline(h=beta[j],col='red')
}

#5. Plot log-likelihood function for convergence.
plot(1:length(output$loglik),output$loglik,type='l',xlab='iterations',col='blue',log='xy',
	main='Quasi-Newton Neg Loglikelihood Function')

#6. Save plots.
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Stats Models for Big Data/Exercise 03 LaTeX Files/Ex03_quasinewtonloglhood.jpg')
plot(1:length(output$loglik),output$loglik,type='l',xlab='iterations',col='blue',log='xy',
	main='Quasi-Newton Neg Loglikelihood Function')
dev.off()

jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Stats Models for Big Data/Exercise 03 LaTeX Files/Ex03_quasinewtonbetas.jpg')
par(mfrow=c(4,3))
for (j in 1:length(output$beta_hat)){
	plot(1:nrow(output$betas),output$betas[,j],type='l',xlab='iterations',ylab=paste('beta',j))
	abline(h=beta[j],col='red')
}

dev.off()
