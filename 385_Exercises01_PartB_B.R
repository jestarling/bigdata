### SDS 385 - Exercises 01 - Part B - Problem B
#This code implements gradient descent to estimate the 
#beta coefficients for binomial logistic regression.

#Jennifer Starling
#26 August 2016

library(Matrix)
rm(list=ls())

#PART B:

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
	logl <- - sum(Y*log(w+.01) + (m-Y)*log(1-w+.01)) #Calculate log-likelihood.
		#Adding .01 to resolve issues with probabilities near 0 or 1.	
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
#Gradient Descent Algorithm:
#Inputs:
#	X: n x p design matrix.
#	Y: response vector length n.
#	m: vector length n.
#	conv: Tolerance level for determining convergence, (length of gradient) < conv.
#	a: Step size.

#Outputs:
#	beta_hat: A vector of estimated beta coefficients.
#	iter: The number of iterations until convergence.
#	converged: 1/0, depending on whether algorithm converged.
#	loglik: Log-likelihood function.

gradient_descent <- function(X,Y,m,maxiter=50000,conv=1*10^-3,a=.01){
	
	#1. Initialize values.
	loglik <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood function.
	
	#Initialize matrix to hold gradients for each iteration.					
	grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 		

	#Initialize matrix to hold beta vector for each iteration.
	betas <- matrix(0,nrow=maxiter+1,ncol=ncol(X)) 	

	converged <- 0		#Indicator for whether convergence met.
	betas[1,] <- rep(0,ncol(X))	#Initialize beta vector to 0 to start.

	#2. Perform gradient descent.
	for (i in 1:maxiter){
	
		#Calculate loglikelihood for each iteration.
		loglik[i] <- logl(X,Y,betas[i,],m)
	
		#Calculate gradient for beta.
		grad[i,] <- gradient(X,Y,betas[i,],m)
	
		#Set new beta equal to beta - a*gradient(beta).
		betas[i+1,] <- betas[i,] - a * grad[i,]
	
		iter <- i + 1	#Track iterations.
	
		#Check if convergence met: If yes, exit loop.
		if (norm_vec(grad[i,]) < conv){
			converged=1;
			break;
		}
	
	} #End gradient descent iterations.
		
	return(list(beta_hat=betas[i,], iter=iter, converged=converged, loglik=loglik[1:i]))
}

#------------------------------------------------------------------
#Run gradient descent and view results.

#1. Fit glm model for comparison. (No intercept: already added to X.)
glm1 = glm(y~X-1, family='binomial') #Fits model, obtains beta values.
beta <- glm1$coefficients

#2. Call gradient descent function to estimate.
beta_hat <- gradient_descent(X,Y,m,maxiter=100000,conv=1*10^-3,a=.01)

#3. Eyeball values for accuracy & display convergence.
beta				#Glm estimated beta values.
beta_hat$beta_hat	#Gradient descent estimated beta values.

print(c("Algorithm converged? ",beta_hat$converged, " (1=converged, 0=did not converge)"))
print(beta_hat$iter)

#4. Plot log-likelihood function for convergence.
plot(1:length(beta_hat$loglik),beta_hat$loglik,type='l',xlab='iterations',col='blue',log='xy')

#Save plot.
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Stats Models for Big Data/385_Exercise_R_Code/R_Output/Ex01_B_loglik.jpeg')
plot(1:length(beta_hat$loglik),beta_hat$loglik,type='l',xlab='iterations',col='blue')
dev.off()
