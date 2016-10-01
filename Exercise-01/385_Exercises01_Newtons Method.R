### SDS 385 - Exercises 01 - Part B - Problem D
#This code implements Newton's Method to estimate the 
#beta coefficients for binomial logistic regression.

#Jennifer Starling
#26 August 2016

rm(list=ls())
library(Matrix)

#PART C:

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
	w <- 1 / (1 + exp(-X %*% beta))	#SCalculate probabilities vector w_i.
	
	gradient <- array(NA,dim=length(beta))	#Initialize the gradient.
	gradient <- -apply(X*as.numeric(Y-m*w),2,sum) #Calculate the gradient.
	
	return(gradient)
}

#------------------------------------------------------------------
#Gradient Function: 

hessian <- function(X,Y,beta,m){
	w <- 1 / (1 + exp(-X %*% beta))	#Calculate probabilities vector w_i.
	
	#Create diag matrix of weights with ith element equal to m_i*w_i*(1-w_i)
	A <- Diagonal(length(m),m*w*(1-w)) 
	
	#Calculate Hessian as X'AX.
	H <- t(X) %*% A %*% X
	return(H)
}

#------------------------------------------------------------------
#QR Solver Function:
qr_decomp <- function(A,b){
	#Solves linear system Ax=b.
	
	#Obtain QR decomposition of matrix A.  Extract components.
	QR <- qr(A)
	Q <- qr.Q(QR)
	R <- qr.R(QR)
	
	#Backsolve for x.
	x <- qr.solve(A,b)
	return(x)
}
#------------------------------------------------------------------

cholesky_method  <- function(X,W,y){	
	#Solves linear system Ax=b.
	#Since we have (X'WX)B=X'Wy, B (beta) acts as x, with A and b as follows.

	#Finding B (beta_hat) in equation 
	A = (t(X) * diag(W)) %*% X  #Efficient way of A = t(X) %*% W %*% X as W diag. 
								#Avoids mult by 0's.
	b = (t(X) * diag(W)) %*% y	#b'Wy
	R <- chol(A)	#Find right/upper cholesky decomposition of A.
	
	#Now we have R'R=A.
	
	#1. Solve R'z=b for z.  This is z = inv(R')b.
	z = solve(t(R)) %*% b
	
	#2. Solve Rx=z for x.  This is x = inv(R)z. (x = beta_hat)
	B_hat_chol <- solve(R) %*% z
	
	return(B_hat_chol)
}

#------------------------------------------------------------------
#Newton's Method algorithm:

#1. Fit glm model for comparison. (No intercept: already added to X.)
glm1 = glm(y~X-1, family='binomial') #Fits model, obtains beta values.
beta <- glm1$coefficients

loglik <- 0 		#Initialize vector to hold loglikelihood function.
grad <- list() 		#Initialize list to hold gradients for each iteration.
hess <- list()		#Initialize list to hold hessians for each iteration.
maxiter <- 100000 	#Specify max iterations allowed.
betas <- list()		#Initialize list to hold beta vector for each iteration.

conv <- 1E-6		#Set convergence level.

#Initialize first iteration of values.
betas[[1]] <- rep(0,ncol(X))			#i=1 betas
loglik[1] <- logl(X,Y,betas[[1]],m) 	#i=1 loglikelihood
grad[[1]] <- gradient(X,Y,betas[[1]],m)	#i=1 gradient
hess[[1]] <- hessian(X,Y,betas[[1]],m)	#i=1 hessian
step <- qr_decomp(hess[[1]],grad[[1]])

#2. Perform Newton's method.
for (i in 2:maxiter){
	
	#Set new beta equal to beta - gradient(beta)/hessian(beta). (inv(H) %*% grad)
	betas[[i]] <- as.vector(betas[[i-1]] - step)
	
	#Calculate loglikelihood for each iteration.
	loglik[i] <- logl(X,Y,betas[[i]],m)
	
	#Calculate gradient for beta.
	grad[[i]] <- gradient(X,Y,betas[[i]],m)
	
	#Calculate hessian for beta.
	hess[[i]] <- hessian(X,Y,betas[[i]],m)
	
	#Use QR decomp to solve Hess * dir = Gradient to get dir = H^-1 %*% Gradient
	#This gives step in next direction to update betas.
	step <- qr_decomp(hess[[i]],grad[[i]])
	
	#Check if convergence met: If yes, exit loop.
	if (abs(loglik[i]-loglik[i-1])/abs(loglik[i-1]+1E-3) < conv){
		print('Algorithm has converged.')
		print(i)
		break;
	}
	
	#Check if max iterations met: If yes, exit loop.
	if (i >= maxiter){
		print('Algorithm ending without convergence; max iterations reached.')
		break;
	} 	
} #End newton method iterations.

#Post-processing steps.
beta_newt <- betas[[i]]	#Save and output estimated beta values.
beta 					#Output glm beta values for comparison.
beta_newt

#Plot log-likelihood function for convergence.
plot(1:i,loglik,type='l',xlab='iterations',col='blue')	

#Save plot.
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Stats Models for Big Data/385_Exercise_R_Code/R_Output/Ex01_B_loglik.jpeg')
plot(1:i,loglik,type='l',xlab='iterations',col='blue')	
dev.off()
