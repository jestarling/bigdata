#Big Data Exercise 9
#November 10 ,2016
#Jennifer Starling

#References paper:
# Witten, Tibshirani, Hastie.  2009. A penalized matrix decomposition...

#################################
###   REQUIRED FUNCTIONS:     ###
#################################

#l1 penalty; Soft thresholding operator.
soft <- function(x,theta){
  return(sign(x)*pmax(0, abs(x)-theta))
}

#l1 norm of a vector.
l1norm <- function(vec){
  a <- sum(abs(vec))
  return(a)
}
 
#l2 norm of a vector.
l2norm <- function(vec){
  a <- sqrt(sum(vec^2))
  return(a)
}

#Binary search function 
#(Source: Witten, Hastie & Tibshirani R package PMA: https://cran.r-project.org/web/packages/PMA/)
#(For finding theta for soft-thresholding for each iteration)
BinarySearch <- function(argu,sumabs){

  l2n = function(vec) {return(sqrt(sum(vec^2)))}	
  soft = function(x,theta) { return(sign(x)*pmax(0, abs(x)-theta))}
	
  if(l2n(argu)==0 || sum(abs(argu/l2n(argu)))<=sumabs) return(0)
  lam1 <- 0
  lam2 <- max(abs(argu))-1e-5
  iter <- 1
  while(iter < 150){
    su <- soft(argu,(lam1+lam2)/2)
    if(sum(abs(su/l2n(su)))<sumabs){
      lam2 <- (lam1+lam2)/2
    } else {
      lam1 <- (lam1+lam2)/2
    }
    if((lam2-lam1)<1e-6) return((lam1+lam2)/2)
    iter <- iter+1
  }
  warning("Didn't quite converge")
  return((lam1+lam2)/2)
}

######################################################
###   PENALIZED MATRIX DECOMPOSITION FUNCTION:     ###
######################################################


#---------------------------------------------------------------------
#Sparse Matrix Factorization (Penalized Matrix Decomposition) Function
#For a single factor, ie K=1 (rank-1 approximation to original matrix)
#Inputs:
#	X = matrix to be factorized
#	lambdaU = the u penalty (c1)
# 	lambdaV = the v penalty (c2)
#	   *If lambda1 = lambda2 = 0, function returns the non-sparse Rank 1 SVD of X.
# 	maxiter = maximum number of iterations allowed
#	tol = tolerance level for convergence check
#Output: List object, including the following:
#	Xsp = sparse matrix factorization of X.
#	U, D, V = the decomposed elements of X, where X = U * D * t(V)

sparse.matrix.factorization.rank1 = function(X, lambdaU=1, lambdaV=1, maxiter=20, tol=1E-6){ 
	
	#1. Housekeeping parameters.
	i=1					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	p = ncol(X)			#Number of columns of X matrix.
	
	#2. Initializations
	v.old = rnorm(p)	#Initialize v.old to a random vector. (To get iterations started.)
	v = rep(sqrt(1/p),p) #Initialize v to meet constraint l2norm(v) = 1.
	
	#Iterate until convergence.
	for (i in 1:maxiter){
		
		#1. Update u.
		
		#First, calculate theta for sign function.
		u.arg = X %*% v		#Argument to go into sign function: Xv
		u.theta = BinarySearch(u.arg,lambdaU)
		
		#Second, update u.
		u = matrix( soft(u.arg,u.theta) / l2norm(soft(u.arg,u.theta)), ncol=1)

		#------------------------
		#2. Update v.
		
		#First, calculate theta for sign function.
		v.arg = t(X) %*% u
		v.theta = BinarySearch(v.arg,lambdaV)
		
		#Second, update v.
		v = matrix( soft(v.arg,v.theta) / l2norm(soft(v.arg,v.theta)), ncol=1)
		
		#------------------------
		#3. Convergence check steps.
		
		#Exit loop if converged.
		if(sum(abs(v.old - v)) < tol){
			converged=1
			break
		}
		
		#If not converged, update v.old for next iteration.
		v.old = v	
	}
	
	#Set d value.
	d = as.numeric(t(u) %*% (X %*% v))
	
	#Reconstruct sparse X matrix.
	Xsp = d * tcrossprod(u,v)
	
	#Return	function results.
	return(list(Xsp=Xsp,u=u,d=d,v=v,lambdaU=lambdaU,lambdaV=lambdaV,converged=converged,iter=i))
}	
#---------------------------------------------------------------------

#Simulate a matrix to chek that results behaving sensibly.

X = matrix(rnorm(20),nrow=5,ncol=4)
n = nrow(X)
p = ncol(X)

#Paper notes that if you want u and v to be equally sparse, set a constant c,
#and let lambdaU = c*sqrt(n), and let lambdaV = c * sqrt(p)

c = 2
lambdaU = c*sqrt(n)
lambdaV = c*sqrt(p)

test2 = sparse.matrix.factorization.rank1(X,lambdaU,lambdaV,maxiter=20,tol=1E-6)

#Just a few random tests with various lambda values.
#Confirm that u and v getting more sparse as lambdas decrease.
test2 = sparse.matrix.factorization.rank1(X,lambdaU=2,lambdaV=2,maxiter=20,tol=1E-6)
test1.5 = sparse.matrix.factorization.rank1(X,lambdaU=1.5,lambdaV=1.5,maxiter=20,tol=1E-6)
test1 = sparse.matrix.factorization.rank1(X,lambdaU=1,lambdaV=1,maxiter=20,tol=1E-6)
test.5 = sparse.matrix.factorization.rank1(X,lambdaU=.5,lambdaV=.5,maxiter=20,tol=1E-6)
test0 = sparse.matrix.factorization.rank1(X,lambdaU=0,lambdaV=0,maxiter=20,tol=1E-6)

#Number of non-sparse u and v in each test.
lambdas = c(2,1.5,1,.5,0)
nonzero.u = c(sum(test2$u!=0),sum(test1.5$u!=0),sum(test1$u!=0),sum(test.5$u!=0),sum(test0$u!=0))
nonzero.v = c(sum(test2$v!=0),sum(test1.5$v!=0),sum(test1$v!=0),sum(test.5$v!=0),sum(test0$v!=0))

cbind.data.frame(lambdas=lambdas,nonzero.u=nonzero.u,nonzero.v=nonzero.v)

print("lambdaU = lambdaV = 2")
test2$u
test2$v

print("lambdaU = lambdaV = 1.5")
test1.5$u
test1.5$v

print("lambdaU = lambdaV = 1")
test1$u
test1$v

print("lambdaU = lambdaV = .5")
test.5$u

print("lambdaU = lambdaV = 0")
test0$u

###########################################
###   APPLICATION TO MARKETING          ###
###########################################

#Read in marketing data.
data = read.csv('/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Course Data/social_marketing.csv',header=T)
X.counts = as.matrix(data[,-1])	#Gets rid of ID column.

#Square Root or Anscombe-transform data, because it is count data.
#Chose square root in this case.
X = sqrt(X.counts)

#Eyeball data to see which categories might be interesting.
cbind(colmeans=sort(colMeans(X),decreasing=T))

#Try various lambda values to see which categories are interesting.
result1 = sparse.matrix.factorization.rank1(X, lambdaU=5, lambdaV=5, maxiter=20, tol=1E-6)
cbind.data.frame(colnames(X),result1$v)

result2 = sparse.matrix.factorization.rank1(X, lambdaU=2, lambdaV=2, maxiter=20, tol=1E-6)
cbind.data.frame(colnames(X),result2$v)	#Show all.
cbind.data.frame(cols=colnames(X)[which(result2$v!=0)],v=result2$v[which(result2$v!=0)])
	#show non-zero only.

result3 = sparse.matrix.factorization.rank1(X, lambdaU=1.5, lambdaV=1.5, maxiter=20, tol=1E-6)
cbind.data.frame(colnames(X),result3$v)
cbind.data.frame(cols=colnames(X)[which(result3$v!=0)],v=result3$v[which(result3$v!=0)])
	#show non-zero only.

