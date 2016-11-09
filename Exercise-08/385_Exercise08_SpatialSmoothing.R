### SDS 385 - Exercise 08 - Discrete Spatial Smoothing
### Jennifer Starling
### Oct 26, 2016

rm(list=ls())	#Clean workspace.
library(Matrix)
library(gplots)	#For heatmap.2 function.

########################################
#             FUNCTIONS
########################################

#Constructs the oriented edge matrix D for a 2-d grid of size nx by ny.
makeD2_sparse = function (dim1, dim2)  {
	require(Matrix)
    D1 = bandSparse(dim1 * dim2, m = dim1 * dim2, k = c(0, 1), 
        diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * 
            dim2 - 1)))
    D1 = D1[(seq(1, dim1 * dim2)%%dim1) != 0, ]
    D2 = bandSparse(dim1 * dim2 - dim1, m = dim1 * dim2, k = c(0, 
        dim1), diagonals = list(rep(-1, dim1 * dim2), rep(1, 
        dim1 * dim2 - 1)))
    return(rBind(D1, D2))
}

#Direct solver for Ax=b.
direct.sparse.solver = function(A,b){
	x=solve(A,y,sparse=T)
	return(x)
}

#Jacobi iterative solver for Ax=b.
jacobi.solver = function(A,b,x0 = rep(1,length(b)),maxiter=100,tol=1E-14){
	#x0 = Initial guess for vector x solution.
	#A = input matrix
	#b = input vector
	#maxiter = maximum iterations
	
	#Initialize values.
	xold = x0
	Dinv = 1/diag(A)	#Diagonal of matrix A.
	R = A		#Remainder; A with diagonal set to 0.
	diag(R) = 0
	
	for (i in 1:maxiter){
		
		#Update x values.
		xnew = Dinv * (b - crossprod(R,xold))
		
		#Check convergence.
		if(max(abs(xnew-xold)) < tol) { break; }
		
		#Update xold.
		xold=xnew
	}
	return(xnew)
}

#Gauss-Seidel iterative solver for Ax=b. (c++)
sourceCpp(file='/Users/jennstarling/UTAustin/starlib/R/gauss_seidel_solver.cpp')



########################################
#             MAIN CODE
########################################

#Read in data. data.
data <- as.matrix(read.csv(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Course Data/fmri_z.csv',header=T))
data.mat = Matrix(data)

#Heat map of noisy MRI data.
image(t(data.mat), sub='', xlab='',ylab='',cuts=80)

#Represent signal values as a vector.
y = as.vector(data.mat)

#Oriented edge matrix.
D = makeD2_sparse(128,128)

#Laplacian matrix.
L = t(D) %*% D

#Lambda penalty.
lambda=10

#Set up A = (I + lambda*L)
A = lambda*L
diag(A) = diag(A) + 1

#DIRECT SOLVER
xD = direct.sparse.solver(A,y)
xD.grid = Matrix(as.vector(xD),128,128,byrow=T)	#Reconstruct de-noised grid.
xD.grid[which(data.mat==0)]=0					#Set values which were 0 in original grid to 0.

#JACOBI SOLVER
xJ = jacobi.solver(A,y,maxiter=1000)
xJ.grid = Matrix(as.vector(xJ),128,128,byrow=T)	#Reconstruct de-noised grid.
xJ.grid[which(data.mat==0)]=0					#Set values which were 0 in original grid to 0.

#Quick comparison of Jacobi and Direct solver methods.
cbind(xD,xJ)

#Side by side heat maps of noisy and de-noised data.
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 08 LaTeX Files/lp.smoothing.noisey.jpg')
image(t(data.mat), sub='', xlab='',ylab='',cuts=80)
dev.off()
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 08 LaTeX Files/lp.smoothing.jacobismoothed.jpg')
image(t(xJ.grid), sub='', xlab='',ylab='',cuts=80)
dev.off()
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 08 LaTeX Files/lp.smoothing.directsmoothed.jpg')
image(t(xD.grid), sub='', xlab='',ylab='',cuts=80)
dev.off()


########################################
#            GRAPH FUSED LASSO
########################################

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
#Generalized Lasso:

#Inputs:
#   A = Design matrix. Use identity for graph-fused lasso.  (nxp)
#	b = response vector (length n)
#	D = Difference matrix (first diff matrix for graph fused lasso.)
#	x0 = initial vector of x guesses. (length n)
#	rho = step size
#	lambda = l1 norm penalty constant.
#	maxiter = maximum iterations
#	eps_abs = primal tolerance for convergence (epsilon_abs)
#	eps_rel = dual tolerance for convergence (epsilon_rel)

#Output:
#	List including estimated beta (x) values and objective function.
#	Note: In optimization notation, A=X, b=Y, x=beta (minimizing x).

gfl_admm_slow = function(D,y,x0,
							rho=1,
							lambda=.1,
							maxiter=1000,
							e_abs=1E-3,
							e_rel=1E-6){
		
	#Define dimensions n and p.
	p = nrow(D)
	
	#Set up identity matrix.  (Use to check if A==I to streamline matrix operations.)
	#I = diag(1,length(b))
	
	#Define function Euclidian (l2) norm of a vector.
	l2norm <- function(x) sqrt(sum(x^2))
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#Initialize data structures.
	x = x0								#Vector to hold x's, initialized to x0.
	obj <- rep(0,maxiter) 				#Initialize vector to hold loglikelihood fctn.
	z = matrix(0,nrow=maxiter,ncol=p)	#Initialize z vector to all zeros.
	u = rep(0,p)						#Initialize u vector to all zeros.
	
	#Pre-cache (AtA + rho*DtD)^-1
		inv = rho * crossprod(D)
		diag(inv) = diag(inv) + 1
		inv = solve(inv)
	
	#Initialize residual vectors.
	s = 0	#dual residual
	r = 0	#primal residual

	#ADMM looping.
	for (i in 2:maxiter){
		
		#Update betas.
		x = inv %*% (y + rho * t(D) %*% (z[i-1,] - u))
		#x = crossprod(inv, (Atb + rho * (z[i,]-u)) )
		
		#Recalculate D %*% x
		Dx = D %*% as.vector(x)
		
		#Update z.
		z[i,] = prox_l1(as.vector(Dx + u),lambda/rho)
		
		#Update u.
		u = u + Dx - z[i,]
		
		#Update objective function.
		#obj[i] = fx(X,y,lambda=lambda,beta=x)
		
		#--------------------------
		#Convergence check:
		
		#Calculate residuals for iteration i.
		#r = x - z[i,]
		#s = -rho * (z[i,] - z[i-1,])
		
		#r.norm = l2norm(r)
		#s.norm = l2norm(s)
		
		#e.primal = sqrt(p)*e_abs + e_rel * max(l2norm(x), l2norm(z[i,])) 
		#e.dual =  sqrt(p)*e_abs + e_rel * l2norm(u)
		
		#if (r.norm <= e.primal && s.norm <= e.dual){
		#	converged=1
		#	break
		#}
		#--------------------------
		print(i)
	}
	
	#Return function values.
	return(list(x=x))
}

#----------------------------------------------

#Run graph-fused lasso.
gl = gfl_admm_slow(D,y,x0=rep(1,length(y)),rho=1,lambda=5,maxiter=10)
gfl.grid = Matrix(as.vector(gl$x),128,128,byrow=T)	#Reconstruct de-noised grid.
gfl.grid[which(data.mat==0)]=0					#Set values which were 0 in original grid to 0.
image(t(gfl.grid), sub='', xlab='',ylab='',cuts=80)

#----------------------------------------------
#Generalized Lasso:

#Inputs:
#   A = Design matrix. Use identity for graph-fused lasso.  (nxp)
#	b = response vector (length n)
#	D = Difference matrix (first diff matrix for graph fused lasso.)
#	x0 = initial vector of x guesses. (length n)
#	rho = step size
#	lambda = l1 norm penalty constant.
#	maxiter = maximum iterations
#	eps_abs = primal tolerance for convergence (epsilon_abs)
#	eps_rel = dual tolerance for convergence (epsilon_rel)

#Output:
#	List including estimated beta (x) values and objective function.
#	Note: In optimization notation, A=X, b=Y, x=beta (minimizing x).



graph_fused_lasso_admm = function(y,D,x0,rho=1,lambda=5,maxiter=100,e_abs=1E-3,e_rel=1E-6){
		
	#Define dimensions n and p.
	p = length(y)
	
	#Rescale lambda to match glmnet results.
	lambda = lambda * n
	
	#Define function Euclidian (l2) norm of a vector.
	l2norm <- function(x) sqrt(sum(x^2))
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#Initialize data structures.
	x = x0									#Vector to hold x's, initialized to x0.
	obj <- rep(0,maxiter) 					#Initialize vector to hold loglikelihood fctn.
	z = matrix(0,nrow=maxiter,ncol=nrow(D))	#Initialize z vector to all zeros.
	u = rep(0,nrow(D))						#Initialize u vector to all zeros.
	
	#Pre-cache (AtA + rho*DtD)^-1
	inv = rho*crossprod(D)
	diag(inv) = diag(inv) + 1
	inv = solve(inv)
	
	#Initialize residual vectors.
	s = 0	#dual residual
	r = 0	#primal residual

	#ADMM looping.
	for (i in 2:maxiter){
		
		#Update betas.
		x = inv %*% (b + rho * t(D) %*% (z[i,]-u))
		
		#Recalculate D %*% x
		Dx = D %*% x
				
		#Update z.
		z[i,] = prox_l1(as.vector(Dx + u),lambda/rho)
		
		#Update u.
		u = u + Dx - z[i,]
		
		#Update objective function.
		#obj[i] = fx(X,y,lambda=lambda,beta=x)
		
		#--------------------------
		#Convergence check:
		
		#Calculate residuals for iteration i.
		r = x - z[i,]
		s = -rho * (z[i,] - z[i-1,])
		
		r.norm = l2norm(r)
		s.norm = l2norm(s)
		
		e.primal = sqrt(p)*e_abs + e_rel * max(l2norm(x), l2norm(z[i,])) 
		e.dual =  sqrt(p)*e_abs + e_rel * l2norm(u)
		
		if (r.norm <= e.primal && s.norm <= e.dual){
			converged=1
			break
		}
		#--------------------------
	}
	
	#Return function values.
	return(list(x=x, converged=converged, iter=i))
}
#----------------------------------------------




#----------------------------------------------
#Generalized Lasso:

#Inputs:
#	y = response vector (length n)
#	D = Difference matrix (first diff matrix for graph fused lasso, mxn)
#	x0 = initial vector of x guesses. (length n)
#	rho = step size parameter
#	lambda = l1 norm penalty constant.
#	maxiter = maximum iterations
#	eps_abs = primal tolerance for convergence (epsilon_abs)
#	eps_rel = dual tolerance for convergence (epsilon_rel)

#Output:
#	List including estimated beta (x) values and objective function.
#	Note: In optimization notation, A=X, b=Y, x=beta (minimizing x).

gflasso_admm_wes = function(D,y,x0=rep(1,length(y)),rho=1,lambda=5,maxiter=100,e_abs=1E-3,e_rel=1E-6){
				
	#Define some dimensions.								
	n = length(y)	#Number of observations in grid.
	m = nrow(D)		#Number of edges in grid.
	
	#Define function Euclidian (l2) norm of a vector.
	l2norm <- function(x) sqrt(sum(x^2))
	
	i=0					#Initialize iterator.
	converged <- 0		#Indicator for whether convergence met.
	
	#Initialize data structures.
	x = x0								#(nx1) vector to hold x values.
	r = rep(0,nrow(D))					#(mx1) vector to hold Dx values.
	
	z = matrix(0,nrow=maxiter,ncol=n)	#(nx1) vectors (as matrix) to hold slack var for x.
	s = matrix(0,nrow=maxiter,ncol=m)	#(mx1) vectors (as matrix) to hold slack var for r.
	
	u = rep(0,n)						#(nx1) dual variable.
	t = rep(0,m)						#(mx1) dual variable.
	
	#Pre-cache matrix inverse (I + DtD)^-1
	inv = crossprod(D)
	diag(inv) = diag(inv) + 1
	inv = solve(inv)
	
	#Initialize residual vectors.
	dr = 0	#Dual residual
	pr = 0	#Primal residual

	#ADMM looping.
	for (i in 2:maxiter){
		
		#Update x vector.
		x = (1/rho) * (y + rho * (z[i-1,] - u) )
		
		#Update r vector.
		r = prox_l1(s[i-1,] + t, lambda/rho)
		
		#Update z and s together.  (z first, then s = Dz. Must be done jointly.)
		#This is solving eqn (16), pg 13: (I + DtD) z = w + Dtv
		#	where w = x_k+1 + u_k
		#	and v = r_k+1 + t_k
		w = x + u
		v = r + t
		
		z[i,] = as.vector(inv %*% (w + crossprod(D,v))) #inv is pre-cached (I + DtD)^-1
		s[i,] = as.vector(D %*% z[i,])

		#Update dual variables u and t
		u = u + x - z[i,]
		t = t + r - s[i,]
		
		#Update objective function.
		#obj[i] = fx(X,y,lambda=lambda,beta=x)
		
		#--------------------------
		#Convergence check:
		
		#Calculate residuals for iteration i.
		pr = c(x - z[i,], r - s[i,])
		dr = c( rho*(z[i,]-z[i-1,]), rho*(s[i,] - s[i-1,]) )
		
		pr.norm = l2norm(pr)
		dr.norm = l2norm(dr)
		
		e.primal = sqrt(n)*e_abs + e_rel * max(l2norm(x), l2norm(z[i,])) 
		e.dual =  sqrt(n)*e_abs + e_rel * l2norm(u)
		
		if (r.norm <= e.primal && s.norm <= e.dual){
			converged=1
			break
		} #End convergence check.

	} #End ADMM loop.
	
	#Return function values.
	return(list(x=x, converged=converged, iter=i,ep=e.primal,ed=e.dual,pr.norm=pr.norm,dr.norm=dr.norm))
}
#----------------------------------------------

#Run graph-fused lasso.
gfl_wes = gflasso_admm_wes(D,y,x0=rep(1,length(y)),rho=1,lambda=5,maxiter=50)
gfl_wes.grid = Matrix(as.vector(gl$x),128,128,byrow=T)	#Reconstruct de-noised grid.
gfl_wes.grid[which(data.mat==0)]=0					#Set values which were 0 in original grid to 0.
image(t(gfl_wes.grid), sub='', xlab='',ylab='',cuts=80)


#OUTPUT IMAGES:
jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 08 LaTeX Files/gfl_slow.jpg')
image(t(gfl.grid), sub='', xlab='',ylab='',cuts=80)
dev.off()

jpeg(file='/Users/jennstarling/UTAustin/2016_Fall_SDS 385_Big_Data/Exercise 08 LaTeX Files/gfl_wes.jpg')
image(t(gfl_wes.grid), sub='', xlab='',ylab='',cuts=80)
dev.off()
