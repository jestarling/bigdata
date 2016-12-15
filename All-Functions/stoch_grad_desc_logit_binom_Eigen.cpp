#include <RcppEigen.h>
#include <algorithm>    // std::max

using namespace Rcpp;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::MatrixXi;
using Eigen::Upper;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseVector;
typedef Eigen::MappedSparseMatrix<double>  MapMatd;
typedef Map<MatrixXi>  MapMati;
typedef Map<VectorXd>  MapVecd;
typedef Map<VectorXi>  MapVeci;

typedef MapMatd::InnerIterator InIterMat;
typedef Eigen::SparseVector<double> SpVec;
typedef SpVec::InnerIterator InIterVec;

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

//Function to calculate inverse square root.
//Using this because built-in sqrt function in C++ is very slow.
//Source:  James' code.
inline double invSqrt( const double& x ) {
    double y = x;
    double xhalf = ( double )0.5 * y;
    long long i = *( long long* )( &y );
    i = 0x5fe6ec85e7de30daLL - ( i >> 1 );//LL suffix for (long long) type for GCC
    y = *( double* )( &i );
    y = y * ( ( double )1.5 - xhalf * y * y );
    
    return y;
}

//#################################################################
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]

List sparse_sgd_logit(MapMatd X, VectorXd Y, VectorXd m, double step, 
					VectorXd beta0, double lambda=0.0, int npass=1){
  // X is the design matrix stored in column-major format
  // i.e. with features for case i stores in column i
  // Y is the vector of counts
        // M is the vector of sample sizes
  // Thus Y[i] ~ Binomial( M[i], w[i])  )
  // w[i] = 1/{1+exp(- x[i] dot beta)}
  // where Beta is the regression vector we want to estimate
  // lambda is the regularization parameter

    //Inputs:
        //X: a nxp design matrix.Design matrix X
        //Y: a nx1 vector of 1's and 0's; the response vector.
        //m: a nx1 vector of binomial sample sizes.
		//discount: The discount to use for the exponential running average for the neg loglhood.
        //step: the master step size for Adagrad.
        //maxiter: maximum allowed iterations of algorithm.
        //tol: tolerance for convergence checking.
        //samps: a maxiter-length vector specifying X row sampling order.
        //lambda: the l2 regularization scalar penalty, lambda>0.
		//npass: the number of passes through the data set to take.
    //Output: A list containing the following objects.
        //beta_hat: vector of estimated beta coefficient values.
        //iter: number of iterations to converge.
        //loglik: vector of loglikelihood values for each iteration.
        //loglik_ra: vector of running avg loglikelihood for each iteration.
    //Notes:
        //Runs through URL data npass times, in order in file.  No convergence check.
        //Uses L2 regulariation, with penalty scalar lambda > 0 provided by user.
        //Uses AdaGrad algorithm.
	
	//-----------------------------------
	//INITIALIZE VALUES:
	int p = X.rows(); 		//Number of features in X matrix.
	int n = X.cols(); 		//Number of observations in X matrix.
	int iter = 0;			//Global iteraton counter.
	int j = 0;				//Inner iterator row number.
	double epsilon = 1E-6;	//Constant for Adagrad numeric stability.
	
	//Initialize vectors for beta, gradient, and doubles for Adagrad updates in sparse row loop.
	VectorXd beta_hat(p);			//Beta_hat vector, length p.
	beta_hat = beta0;				//Set beta_hat to initial beta input value.
	VectorXd hist_grad(p);		//Vector to hold running hist_grad.  Will be updated piecewise in Sparse Row Loop.
	VectorXd adj_grad(p);		//Vector to hold adj_grad_j for each beta_hat_j.
	
	//Initialize hist_grad values at 1E-3 for numerical stability.
	for (int i=0;i<p;i++){
		hist_grad(i) = 1E-3;
		adj_grad(i) = 0;
	}
	
	double grad_j = 0;				//Holds jth element of gradient.  Do not need to store whole gradient at once.
	//double adj_grad_j = 0;			//Holds jth element of adj_grad for Adagrad.  (In Sparse Row Loop)
	
	//Initialize elements to hold X, Y and m for a single observation (column).
    SparseVector<double> Xi(n);
    Xi=X.innerVector(0);
	
	double Yi = Y(0);
	double mi = m(0);
	double wi = .5;			//wi will be a scalar, since calculating weights in inner vector.
	double wi_exponent = 0;	//Holds the exponential part of the wi update.
	
	//Initialize vector (length p) to keep track of when each predictor updated, for lazy updates.
	NumericVector last_updated(p,0.0);
	double skip = 1;	//Holds how many iterations since last update for a j row of ith col.
	
	//Initialize vectors to hold log-likelihood and running avg neg log-likelihood.
	double nll = 0;								//Holds avg neg loglikelihood for the current i obs.	
	NumericVector nll_ra(n*npass,0.0);			//Stores running avg loglikelihood.
	
	//Initialize variable to hold accumulated penalty for a beta_j.
	double accum_l2_penalty = 0;	//Holds accumulated penalty.
	double gam = 0;					//Holds step*adj_grad_j, for use in calculating accumulated penalty.
	
    //-----------------------------------
	//LOOPING THROUGH DATA SET:
	
    //Loop 1: Loop over entire data set npass times.
	for (int np=0; np < npass; ++np){
		
		//Rcpp::Rcout << "npass:" << np << std::endl; //REMOVE LATER: Output start of each npass through data to R console.
				
		//Loop 2:  Loop through observations (cols) of X data set.
		for (int i=0; i<n; ++i){
		
			//Set up the single observation for the next stochastic iteration.  
			Xi = X.innerVector(i);	//Select new X observation; the ith column of matrix X.
			Yi = Y(i);				//Select new Y observation; the ith value of vector Y.
			mi = m(i);				//Select new m observation; the ith value of sample size vector m.			
		
			//Update wi probability value.  (w is scalar, since looking at one obs.)
			wi_exponent = Xi.dot(beta_hat); //breaking out exponential term helps with efficiency.
			wi = 1 / (1 + exp(-wi_exponent));
		
			//Update neg loglikelihood and running average.
			nll = -(Yi * log(wi + epsilon) + (mi - Yi) * log(1 - wi + epsilon));
			if(iter > 0) {
				nll_ra(iter) = (nll_ra(iter-1) * (iter-1) + nll) / iter;
			}			
						
			//Loop 3: Loop through active feature values (rows) of X data set for ith obs (col).
			for (InIterVec it(Xi); it; ++it){
				
				//Set j to the feature (row) number.
				j = it.index();
				
				//--------------------------------
				//STEP 1: Part 1 of Row Updates for ith Feature:  Deferred (lazy) L2 penalty updates.
				//This aggregates all of the penalty-only updates since last time a feature was updated.
				//"Penalty-only" updates refers to the gradient not being updated except
				//for adding the 2*lambda*beta penalty term.
				
				//Cap maximum number of recursive updates at 5, for numeric stability.
				//This works bc updates go to zero fairly quickly when lambda<1.
				skip = iter - last_updated(j);	//Number of iters since last update. (Skip=1 means updated last iter.)
				if (skip > 5){  skip = 5;}		
				last_updated(j) = iter;			//Update the last_updated flag for all j's active in this iter.
				
				//Calculate accum penalty.  Based on recursion defined in my notes.
				//NOTE: This is the gradient for minimizing the neg log-lhood.  
				//See final note in recursion doc.
				gam = step*adj_grad(j);
				accum_l2_penalty = beta_hat(j) * ( (1 - pow(1+lambda*gam,skip)) / (1-lambda*gam) );
				
				//Add accum l2 penalty to beta_hat_j before doing current iteration update.
				beta_hat(j) -= accum_l2_penalty; 
			
				//--------------------------------
				//STEP 2: Continue with updates for jth row in ith col.
				
				//Calculate l2 norm penalty.
				double l2penalty = 2*lambda*beta_hat(j);
				
				//Update the jth gradient term.  Note: it.value() looks up Xji for nonzero entries.
				grad_j = (mi*wi-Yi) * it.value() + l2penalty;  
				
				//Update the jth hist_grad term for Adagrad.  
				//This is the running total of the jth squared gradient term.
				hist_grad(j) += grad_j * grad_j;
				
				//Calculate the jth adj_grad term for Adagrad.
				adj_grad(j) = grad_j * invSqrt(hist_grad(j) + epsilon);
				
				//Calculate the updated jth beta_hat term.
				beta_hat(j) -= step*adj_grad(j);	
			}			
			++iter; //Update global counter.  (Counts each iteration.)
		} //End Loop 2: Loop through observations (cols) of X data set.	
	} //End Loop 1: Loop over entire data set npass times.
	
	//-----------------------------------
	//Loop 4: Loop over predictors to catch any last accumulated penalty updates
	//for predictors not updated in last iteration.
	for (int j=0; j<p; ++j){
		//Using (iter-1) since last_updated indexes from 0, and n is based on counting rows from 1.
		skip = (iter-1) - last_updated(j); 
		
		//Cap maximum number of recursive updates at 5, for numeric stability.
		//This works bc updates go to zero fairly quickly when lambda<1.	
		if (skip > 5){  skip = 5;}			
		
		//Calculate accum penalty.
		gam = step*adj_grad(j);	
		accum_l2_penalty = beta_hat(j) * ( (1 - pow(1+lambda*gam,skip)) / (1-lambda*gam) );
		
		//Update beta_j's to add accum penalty.
		beta_hat(j) -= accum_l2_penalty; 
	}	
	
	//-----------------------------------
	//Return function values.
    return Rcpp::List::create(
        _["n"] = n,
        _["p"] = p,
		_["iter"] = iter,
       	_["beta_hat"] = beta_hat,
		_["loglik"] = nll_ra
    ) ;
		
}



