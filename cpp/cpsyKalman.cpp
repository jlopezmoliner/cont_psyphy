#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma; 

/* 
 * Code in C++ developed by Joan López-Moliner
 */

  /* Dimensions in Kalman. */
  /*
  size_t l;           Dimension of input vector (number of control signals). Can be zero for autonomous systems. 
  size_t m;           Dimension of output vector (number of measured values). Can be zero for prediction only.
  size_t n;           Dimension of state vector 
   System matrices:
  arma::mat A;        State matrix with dimensions n x n 
  arma::mat B;        Input matrix with dimensions n x l 
  arma::mat H;        Output matrix with dimensions m x n 
  
  arma::mat Q;        Process covariance matrix with dimensions n x n 
  arma::mat R;        Measurement covariance matrix with dimensions m x m 
  // Identity matrix
  arma::mat I;        Identity matrix with dimensions n x n 
  arma::mat S;        Innovation covariance matrix with dimensions m x m 
  
  // Kalman gain matrix:
  arma::mat Kg;       Kalman gain; matrix with dimensions n x m 
  arma::mat pest;       Estimate covariance matrix with dimensions n x n 
  arma::mat xpred;   predicted stated 
  arma::vec xest;    Estimated state vector with dimension n 
  arma::vec u;       Input vector with dimension l 
  arma::colvec z;    Input vector with dimension m 
  */

/* cpsy_get_nll_: implements simple model Bonnen at al 2015.  
 It is not for general purposes:  use K_logLik instead for more general models
*/  
  
// [[Rcpp::export]]
arma::mat cpsy_get_nll_(const arma::mat &x, const arma::mat &xhat,const arma::mat &q, const arma::mat &r)
  {
    /* 
    x: matrix target motion: rows: samples; cols: trials
    xhat: matrix tracking response: rows: samples; cols: trials
    q: matrix (known) process noise (n x n, n= number of state variables) 
    r: matrix (provided as log(r))  (to be estimated) measurement noise (m x m, m=number of measured variables)
  */
    
    // x= trials are cols 
    arma::mat nLL=zeros(1,1);
    size_t numTrials = x.n_cols;
    size_t N = x.n_rows; // number of measurements
    
    mat rr = exp(r);
    
    // general solution below to compute Kg:
    // s = h%*%q%*%t(h)+r // h = m x n 
    // Kg =  q * H.t() * inv(s) 
    
    mat pest = q/2 * (sqrt(1+4*rr/q)-1); // Posterior variance
    mat kg = (pest+q)/(pest+q+rr); /* Kalman gain; matrix with dimensions n x m */ 
    double k = as_scalar(kg);
    
    mat d = eye(N,N);
    uvec lower_indices = trimatl_ind( size(d),-1);
    d.elem(lower_indices).fill(k-1);
    lower_indices = trimatl_ind( size(d),-2);
    d.elem(lower_indices).fill(0.0);
    
    for(size_t i=0; i<numTrials; i++){
      colvec x1 = x.col(i); // N(samples) x 1
      colvec xhat1 = xhat.col(i);
      mat temp = d*xhat1 - (k*x1); // eq B11
      mat tt = (-1/(2*pow(k,2)*rr)*temp.t()*temp - N/2*log(rr) - N*log(k)); 
      nLL = nLL - tt; // the negative log likelihood that (D*xhat - K*x) ~ N(0,K^2*R) 
    }
    
    return nLL;
    
  }

// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  // sigma is the covariance matrix
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}


// [[Rcpp::export]]
arma::mat simulSequence(const arma::mat &A, const arma::mat &C, const arma::mat &V, const arma::mat &W, const rowvec &x0, const rowvec &xhat0, const size_t &T)
{
  
  arma::vec M(A.n_rows, fill::zeros);
  arma::vec Z(C.n_rows, fill::zeros);
  
  
//  arma::mat epsilon=mvnrnd(M, V*V.t(), T);
//  arma::mat eta=mvnrnd(Z, W*W.t(), T);
  arma::mat epsilon=mvrnormArma(T, M, V*V.t()); // V is sigma
  arma::mat eta=mvrnormArma(T,Z, W*W.t());// W is sigma
  arma::mat kalman_gain_(arma::mat A, arma::mat C, arma::mat V, arma::mat W, size_t T);
  
  
  arma::mat X(T,A.n_rows);
  arma::mat Xhat(T,C.n_rows);
  
  arma::mat K = kalman_gain_(A,C,V*V.t(),W*W.t(), T);
  
  X(0,span())=x0;
  Xhat(0,span())=xhat0;
  
//  X(0,span())= X(0,span()) + join_cols(epsilon(0,span()),eta(0,span()));
  for(size_t i=1;i<T;i++){
    vec xhat_prev = Xhat.row(i - 1).t();
    vec x_prev = X.row(i - 1).t();
    //(A*X(span(i-1),span()).t() + epsilon(i)).t();
     X.row(i) = (A*x_prev).t() + epsilon.row(i-1);// Attention: epsilon & eta after .t()+
    // generate observation
    arma::mat y =  (C*X.row(i).t()).t() + eta.row(i-1); //1x2
 //   printf("%d %d\n",y.n_rows,y.n_cols);
    // update belief
    arma::mat xpred = A*xhat_prev; // Xhat column vector, also xpred
    // Xhat(i,span()) = (C*X(i,span()).t() + eta(i)).t();
     Xhat.row(i) = (xpred + (K*(y.t()- C*xpred))).t();
  } 
  
  arma::mat Res = join_horiz(X,Xhat);
  if(A.n_cols==2)
    Res.swap_cols(1,2); // interleave state and its estimate
  
  return Res;
    
}
//R>> simulSequence(Am,C,V,W,x0 = c(-0.14,0.0),xhat0 = c(-0.14,0),T=1500)

/* K_logLik: returns Kalman gain and logLik of the Kalman model */

// [[Rcpp::export]]
List K_logLik(const arma::cube &x,const arma::mat &A, const arma::mat &C, const arma::mat &V, const arma::mat &W)
{
  /* 
   x: matrix rows: samples; cols: trials; slices: dim [states interleaved with their estimates]
   A: system transition matrix
   C: observation matrix
   V: matrix (known) process noise (n x n, n= number of state variables) SD
   W: matrix (provided as SD)  (to be estimated) measurement noise (m x m, m=number of measured variables)
   */
  arma::mat kalman_gain_(arma::mat A, arma::mat C, arma::mat V, arma::mat W, size_t T);
  arma::vec dmvnrm_arma_fast(arma::mat const &x,  
                             arma::rowvec const &mean,  
                             arma::mat const &sigma, 
                             bool const logd = false);
  arma::mat F;
  arma::mat G;
  
  // x= trials are cols 
  arma::mat Az=A;
  Az.zeros();
  arma::mat Vz=V;
  Vz.zeros();
  
  size_t T = x.n_rows; // number of samples
  size_t n = x.n_cols; // number of trials
  size_t d = x.n_slices;// number of dimensions (state+estimates)  

  arma::cube mu(T-1,n,d);
  arma::cube Sigma(T-1,d,d);
  arma::mat mu_(n,d);
  arma::mat Sigma_(d,d);
  arma::mat llik(T-1,n);
  
  arma::mat K = kalman_gain_(A,C,V*V.t(),W*W.t(), T);
  
  // Compute logLik
  
  F = join_cols(join_rows(A,Az),join_rows(K*C*A,A-K*C*A)); 
  //cout << F << "\n";
  if(d==4){
    F.swap_cols(1,2);
    F.swap_rows(1,2);
  }
  
  G = join_cols(join_rows(V,Vz),join_rows(K*C*V,K*W));
  if(d==4){
    G.swap_cols(1,2);
    G.swap_rows(1,2);
  }
  
  mu_.zeros(n,d);
  Sigma_ = G * G.t();
  
  for(size_t i=0;i<T-1;i++){
    arma::mat xt = x(span(i),span(),span());

    mu_ = mu_ * F.t() + (xt - mu_) * inv(Sigma_).t() * (F * Sigma_).t(); 
    Sigma_ = F * Sigma_ * F.t() + G * G.t() - (F*Sigma_) * inv(Sigma_) * (Sigma_ * F.t());
    mu.row(i)=mu_;
    Sigma.row(i)=Sigma_;
    
  }
  
  // Actually compute the log-densities
  for(size_t i=0;i<T-1;i++){
    for(size_t j=0;j<n;j++){
      rowvec m = mu(span(i),span(j),span());
      arma::vec a = dmvnrm_arma_fast(x(span(i+1),span(j),span()),m,Sigma.row(i),true);
      llik(i,j)=a(0);
    }
  }
  
  List L = List::create(Named("K") = K, _["loglik"] = accu(llik));
  return L;
  
}


// dmvnrm_arma_fast as defined in:
// https://gallery.rcpp.org/articles/dmvnorm_arma/
  
static double const log2pi = std::log(2.0 * M_PI);

void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;
  
  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}

// [[Rcpp::export]]
arma::vec dmvnrm_arma_fast(arma::mat const &x,  
                           arma::rowvec const &mean,  
                           arma::mat const &sigma, 
                           bool const logd = false) { 
  using arma::uword;
  uword const n = x.n_rows, 
    xdim = x.n_cols;
  
  arma::vec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())), 
    constants = -(double)xdim/2.0 * log2pi, 
    other_terms = rootisum + constants;
  
  arma::rowvec z;
  for (uword i = 0; i < n; i++) {
    z = (x.row(i) - mean);
    inplace_tri_mat_mult(z, rooti);
    out(i) = other_terms - 0.5 * arma::dot(z, z);     
  }  
  
  if (logd)
    return out;
  return exp(out);
}

// implementation of discrete-time Riccati equation
// to compute K (kalman gain) and controller L 
arma::mat riccati(arma::mat A, arma::mat B, arma::mat Q,arma::mat R,size_t T)
{
  
  arma::mat S = Q;
  for(size_t i=0; i<T; i++){
    S = A.t() * (S - S *  B *  inv(B.t() * S * B + R) * B.t() * S) * A + Q;
  }
  
  return S;
}

arma::mat kalman_gain_(arma::mat A, arma::mat C, arma::mat V, arma::mat W, size_t T)
{
  arma::mat riccati(arma::mat A, arma::mat B, arma::mat Q,arma::mat R,size_t T);
  arma::mat P = riccati(A.t(), C.t(), V, W, T);
  
  arma::mat S = C * P * C.t() + W;
  arma::mat K = P * C.t() * inv(S);
  return K;
  
}


// Args:
// A (arma::mat): state transition matrix
// B (arma::mat): control matrix
// Q (arma::mat): control costs
// R (arma::mat): action costs
// T (size_t): time steps

arma::mat control_law_(arma::mat A, arma::mat B, arma::mat Q, arma::mat R, size_t T)
{
  arma::mat riccati(arma::mat A, arma::mat B, arma::mat Q, arma::mat R, size_t T);
  
  arma::mat S = riccati(A, B, Q, R, T);

  arma::mat L = inv(B.t()*S*B+R) * B.t()*S*A;
  return L;
    
}

// ax <- 0.8
// az <- 0.3
// 
// q <- diag(c(ax/sqrt(2))**2,1,1)
//   r <- 10*diag(c(0.07)**2,1,1)
//   
//   x1 <- matrix(rnorm(100000),nrow = 100,ncol=1000)
//   r1 <- x1 + runif(20,-0.1,0.1)
//   cpsy_get_nll_(t(x1),t(r1),q,log(r))
  