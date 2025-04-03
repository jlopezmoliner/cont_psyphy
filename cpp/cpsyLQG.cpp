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
  
#define A_ind 0
#define B_ind 1
#define C_ind 2
#define V_ind 3
#define W_ind 4
#define Q_ind 5
#define R_ind 6
  
// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  // sigma is the covariance matrix
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

// [[Rcpp::export]]
arma::mat lqg_simSequence(List dyn, List actor, const rowvec &x0, const rowvec &xhat0, const size_t &T, const bool return_all=false)
{
  
  arma::mat kalman_gain_(arma::mat A, arma::mat C, arma::mat V, arma::mat W, size_t T);
  arma::mat control_law_(arma::mat A, arma::mat B, arma::mat Q, arma::mat R, size_t T);
  
  const arma::mat a = as<arma::mat>(dyn["A"]);
  const arma::mat b = as<arma::mat>(dyn["B"]);
  const arma::mat c = as<arma::mat>(dyn["C"]);
  const arma::mat v = as<arma::mat>(dyn["V"]);
  const arma::mat w = as<arma::mat>(dyn["W"]);
  
  const arma::mat A = as<arma::mat>(actor["A"]);
  const arma::mat B = as<arma::mat>(actor["B"]);
  const arma::mat C = as<arma::mat>(actor["C"]);
  const arma::mat V = as<arma::mat>(actor["V"]);
  const arma::mat W = as<arma::mat>(actor["W"]);
  const arma::mat Q = as<arma::mat>(actor["Q"]);
  const arma::mat R = as<arma::mat>(actor["R"]);
  
  
  arma::vec M(a.n_rows, fill::zeros);
  arma::vec Z(c.n_rows, fill::zeros);
  arma::mat Res;
  arma::mat X(T,a.n_rows); 
  arma::mat Xhat(T,A.n_rows);// actor
  arma::mat u(T,B.n_cols);
  arma::mat epsilon=mvrnormArma(T, M, v*v.t()); // V is sigma
  arma::mat eta=mvrnormArma(T,Z, w*w.t());// W is sigma
  arma::mat K = kalman_gain_(A,C,V*V.t(),W*W.t(), T);
  arma::mat L = control_law_(A, B, Q, R, T); // row vector
  // K.print();
  // L.print();
//   w.print();
  X(0,span())=x0;
  Xhat(0,span())=xhat0; // 
  
  for(size_t i=1;i<T;i++){
    vec xhat_prev = Xhat.row(i - 1).t();
    vec ut = -L * xhat_prev;
    
    X(i,span()) = (a*X(span(i-1),span()).t() + b*ut ).t() + epsilon.row(i-1);
    // generate observation
    arma::mat y =  (c*X(i,span()).t() ).t() + eta.row(i-1); //1x2
    
    // update belief
     arma::mat xpred = A*xhat_prev + B*ut; // Xhat column vector, also xpred
      Xhat.row(i) = (xpred + (K*(y.t()- C*xpred))).t();
     u.row(i) = ut;
  } 
  
  if(return_all)
    Res = join_horiz(X,Xhat,u); // we do not interleave
  else
    Res = X;
//  colnames(Res) = CharacterVector::create("x1", "x2", "xhat1","xhat2");

  return Res;
  // List out = List::create(Named("x") = X, _["xhat"] = Xhat, _["u"] = u);
  // return out;
  
}
//R>> simulSequence(Am,C,V,W,x0 = c(-0.14,0.0),xhat0 = c(-0.14,0),T=1500)

/* lqg: returns Kalman gain, control gain (L) and logLik of the lqg model */

// [[Rcpp::export]]
List lqg(const arma::cube &x, List dyn, List actor)
{
  /* 
   x: matrix rows: samples/time; cols: trials; slices: dim [states + their estimates]
   dyn: list with true matrices of the world:
      A: system transition matrix
      B: input matrix
      C: observation matrix
      V: matrix (known) process noise (n x n, n= number of state variables) as SD
      W: matrix (provided as SD)  (to be estimated) measurement noise (m x m, m=number of measured variables)
   actor: 
      A: same as above or subjective transition matrix
      B: input matrix
      C: observation matrix
      V: matrix (known) process noise (n x n, n= number of state variables) as SD
      W: matrix (provided as SD)  (to be estimated) measurement noise (m x m, m=number of measured variables)
      Q: state cost function matrix
      R: action cost
   */
  
  arma::mat kalman_gain_(arma::mat A, arma::mat C, arma::mat V, arma::mat W, size_t T);
  arma::mat control_law_(arma::mat A, arma::mat B, arma::mat Q, arma::mat R, size_t T);
  // arma::vec dmvnrm_arma_fast(arma::mat const &x,  
  //                            arma::rowvec const &mean,  
  //                            arma::mat const &sigma, 
  //                            bool const logd = false);
  // double llik(const arma::cube &x, const arma::mat &K, const arma::mat &L, const arma::mat &A, const arma::mat &B, const arma::mat &C, const arma::mat &a, const arma::mat &b, const arma::mat &c, const arma::mat &v, const arma::mat &w);
  List llik(const arma::cube &x, const arma::mat &K, const arma::mat &L, const arma::mat &A, const arma::mat &B, const arma::mat &C, const arma::mat &a, const arma::mat &b, const arma::mat &c, const arma::mat &v, const arma::mat &w);
    
  const arma::mat a = as<mat>(dyn["A"]);
  const arma::mat b = as<mat>(dyn["B"]);
  const arma::mat c = as<mat>(dyn["C"]);
  const arma::mat v = as<mat>(dyn["V"]);
  const arma::mat w = as<mat>(dyn["W"]);
  
  const arma::mat A = as<mat>(actor["A"]);
  const arma::mat B = as<mat>(actor["B"]);
  const arma::mat C = as<mat>(actor["C"]);
  const arma::mat V = as<mat>(actor["V"]);
  const arma::mat W = as<mat>(actor["W"]);
  const arma::mat Q = as<mat>(actor["Q"]);
  const arma::mat R = as<mat>(actor["R"]);
  
  
  
  // x= trials are cols 
  // arma::mat Az=A;
  // Az.zeros();
  // arma::mat Vz=V;
  // Vz.zeros();
  
  size_t T = x.n_rows; // number of samples
//  size_t n = x.n_cols; // number of trials
//  size_t d = x.n_slices;// number of dimensions (state+estimates)  
  
  arma::mat K = kalman_gain_(A,C,V*V.t(),W*W.t(), T);
  arma::mat L = control_law_(A, B, Q, R, T);
  
  // inici move
  
  // double ll = llik(x, K, L, A, B, C, a, b, c, v, w); // compute the log lik
  List rL = llik(x, K, L, A, B, C, a, b, c, v, w); // compute the log lik
  // end move
//  List L = List::create(Named("K") = K, _["loglik"] = accu(llik));
//  List out = List::create(Named("K") = K, _["L"] = L, _["loglik"] = accu(llik));
  double ll=as<double>(rL["loglik"]);
  List out = List::create(Named("K") = K, _["L"] = L, _["loglik"] = ll, _["LL"] = as<mat>(rL["LL"]));
  return out;
    
}

// computes the log lik
//double llik(const arma::cube &x, const arma::mat &K, const arma::mat &L, const arma::mat &A, const arma::mat &B, const arma::mat &C, const arma::mat &a, const arma::mat &b, const arma::mat &c, const arma::mat &v, const arma::mat &w)
List llik(const arma::cube &x, const arma::mat &K, const arma::mat &L, const arma::mat &A, const arma::mat &B, const arma::mat &C, const arma::mat &a, const arma::mat &b, const arma::mat &c, const arma::mat &v, const arma::mat &w)
{
//  double ll; 
  arma::vec dmvnrm_arma_fast(arma::mat const &x,  
                             arma::rowvec const &mean,  
                             arma::mat const &sigma, 
                             bool const logd = false);
  
  size_t T = x.n_rows; // number of samples
  size_t n = x.n_cols; // number of trials
  size_t d = x.n_slices;// number of dimensions (state+estimates)  
  
  arma::mat F;
  arma::mat G;
  
  arma::mat Ctz=c.t();
  Ctz.zeros();
  
  arma::cube mu(T-1,n,a.n_rows+a.n_rows);
  arma::cube Sigma(T-1,a.n_rows+a.n_rows,a.n_rows+a.n_rows);
  arma::mat mu_(n,a.n_rows+a.n_rows);
  arma::mat Sigma_(a.n_rows+a.n_rows,a.n_rows+a.n_rows);
  arma::mat llik(T-1,n);
  
  // hstack is join_rows in armadillo
  // vstack is join_cols in armadillo
  
  // Compute logLik 
  
  F = join_cols(join_rows(A,-b*L),join_rows(K*c*a,A-B*L-K*C*A));  // d+dxd+d
  
  // G = jnp.vstack([jnp.hstack([self.dynamics.V, jnp.zeros_like(self.dynamics.C.T)]),
  //                jnp.hstack([K @ self.dynamics.C @ self.dynamics.V, K @ self.dynamics.W])])
  
  G = join_cols(join_rows(v,Ctz),join_rows(K*c*v,K*w));  
  
  mu_.zeros(n,a.n_rows+a.n_rows);
  Sigma_ = G * G.t();
  //  Sigma_.print();
  
  
  for(size_t i=0;i<T-1;i++){
    arma::mat xt = x(span(i),span(),span());
    
    arma::mat aux1=(F*Sigma_);
    mu_ = mu_ * F.t() + (xt - mu_(span(),span(0,d-1))) * inv(Sigma_(span(0,d-1),span(0,d-1))).t() *  aux1(span(),span(0,d-1)).t(); 
    arma::mat aux2=(Sigma_*F.t());
    Sigma_ = F * Sigma_ * F.t() + G * G.t() - aux1(span(),span(0,d-1)) * inv(Sigma_(span(0,d-1),span(0,d-1))) * aux2(span(0,d-1),span());
    mu.row(i)=mu_;
    Sigma.row(i)=Sigma_;
  }
  
  // Actually compute the log-densities
  for(size_t i=0;i<T-1;i++){
    for(size_t j=0;j<n;j++){
      rowvec m = mu(span(i),span(j),span(0,d-1));
      arma::mat sig1 = Sigma.row(i);
      // sig.print();
      arma::vec a = dmvnrm_arma_fast(x(span(i+1),span(j),span()),m,sig1(span(0,d-1),span(0,d-1)),true);
      llik(i,j)=a(0);
    }
    // printf("%f\n",llik(i,0));
  }
  
  // return accu(llik);
  List out = List::create(Named("LL") = llik, _["loglik"] = accu(llik));
  return out;
}


// [[Rcpp::export]]
List maxLLik(const arma::cube &x, const List &dyn, const List &actor, const arma::vec &p, const arma::vec &v)
{
  List modifyMatrixInList(List matrixList, int matrixIndex, int row, int col, double value);
  List lqg(const arma::cube &x, List dyn, List actor);
  double maxKValue = -std::numeric_limits<double>::infinity(); // Initialize with negative infinity
  double maxP = 0.0; // Variable to store the corresponding value of p
  double maxV = 0.0; // Variable to store the corresponding value of v
  
  size_t n = p.size();  
  size_t m = v.size();  
  List d = Rcpp::clone(dyn); // Create a deep copy of dyn list
  List a = Rcpp::clone(actor); // Create a deep copy of actor list
  List q;
  
  for(size_t i=0;i<n;i++){
    for(size_t j=0;j<m;j++){
            
        d=modifyMatrixInList(d,W_ind,0,0,p(i));
        d=modifyMatrixInList(d,W_ind,1,1,v(j));
        a=modifyMatrixInList(a,W_ind,0,0,p(i));
        a=modifyMatrixInList(a,W_ind,1,1,v(j));
        q=lqg(x,d,a);
        double kValue = as<double>(q["loglik"]);  
        if (kValue > maxKValue) {
          maxKValue = kValue; // Update the maximum value if a higher value is encountered
          maxP = p(i); // Store the corresponding value of p
          maxV = v(j); // Store the corresponding value of v
          
        }
        
       // std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1) << kValue << std::endl;
    }
  }
  
  // std::cout << "Maximum kValue: " << maxKValue << std::endl;
  // std::cout << "Corresponding p value: " << maxP << std::endl;
  // std::cout << "Corresponding v value: " << maxV << std::endl;
  
  
   // Create a List object to store the results
  List result;
  result["pos"] = maxP;
  result["vel"] = maxV;
  result["loglik"] = maxKValue;
  
  return result; // Return the List

}

// modif
List modifyMatrixInList(List matrixList, int matrixIndex, int row, int col, double value) {
  // Access the matrix within the list using matrixIndex
  arma::mat matrix = as<arma::mat>(matrixList[matrixIndex]);
  
  // Modify the desired element in the matrix
  matrix(row, col) = value;
  // Update the modified matrix back into the list
  matrixList[matrixIndex] = matrix;
  // Return the modified matrix list
  return matrixList;
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

// [[Rcpp::export]]
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
  