RW.op <- function(p,x,A=matrix(1,1,1),C=matrix(1,1,1),V=matrix(1,1,1))
{
  
  W <- diag(p,1,1) # 
  res <- -K_logLik(x=x,A = A,C = C, V=V, W=W)$loglik
  
}

PosVel.op <- function(p, x, A= matrix(c(1,0,1/90,1),2,2), C = diag(1,2,2), V=diag(c(0.01,0.00001),2,2))
{
  
  W <- diag(abs(p),2,2) # guess for sensory noise
  res <- -K_logLik(x=x,A = A,C = C, V=V, W=W)$loglik
  
}

lqgRW.op <- function(p,x,dyn,actor,sigma_cursor=1)
{
  
  W <- diag(c(sigma_target,sigma_cursor),2,2)
  W[1,1] <- p # p[1] p[2]
  #W[2,2] <- p[2]
  dyn$W <- W
  actor$W <- W
  #  res <- -K_logLik(x=x,A = A,C = C, V=V, W=W)$loglik
  res <- -lqg(x,dyn,actor)$loglik
}


dxdt <- function(x,dt=1)
{
  
  diff(x)/dt
  
}

correlogram <- function(x,maxlag,deriv)
{
  if (deriv) {
    x <- apply(x, 2, diff)
  }
 res <- ccf(x[,2],x[,1],plot=F,lag.max = maxlag,na.action = na.contiguous)
 res <- data.table(lag=as.numeric(res$lag),ccf=as.numeric(res$acf))
 res
}

compute_ccf <- function(x,maxlag=100,deriv=TRUE)
{
  ntrials <- dim(x)[2]
  res <- apply(x,2,correlogram,maxlag=maxlag,deriv=deriv)
  res2 <- do.call(rbind,res)
  res2$trial <- rep(1:ntrials,each=maxlag*2+1)
  res2
}


