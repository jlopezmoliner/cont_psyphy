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

dxdt <- function(x,dt=1)
{
  
  diff(x)/dt
  
}

correlogram <- function(x,maxlag)
{
  
 res <- ccf(x[,2],x[,1],plot=F,lag.max = maxlag,na.action = na.contiguous)
 res <- data.frame(lag=as.numeric(res$lag),step=as.numeric(res$lag),ccf=as.numeric(res$acf))
 res
}

compute_ccf <- function(x,maxlag=20)
{
  ntrials <- dim(x)[2]
  res <- apply(x,2,correlogram,maxlag=maxlag)
  res2 <- do.call(rbind,res)
  res2$trial <- rep(1:ntrials,each=maxlag*2+1)
  res2
}


lqgRW.op <- function(p,x,dyn,actor,sigma_cursor=1)
{
  
  W <- diag(c(sigma_target,sigma_cursor),2,2)
  W[1,1] <- p
  dyn$W <- W
  actor$W <- W
  #  res <- -K_logLik(x=x,A = A,C = C, V=V, W=W)$loglik
  res <- -lqg(x,dyn,actor)$loglik
}
