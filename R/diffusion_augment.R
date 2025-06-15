### Essential Functions needed for fitting NIFTY
compute_upiece <- function(u, W, L){ 
  n <- dim(u)[1]
  H <- dim(W)[1]
  K <- dim(u)[2]
  u_expand <- array(rep(u,each=L),c(L,n,K)) 
  u_1 <-  u_expand > array(rep(rep(c(1:L)/L, n),K),c(L,n,K))
  u_star <- u_expand > array(rep(rep(c(0:(L-1))/L, n),K),c(L,n,K))
  u_2 <- u_expand  < array(rep(rep(c(1:L)/L, n),K),c(L,n,K))
  u_piece <- u_1 / L + (u_star & u_2) * (u_expand %% (1/L))
  weighted_u_piece <- aperm(apply(array(rep(u_piece,H), c(L, n, K, H)) * aperm(array(rep(W, L*n), c(H, K, L, n)), c(3, 4, 2, 1)), c(1,2,4), sum), c(1,3,2))
  return(weighted_u_piece)
  ##output: LHN
} 

compute_eta <- function(alpha, weighted_u, intercept, n, p, H, K, L){  
  eta <- apply(array(rep(alpha, n), dim(weighted_u)) * weighted_u, c(2,3),sum) 
  return(t(eta))
} 

tr <- function(x){
  sum(diag(x))
}

compute_logprob <- function(y, K, H, Gamma, sigma2, intercept, W, alpha, u, eta){
  loglik <- -tr(t(t(y)-Gamma%*%t(eta)) %*%  diag(1/sigma2) %*% (t(y)-Gamma%*%t(eta)))    
  return(loglik)#+W_prior+alpha_prior)
}

diffusion_dimension <- function(y, eps_list,ratio){
  p <- dim(y)[2] 
  eps <- eps_list
  {
    local_cov <- array(0, c(n,p,p))
    lam <- matrix(0, n, p)
    for (i in 1:n){
      for (j in 1:n){
        if (sqrt(sum((y[i,]-y[j,])^2))<eps){
          local_cov[i,,] = local_cov[i,,]+outer(y[i,]-y[j,], y[i,]-y[j,])/n
        }
      }
      lam[i,] = svd(local_cov[i,,])$d
    } 
  }
  lam <- apply(lam,2,mean)
  for (i in 1:(length(lam)-1)){
    if (lam[i+1]/lam[i]<ratio) return(i)
  }
  return(length(lam))
}

distance_matrix <- function(y){ 
  distance = matrix(NA, n, n)
  for (i in 1:n){
    for (j in 1:n){
      distance[i,j] = sqrt(sum((y[i,]-y[j,])^2))
    }
  }
  return(distance)
}
diffusion_lam <- function(y, eps_list,ratio){ 
  p <- dim(y)[2] 
  eps <- eps_list
  {
    local_cov <- array(0, c(n,p,p))
    lam <- matrix(0, n, p)
    for (i in 1:n){
      for (j in 1:n){
        if (sqrt(sum((y[i,]-y[j,])^2))<eps){
          local_cov[i,,] = local_cov[i,,]+outer(y[i,]-y[j,], y[i,]-y[j,])/n
        }
      }
      lam[i,] = svd(local_cov[i,,])$d
    } 
  }
  lam <- apply(lam,2,mean) 
  return((lam))
}

compute_sge <- function(y, eps, q, C){
  n = dim(y)[1]
  Kernel <- matrix(0, n, n)
  K2 <- matrix(0, n, n) 
  for (i in 1:n){
    for (j in 1:n){
      dist = sum((y[i,]-y[j,])^2)
      if (dist < eps*C^2){
        Kernel[i,j] = exp(-dist/eps)}
      if (dist < 2*eps*C^2){
        K2[i,j] = exp(-dist/eps/2)}
    }}  
  D1 = diag(1/sqrt(apply(Kernel,2,sum)))
  Ksq = D1 %*%Kernel%*%D1
  D1 = diag(1/sqrt(apply(K2, 2,sum)))
  K2t = D1 %*%K2 %*%D1
  # 
  # A = matrix(0,n,n)
  # A2 = matrix(0,n,n)
  # rowsum_A = apply(Kernel,1,sum)
  # rowsum_A2 = apply(K2,1,sum)
  # for (i in 1:n){
  #   for (j in 1:n){
  #     A[i,j] = Kernel[i,j]/rowsum_A[i]/rowsum_A[j]
  #     A2[i,j] = K2[i,j]/rowsum_A2[i]/rowsum_A2[j]
  #   }
  # } 
  # #D = diag(apply(A,1,sum))
  # D1 = diag(1/(apply(A,2,sum)))
  # Ksq = ((D1)%*%A) 
  # D1 = diag(1/(apply(A2,2,sum)))
  # K2t = ((D1)%*%A2) 
  return(max(svd(Ksq%*%Ksq-K2t)$d))# sum((Ksq-K2t)**2)
}

choose_band <- function(y, band, q, Cband){
  n = dim(y)[1]
  sge <- matrix(NA, length(band), length(Cband))
  for (l in 1:length(band)){ 
    eps = band[l]
    for (m in 1:length(Cband)){
      C = Cband[m] 
      sge[l,m] = compute_sge(y, eps, q, C)
    }
  }   
  return(which(sge==min(sge), arr.ind = TRUE))
}
 
dm <- function(y,eps,q,C){ 
  n = dim(y)[1]
  Kernel <- matrix(0, n, n)
  A <- matrix(0, n, n)
  for (i in 1:n){
    for (j in 1:n){
      dist = sum((y[i,]-y[j,])^2)
      if (dist < eps*C^2){
      Kernel[i,j] = exp(-dist/eps)}
    }
  }
  rowsum_A = apply(Kernel,1,sum)
  for (i in 1:n){
    for (j in 1:n){
      A[i,j] = Kernel[i,j]/rowsum_A[i]/rowsum_A[j]
    }
  } 
  #D = diag(apply(A,1,sum))
  D1 = diag(1/(apply(A,2,sum)))
  L = ((D1)%*%A-diag(n))/eps 
  svdL <- svd(L) 
  y_embed <- svdL$u[,(n-q):(n-1)]
  return(list('y'=y_embed, 'lambda'=svdL$d[(n-q):(n-1)]))
}  


dm11 <- function(y,eps,q,C){ 
  n = dim(y)[1]
  Kernel <- matrix(0, n, n) 
  for (i in 1:n){
    for (j in 1:n){
      dist = sum((y[i,]-y[j,])^2)
      if (dist < eps*C^2){
        Kernel[i,j] = exp(-dist/eps)}
    }
  } 
  #D = diag(apply(A,1,sum))
  D1 = diag(1/(apply(Kernel,2,sum)))
  L = ((D1)%*%Kernel-diag(n))/eps 
  svdL <- svd(L) 
  y_embed <- svdL$u[,(n-q):(n-1)]
  return(list('y'=y_embed, 'lambda'=svdL$d[(n-q):(n-1)]))
}  


pre_sample_Gamma <- function(y, Gamma, eta, sigma2, W, sig_G, n, p, H, L, K){ 
  D_1 <- 1 / sig_G
  for (j in 1:p){  
    v = solve(D_1 + sum(eta[,j]^2)/sigma2[j]) 
    mean = v * t(eta[,j]) %*% y[,j]/ sigma2[j]
    Gamma[j] <-  mean + sqrt(v) * rnorm(1) 
  }  
  return(Gamma)
} 

sample_Gamma <- function(y, Gamma, eta, sigma2, sig_G, n, p, H, L, K, global, local){ 
  for (j in 1:p){ 
    sig_G = global * local[j,]  
    D_1 <- diag(1/sig_G) 
    v = solve(D_1 + t(eta)%*%eta) * sigma2[j]
    vchol = chol(v)
    mean = v %*% t(eta) %*% y[,j]/ sigma2[j]
    Gamma[j,] <-  mean +vchol %*% rnorm(H)
  }  
  return(Gamma)
}

sample_global <- function(y, Gamma, eta, sigma2, n, p, H, L, K, global, local, nu, xi){ 
  global <- 1/rgamma(1, (p*H+1)/2, rate = 1/xi + 1/2/sigma2*sum(Gamma^2/local))
  return(global)
}
sample_xi <- function(global){1/rgamma(1, 1, rate=1+1/global)}
sample_local <- function(y, Gamma, eta, sigma2, n, p, H, L, K, global, local, nu, xi){ 
  gv = as.vector(Gamma)
  nu = as.vector(nu)
  global=as.vector(global)
  local <- 1/rgamma(p*H, 1, rate=(1/nu) + gv^2/2/global/sigma2) 
  nu <- 1/rgamma(p*H,1, rate=1+1/local)
  return(list('local'=matrix(local, p, H), 'nu'=matrix(nu, p, H)))
}



sample_sigma <- function(y, Gamma, eta, sigma2, fix_sigma, a_sigma, b_sigma, n, p, H, L, K){  
  if (fix_sigma >0){
    sigma2 = rep(fix_sigma, p)
  }else{
    residual <- y - t(Gamma %*% t(eta))
    for (j in 1:p){
      sigma2[j] = 1 / rgamma(1, shape = a_sigma + n/2, rate = b_sigma + sum((residual[,j])^2/2))
    }
  } 
  return(sigma2)
}

sample_intercept_p <- function(y, intercept, Gamma, eta, sigma2, sig_i,n, p, L, K){
  posvar = 1/(n/sigma2+1/sig_i)
  #if(p==1){posvar = as.matrix(posvar)}
  posmean = posvar * (diag(1/sigma2) %*% apply(t(y-eta%*%t(Gamma)),1,sum))
   
  intercept = diag(sqrt(posvar)) %*% rnorm(p) + posmean  ##sample_W  
  return(as.vector(intercept) )
}
sample_intercept <- function(y, intercept, Gamma, eta, sigma2, sig_i,n, H, L, K){ 
  posvar = solve(n*t(Gamma) %*% diag(1/sigma2) %*% Gamma + diag(H)*(1/sig_i))
  eta_intercept = eta-matrix(rep(intercept,each=n),n,H)
  posmean = posvar %*%  t(Gamma) %*% diag(1/sigma2) %*% apply(t(y-eta_intercept%*%t(Gamma)),1,sum) 
  intercept = chol(posvar) %*% rnorm(H) + posmean  ##sample_W  
  return(intercept)
} 


  
compute_diff_a <- function(Omega, y, sig_a, alpha, eta, u_piece, Gamma, sigma2, n, p, H, K, L){
  #u_piece: LHN pH
  diff_local = aperm(array(rep(Gamma, L*n), c(p,H,L,n)), c(1,3,2,4)) * array(rep(u_piece, each=p), c(p,L,H,n))  ##plhn
  diff = apply(aperm(diff_local,c(4,1,3,2)) * array(rep(rep((y-eta%*%t(Gamma))%*% diag(1/sigma2), H),L),c(n,p,H,L)),c(3, 4),sum)   - 2 * t(Omega %*% alpha)
  t(diff)
}
#einsum('plhn,np->lh', einsum('ph,lhn -> plhn', Gamma, u_piece),(y-eta%*%t(Gamma))%*% diag(1/sigma2))
norm_vec <- function(x) sqrt(sum(x^2))

logQ <- function(x, xnew, diff, eps){
  return(-1/4/eps*(norm_vec(xnew-x-eps*diff))**2)
}

sample_a <- function(Omega, y, sig_a, count_a, alpha, eta, eps_a, u_piece, intercept, W, Gamma, sigma2, n, p, H, K, L){ 
  diff = compute_diff_a(Omega, y, sig_a, alpha, eta, u_piece, Gamma, sigma2, n, p, H, K, L)
  anew = alpha + eps_a * diff + sqrt(2*eps_a)*array(rnorm(H*L), c(L,H)) 
  anew[anew<0]=0
  eta_new = compute_eta(anew, u_piece, intercept, n, p, H, K, L)
  if (0){
    diff_new = compute_diff_a(Omega,y, sig_a, anew, eta_new, u_piece, Gamma, sigma2, n, p, H, K, L)
    log_diff = compute_logprob(y, K, H, Gamma, sigma2, intercept, W, anew, u, eta_new)-compute_logprob(y, K, H, Gamma, sigma2, intercept, W, alpha, u, eta) -
      logQ(alpha, anew, diff, eps_a) + logQ(anew, alpha, diff_new, eps_a)
    if(log(runif(1))<log_diff){ 
      count_a = count_a + 1
      alpha = anew
      eta = eta_new
    } 
  }else{
    count_a = count_a + 1
    alpha = anew
    eta = eta_new
  }
  return(list('alpha'=alpha, 'eta'=eta, 'count_a'=count_a,'diff'=diff))
}

### lambda -> lambda * alpha
compute_diff_u <- function(y, u, eta, alpha, W, Gamma, sigma2, lam, n, p, H, L, K){
  diff_u <- matrix(NA, n, K)
  for (k in 1:K){
    #u_rank =  order(u[,k])/n
    weighted_alpha =  alpha %*% diag(W[,k]) 
    weighted_Gamma =  Gamma %*% diag(W[,k]) 
    diff_local = weighted_alpha[u[,k]%/%(1/L)+1,] %*% t(weighted_Gamma) * (y-eta%*%t(Gamma))%*% diag(1/sigma2) #n*J 
    diff_u[,k] = (apply(diff_local, 1, sum))#- (u[,k] - u_rank)*lam*2
  }
  return(diff_u)
} 

# sample_u <- function(y, count_u, u, eta, eps_u, u_piece, Gamma, alpha, sigma2, lam, intercept, W, n, p, H, L, K){
#   diff = compute_diff_u(y, u, eta, alpha, W, Gamma, sigma2, lam, n, p, H, L, K)
#   if 
#   
#   return(u)
#   }

sample_u <- function(y, count_u, u, eta, eps_u, u_piece, Gamma, alpha, sigma2, lam, intercept, W, n, p, H, L, K){
  diff = compute_diff_u(y, u, eta, alpha, W, Gamma, sigma2, lam, n, p, H, L, K)
  unew = u + eps_u*diff + sqrt(2*eps_u)*matrix(rnorm(n*K),n,K)
  for (k in 1:K){
    u_rank = rank(unew[,k])/n *(max(unew)-min(unew)) + min(unew)
    unew[,k] = unew[,k] - (unew[,k] - u_rank)*lam*2
  } 
  #scale = apply(unew, 2, function(x)((max(x)-min(x))))
  unew = apply(unew, 2, function(x)((x - min(x))/(max(x)-min(x)))*.9999)
  # grid = matrix(runif(100*K),100,K)
  # unew = matrix(0,n,K)
  # for (i in 1:n){
  #   unew[i,] = grid[which.min(apply(grid, 1, function(x){sum((x-unew_[i,])^2)})),]
  # }
  #unew[unew>1]= .9999-runif(sum(unew>1))*.1
  #unew[unew<0]= 0 + runif(sum(unew<0))*.1
  u_piece_new <- compute_upiece(unew, W, L)
  eta_new = compute_eta(alpha, u_piece_new, intercept, n, p, H, K, L)
  u = unew
  u_piece = u_piece_new
  eta = eta_new
  if (0){
    diff_new = compute_diff_u(y, unew, eta_new, alpha, W, Gamma, sigma2, lam, n, p, H, L, K)
    log_diff = compute_logprob(y, K, H, Gamma, sigma2, intercept, W, alpha, unew,eta_new)-compute_logprob(y, K, H, Gamma, sigma2, intercept, W, alpha, u, eta) -
      logQ(u, unew, diff, eps_u) + logQ(unew, u, diff_new, eps_u)
    if(log(runif(1))<log_diff){ 
      count_u = count_u + 1 
      u = unew
      u_piece = u_piece_new
      eta = eta_new
    }
  }else{
    count_u = count_u + 1 
    u = unew
    u_piece = u_piece_new
    eta = eta_new
  }
  return(list('u'=u, 'u_piece'=u_piece, 'eta'=eta, 'count_u'=count_u, 'diff'=diff,'sclae'=scale))
}



sample_u_star <- function(y, y_aug, count_u, u, eta, eta_aug, eps_u, u_piece, u_piece_aug, Gamma, Gamma_aug, alpha, alpha_aug, sigma2, sigma2_aug, lam, intercept, intercept_aug, W, n, p, H, L, K){
  diff = compute_diff_u(y, u, eta, alpha, W, Gamma, sigma2, lam, n, p, H, L, K)
  diff_star = compute_diff_u(y_aug, u, eta_aug, alpha_aug, diag(K), Gamma_aug, sigma2_aug, lam, n, K, K, L, K)
  diff = diff+diff_star
  unew = u + eps_u*diff + sqrt(2*eps_u)*matrix(rnorm(n*K),n,K)
  for (k in 1:K){
    u_rank = rank(unew[,k])/n *(max(unew)-min(unew)) + min(unew)
    unew[,k] = unew[,k] - (unew[,k] - u_rank)*lam*2
  } 
  #scale = apply(unew, 2, function(x)((max(x)-min(x))))
  unew = apply(unew, 2, function(x)((x - min(x))/(max(x)-min(x)))*.9999)
  # grid = matrix(runif(100*K),100,K)
  # unew = matrix(0,n,K)
  # for (i in 1:n){
  #   unew[i,] = grid[which.min(apply(grid, 1, function(x){sum((x-unew_[i,])^2)})),]
  # }
  #unew[unew>1]= .9999-runif(sum(unew>1))*.1
  #unew[unew<0]= 0 + runif(sum(unew<0))*.1
  u_piece_new <- compute_upiece(unew, W, L)
  eta_new = compute_eta(alpha, u_piece_new, intercept, n, p, H, K, L)
  u = unew
  u_piece = u_piece_new
  eta = eta_new
  if (0){
    diff_new = compute_diff_u(y, unew, eta_new, alpha, W, Gamma, sigma2, lam, n, p, H, L, K)
    log_diff = compute_logprob(y, K, H, Gamma, sigma2, intercept, W, alpha, unew,eta_new)-compute_logprob(y, K, H, Gamma, sigma2, intercept, W, alpha, u, eta) -
      logQ(u, unew, diff, eps_u) + logQ(unew, u, diff_new, eps_u)
    if(log(runif(1))<log_diff){ 
      count_u = count_u + 1 
      u = unew
      u_piece = u_piece_new
      eta = eta_new
    }
  }else{
    count_u = count_u + 1 
    u = unew
    u_piece = u_piece_new
    eta = eta_new
  }
  return(list('u'=u, 'u_piece'=u_piece, 'eta'=eta, 'count_u'=count_u, 'diff'=diff,'sclae'=scale))
}

compute_simple_upiece <- function(u){ 
  n <- dim(u)[1]
  K <- dim(u)[2]
  u_expand <- array(rep(u,each=L),c(L,n,K)) 
  u_1 <-  u_expand > array(rep(rep(c(1:L)/L, n),K),c(L,n,K))
  u_star <- u_expand > array(rep(rep(c(0:(L-1))/L, n),K),c(L,n,K))
  u_2 <- u_expand  < array(rep(rep(c(1:L)/L, n),K),c(L,n,K))
  u_piece <- u_1 / L + (u_star & u_2) * (u_expand %% (1/L))
  return(u_piece)
}

plot_func<-function(h){
  a =  alpha[,h]
  u = matrix(seq(0,.98,.98/(n-1)),n,1)
  u_piece <- matrix(compute_simple_upiece(u),L,n)
  plot(u, t(u_piece)%*%a,  type='l',main='',xlab='',ylab='')
} 
