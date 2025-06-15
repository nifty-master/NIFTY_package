train_nifty <- function(y=y_te, y_augmented=y_augmented_te, n_iter=1000, lam = .1, W = W0, H = H0, K = K0, L = 15, step_a = 1e-1, step_u = 1e-1){
  n = length(y)
  y_star <- cbind(y,y_augmented)     
  ###hyper-parameters
  a_sigma = 1e4
  b_sigma = 1
  
  fix_sigma = FALSE
  sig_G <- 1
  sig_I <- 1
  sig_a <- 1
  
  ###Initialization 
  #y_augmented <- y[,1]
  # y <- y[, c(2,3)]
  # H <- 2
  # K <- 1
  # W <- matrix(1,2,1)
  n <- dim(y_star)[1]
  p <- dim(y)[2] 
  u <- apply(y_augmented, 2, function(x)((x - min(x))/(max(x)-min(x)))*.9999)#matrix(runif(n*K), n, K)
  alpha <- array(runif(L*H)*5, c(L, H))
  alpha_aug <- array(runif(L*K)*5, c(L, K))
  
  intercept <- runif(p) 
  intercept_aug <- runif(K)
  Gamma <- matrix(1, p, H)  
  
  nu = 1
  xi = 1
  global = .0001
  local = Gamma
  sigma2 <- as.vector(rep(1e-4, p))#apply(y,2,var) / 2 
  sigma2_aug <- as.vector(rep(1e-2, K)) 
  if (K==1){sigma2_aug = as.matrix(1e-4)}
  u_piece <- compute_upiece(u, W, L)
  eta <- compute_eta(alpha, u_piece, intercept, n, p, H, K, L)
  u_piece_aug <- compute_upiece(u, diag(K), L)
  eta_aug <- compute_eta(alpha_aug, u_piece_aug, intercept_aug, n, K, K, K, L)
  
  pb = txtProgressBar(1, n_iter, style= 3)
  count_a = 0
  count_u = 0
  u_list <- vector("list",n_iter)
  a_list <- numeric(n_iter)
  sig_list <- numeric(n_iter)
  #u <- apply(y, 2, function(x)((x - min(x))/(max(x)-min(x)))) 
  p_star <- p + K
  log_prob_list <- numeric(n_iter)
  
  Omega <- diag(L)*2
  Omega[1,1] <- 1
  Omega[L,L] <- 1
  for (i in 1:(L-1)){
    Omega[i,i+1] <- -1
    Omega[i+1,i] <- -1
  }
  eps_u = 1e-8
  eps_a = 1e-7
  Omega <- (Omega + diag(L)*.1)*10
  Gamma_aug <- (runif(K))
  if (K==1){Gamma_aug = as.matrix(Gamma_aug)}
  Gamma_list = vector("list", n_iter)
  eta_list = vector("list", n_iter)
  alpha_list = vector("list", n_iter)
  int_list = vector("list", n_iter)
  predict_list =  vector("list", n_iter)
  
  for (step in 1:n_iter){
    Gamma <- sample_Gamma(t(t(y)-intercept), Gamma, eta, sigma2, sig_G, n, p, H, L, K, global, local)
    ln <- sample_local(y, Gamma, eta, sigma2, n, p, H, L, K, global, local, nu, xi)
    local <- ln$local 
    nu <- ln$nu  
    if(step%%1000==0){
      step_u=step_u/3
      step_a=step_a/3
      }
    if (step>1000){sigma2 <- sample_sigma(y, Gamma, eta, sigma2, fix_sigma, a_sigma, b_sigma, n, p, H, L, K)}
    #print(sigma2[1], sum(((y[,1]-t(t(eta%*%t(Gamma))+intercept))[,1])**2))
    #global <- sample_global(y, Gamma, eta, sigma2, n, p, H, L, K, global, local, nu, xi)
    #xi <- sample_xi(global)
    Gamma_aug <- pre_sample_Gamma(t(t(y_augmented)-intercept_aug), Gamma_aug, eta_aug, sigma2_aug, W, sig_G, n, K, K, L, K)
    intercept <- sample_intercept_p(y, intercept, Gamma, eta, sigma2, sig_I, n, p, L, K)
    intercept_aug<- sample_intercept_p(y_augmented, intercept_aug, diag(Gamma_aug), eta_aug, sigma2_aug, sig_I, n, K, L, K)
    #intercept <- sample_intercept(y_augmented, intercept, Gamma_aug, eta, sigma2_aug, sig_I, n, H, L, K)
    eta <- compute_eta(alpha, u_piece, intercept, n, p, H, K, L)
    #eta_aug <- compute_eta(alpha_aug, u_piece_aug, intercept_aug, n, K, K, K, L)
    #a_e <- sample_a(Omega, t(t(y_star)-intercept), sig_a, count_a, alpha, eta, eps_a, u_piece, intercept, W, Gamma_star, sigma2_star, n, p_star, H, K, L) 
    a_e <- sample_a(Omega, t(t(y)-intercept), sig_a, count_a, alpha, eta, eps_a, u_piece, intercept, W, Gamma, sigma2, n, p, H, K, L) 
    a_a <- sample_a(Omega, t(t(y_augmented)-intercept_aug), sig_a, count_a, alpha_aug, eta_aug, eps_a, u_piece_aug, intercept_aug, diag(K), diag(Gamma_aug), sigma2_aug, n, K, K, K, L)
    alpha_aug <- a_a[[1]]
    eta_aug <- a_a[[2]]
    alpha <- a_e[[1]]
    eta <- a_e[[2]]
    count_a = a_e[[3]]
    #u_e <- sample_u(t(t(y_augmented)-intercept_aug), count_u, u, eta_aug, eps_u, u_piece_aug, diag(Gamma_aug), alpha_aug, sigma2_aug, lam, intercept_aug, diag(K), n, K, K, L, K)
    u_e <-sample_u_star(y, y_augmented, count_u, u, eta, eta_aug, eps_u, u_piece, u_piece_aug, Gamma, diag(Gamma_aug), alpha, alpha_aug, sigma2, sigma2_aug, lam, intercept, intercept_aug, W, n, p, H, L, K)
    u <- u_e[[1]]
    u_piece <- u_e[[2]]
    eps_u = step_u/max(abs(u_e[[5]]))
    eps_a = step_a/max(abs(a_e[[4]]))
    u_piece_aug <- compute_upiece(u, diag(K), L)
    eta <- u_e[[3]]
    eta_aug <- compute_eta(alpha_aug, u_piece_aug, intercept_aug, n, K, K, K, L)
    count_u <- u_e[[4]]
    #u_list[step] <- u[1] 
    u_list[[step]] <- u 
    a_list[step] <- alpha 
    sig_list[step] <- sigma2  
    Gamma_list[[step]] <- Gamma%*%diag(apply(eta, 2, sd)) 
    eta_list[[step]] <- eta%*%diag(1/apply(eta, 2, sd)) 
    alpha_list[[step]] <- alpha
    int_list[[step]] <- intercept
    n_te = 500
    unew = matrix(runif(n_te*K),n_te,K)
    u_piece_new = compute_upiece(unew, W, L)
    eta_new = compute_eta(alpha, u_piece_new, intercept, n_te, p, H, K, L)
    ynew = t(t(eta_new %*% t(Gamma)) + intercept)  
    predict_list[[step]] <- ynew
    log_prob_list[step] <- compute_logprob(y,K,H,Gamma,sigma2,intercept,W,alpha,u,eta)
    setTxtProgressBar(pb, step)
  }
  close(pb)  
  return(list("Gamma_list"=Gamma_list,"predict_list"=predict_list,"a_list"=a_list,"u_list"= u_list,"sig_list"=sig_list,"int_list"=int_list,"Gamma_list"=Gamma_list, "eta_list"=eta_list, "alpha_list"=alpha_list,
              "global"=global,"local"=local,
              'u' = u, 'eta'=eta, 'alpha'=alpha, 'alpha_aug'=alpha_aug,
              'intercept'=intercept, 'intercept_aug'=intercept_aug, 'Gamma'=Gamma, 'Gamma_aug'=Gamma_aug, 'sigma2'=sigma2, 'sigma2_aug'=sigma2_aug,
              'lam'=lam, 'W'=W, 'H'=H, 'K'=K, 'L'=L))
}


###Test set
run_nifty_test <- function(y=y_te, y_augmented=y_augmented_te, n_iter=1000, alpha=alpha_tr, alpha_aug=alpha_aug_tr,intercept=intercept_tr,intercept_aug=intercept_aug_tr, Gamma=Gamma_tr, Gamma_aug=Gamma_aug_tr, sigma2=sigma2_tr, sigma2_aug = sigma2_aug_tr, lam=lam0, eps_u = 1e-8, W=W0, H=H0, K=K0, L=L0){
  n = dim(y)[1]
  p = dim(y)[2]
  u <- apply(y_augmented, 2, function(x)((x - min(x))/(max(x)-min(x)))*.9999)#matrix(runif(n*K), n, K)
  u_piece <- compute_upiece(u, W, L)
  u_piece_aug <- compute_upiece(u, diag(K), L)
  eta <- compute_eta(alpha, u_piece, intercept, n, p, H, K, L) 
  
  eta_aug <- compute_eta(alpha_aug, u_piece_aug, intercept_aug, n, K, K, K, L)
  #eta_mean = matrix(0, n,H)
  
  u_list <- numeric(n_iter)
  for (step in 1:n_iter){
    u_e <- sample_u_star(y, y_augmented, 0, u, eta, eta_aug, eps_u, u_piece, u_piece_aug, Gamma, diag(Gamma_aug), alpha, alpha_aug, sigma2, sigma2_aug, lam, intercept, intercept_aug, W, n, p, H, L, K)
    u <- u_e[[1]]  
    u_piece <- u_e[[2]]
    u_piece_aug <- compute_upiece(u, diag(K), L)
    eta <- u_e[[3]] 
    eta_aug <- compute_eta(alpha_aug, u_piece_aug, intercept_aug, n, K, K, K, L)
  }
  plot(u_list)
  print(u_e[[5]])
  return(list('eta'=eta, 'Gamma'=Gamma, 'intercept'=intercept, 'u'=u))
}
