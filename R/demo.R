library(GGally)
library(ggplot2)
source("diffusion_augment.R") 
source("run_NIFTY+.R")

n <- 500
p <- 20
z1 =  rnorm(n)#
z2 =  rnorm(n)# 
y = cbind(z1, z2) %*% Gamma0  + matrix(rnorm(n*p),n,p)*.1 
train = sample(1:n, 200, replace = FALSE)
y_tr = y[train,]
y_te = y[-train,]
set.seed(123)    
q = 4
eps_band = c(1e10)
C_band = c(200)
r = choose_band(y_tr, eps_band, q, C_band)  
eps = eps_band[r[1]]
C = C_band[r[2]]
print(c(eps,C))
y_dm <- dm(y_tr, eps, q, C) 
y_embed <- y_dm$y   
 
K0 = 2
y_augmented <-  y_embed[,(q-K0+1):q]
y_augmented_tr = y_augmented  
par(mfrow=c(2,2))
plot(y_tr)
plot(y_augmented[,1],z1[train],main=seed)
plot(y_augmented[,1],z2[train])
rep_per_k <- 1
W0 <- t(matrix(rep(diag(K0), rep_per_k), nrow=K0))  
H0 = rep_per_k*K0 

param_tr <- train_nifty(step_a = .01, step_u = .01, y = y_tr, y_augmented = y_augmented_tr, W = W0, H = H0, K = K0, n_iter = 1000, L = 10, lam = 0.05)
u = param_tr$u 
eta_tr = param_tr$eta
Gamma_tr = param_tr$Gamma
Gamma_aug_tr = param_tr$Gamma_aug
intercept_tr = param_tr$intercept
intercept_aug_tr = param_tr$intercept_aug
alpha_tr = param_tr$alpha
alpha_aug_tr = param_tr$alpha_aug
sigma2_tr = param_tr$sigma2 

y_fit = t(t(eta_tr%*%t(Gamma_tr))+intercept_tr) 