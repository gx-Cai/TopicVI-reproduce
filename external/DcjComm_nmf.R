# Modified from the original code in the package DcjComm

#' The main function of DcjComm
#' @param params  
# The input params inludes input matrix X, cell-cell similarity W, dimension k1 (topic num) and k2 (Cluster number).
#' @export
#'
library('RSpectra')
library('Matrix')

soft = function( x, A ){
  if (sum( abs(cbind(A)) )==0){
    y <- x
  }else{
    y<- (abs(x) - A)
    y[y < 0] <- 0
    y <- sign(x)*y
  }
  return (y)
   
}

NMFbased = function(
    X, W, k1, k2,
    maxIter = 100
) {

  Norm <- 2
  NormV <- 1
  if (min(min(X)) < 0) {
    stop("Input should be nonnegative!")
  }
  X <- as.matrix(X)
  a <- svds(X, k1)
  U0 <- a$u
  V0 <- a$v
  S0 <- a$d
  S0 <- diag(S0)
  U0 <- abs(U0)
  V0 <- abs(V0)
  S0 <- abs(S0)
  n1 <- dim(X)[1]
  m1 <- dim(X)[2]
  b <- svds(t(V0), k2)
  B0 <- b$u
  F0 <- b$v
  B0 <- abs(B0)
  F0 <- abs(F0)
  mFea <- dim(X)[1]
  nSmp <- dim(X)[2]
  DCol <- as.matrix(apply(W, 2, sum))
  D <- sparseMatrix(dim(W)[1], dim(W)[1])
  diag(D) <- DCol

  paramsUV <- list(U = U0, V = V0, NormV = NormV, Norm = Norm)
  outUV <- Normalize(paramsUV)
  U0 <- outUV$U
  V0 <- outUV$V
  V0 <- t(V0)

  paramsBF <- list(U = B0, V = F0, NormV = NormV, Norm = Norm)
  outBF <- Normalize(paramsBF)
  B0 <- outBF$U
  F0 <- outBF$V
  F0 <- t(F0)

  Uk <- U0
  Vk <- V0
  Bk <- B0
  Fk <- F0
  Ek <- Vk
  Sk <- S0
  Tk <- matrix(0, nrow = k1, ncol = nSmp)
  iter <- 0
  converged <- 0
  tol1 <- 1e-05
  tol2 <- 1e-05
  D11 <- Vk
  QQ <- sqrt(apply(D11 * D11, 2, sum) + 1e-10)
  QQA <- 1 / QQ
  DD <- diag(QQA)
  DDD <- Uk
  QQQ <- sqrt(apply(DDD * DDD, 2, sum) + 1e-10)
  QQQA <- 1 / QQQ
  D1 <- diag(QQQA)
  rm(D11)
  rm(QQ)
  rm(QQA)
  rm(DD)
  rm(DDD)
  rm(QQQ)
  rm(QQQA)
  progressbar <- txtProgressBar(style = 3, min = 0, max = maxIter)

  while ((converged == 0) & (iter < maxIter)) {
    iter <- iter + 1
    derta <- 50
    alpha1 <- norm(X, "1") / norm(Uk, "F")
    alpha2 <- norm(Vk, "1") / norm(Fk, "F")
    Uk1 <- Uk * ((X %*% t(Vk) %*% Sk)) / (Uk %*% Sk %*% (Vk %*% t(Vk)) %*% Sk)
    AA <- t(as.matrix(sqrt(apply(Uk1 * Uk1, 1, sum))))
    Uk1 <- Uk1 / (t(matrix(AA, nrow = k1, ncol = mFea, byrow = T)))
    rm(AA)

    Sk1 <- Sk * ((t(Uk) %*% X %*% t(Vk)) / (Vk %*% t(Vk) %*% t(Uk) %*% Uk %*% Sk))
    I <- diag(k1)

    VV1 <- Sk1 %*% t(Uk1) %*% X + Bk %*% Fk + derta * Ek + Tk
    VV2 <- t(Uk1) %*% Uk1 %*% Sk1 %*% Sk1 %*% Vk + I %*% Vk + derta * Vk
    Vk1 <- Vk * (VV1 / VV2)
    rm(VV1)
    rm(VV2)
    CC <- as.matrix(sqrt(apply(Vk1 * Vk1, 1, sum)))
    Vk1 <- Vk1 / (t(matrix(CC, nrow = nSmp, ncol = k1, byrow = T)))
    rm(CC)

    Ek1 <- soft(Vk1 - Tk / derta, alpha1 / derta) + 2 * alpha1 * D1 %*% Ek
    Tk1 <- Tk + 1.618 * derta * (Ek1 - Vk1)
    EE <- as.matrix(sqrt(apply(Ek1 * Ek1, 1, sum)))
    Ek1 <- Ek1 / t((matrix(t(EE), nrow = nSmp, ncol = k1, byrow = T)))
    rm(EE)

    DD <- as.matrix(sqrt(apply(Tk1 * Tk1, 1, sum)))
    Tk1 <- Tk1 / t((matrix(t(DD), nrow = nSmp, k1, byrow = T)))
    rm(DD)

    Bk1 <- Bk * ((Vk1 %*% t(Fk)) / (Bk %*% Fk %*% t(Fk)))
    Bk1[Bk1 < 0] <- 0
    FF <- as.matrix(sqrt(apply(Bk1 * Bk1, 1, sum)))
    Bk1 <- Bk1 / t((matrix(t(FF), nrow = k2, ncol = k1, byrow = T)))
    rm(FF)

    FF1 <- t(Bk1) %*% Vk1 + alpha2 * Fk %*% W
    FF2 <- as.matrix(t(Bk1) %*% Bk1 %*% Fk) + as.matrix(alpha2 * Fk %*% D)
    Fk1 <- Fk * (FF1) / (FF2)
    Fk1 <- as.matrix(Fk1)
    rm(FF1)
    rm(FF2)
    Fk1[Fk1 < 0] <- 0
    HH <- as.matrix(sqrt(apply(Fk1 * Fk1, 1, sum)))
    Fk1 <- Fk1 / t((matrix(t(HH), nrow = nSmp, ncol = k2, byrow = T)))
    rm(HH)

    paramsBF <- list(U = Bk1, V = t(Fk1), NormV = NormV, Norm = Norm)
    outBF <- Normalize(paramsBF)
    Bk1 <- outBF$U
    Fk1 <- outBF$V
    Fk1 <- t(Fk1)
    Uwk <- Uk
    Vwk <- Vk
    Ewk <- Ek
    Bwk <- Bk
    Fwk <- Fk
    Uk <- Uk1
    Vk <- Vk1
    Bk <- Bk1
    Fk <- Fk1

    temp <- max(norm(Uk1 - Uwk, "F"), norm(Vk1 - Vwk, "F"),
                norm(Bk1 - Bwk, "F"), norm(Fk1 - Fwk, "F"))
    temp <- temp / max(norm(X, "F"))
    temp1 <- max(norm((X - Uk %*% Sk %*% Vk), "F"), norm((Vk -
                                                            Bk %*% Fk), "F")) / max(norm(Bk %*% Fk, "F"), norm(Uk %*%
                                                                                                               Sk %*% Vk, "F"))
    if ((temp1 < tol1) & (temp < tol2)) {
      converged <- 1
    }
    setTxtProgressBar(progressbar, iter)
  }
  close(progressbar)
  U_final <- Uk1
  V_final <- Vk1
  S_final <- Sk1
  B_final <- Bk1
  F_final <- Fk1
  paramsUVF <- list(U = U_final, V = t(V_final), NormV = NormV, Norm = Norm)
  outUVF <- Normalize(paramsUVF)
  U_final <- outUVF$U
  V_final <- outUVF$V
  V_final <- t(V_final)
  paramsBFF <- list(U = B_final, V = t(F_final), NormV = NormV, Norm = Norm)
  outBFF <- Normalize(paramsBFF)
  B_final <- outBFF$U
  F_final <- outBFF$V
  F_final <- t(F_final)

  Label = 0
  for (e in 1:dim(F_final)[2]) {
    v = F_final[, e]
    ma = max(v)
    s = which(v == ma)
    Label[e] = s
  }
  Label <- as.matrix(Label)

  OUT_result <- list(U_final = U_final, S_final = S_final, V_final = V_final,
                     B_final = B_final, F_final = F_final, Label = Label)
  return(OUT_result)
}


#' Normalize the input data
#' @param params  The input params.
#' @export
#'
Normalize <- function(params) {
  U <- params$U
  V <- params$V
  NormV <- params$NormV
  Norm <- params$Norm
  nSmp <- dim(V)[1]
  mFea <- dim(U)[1]

  if (Norm == 2) {
    if (NormV) {
      norms <- sqrt(apply(V * V, 2, sum))
      norms <- as.matrix(norms)
      V <- V / (matrix(t(norms), nrow = nSmp, ncol = length(norms), byrow = T))
      U <- U * (matrix(t(norms), nrow = mFea, ncol = length(norms), byrow = T))
    } else {
      norms <- sqrt(apply(U * U, 2, sum))
      norms <- as.matrix(norms)
      U <- U / (matrix(t(norms), nrow = mFea, ncol = length(norms), byrow = T))
      V <- V * (matrix(t(norms), nrow = nSmp, ncol = length(norms), byrow = T))
    }
  } else
    if (NormV) {
      norms <- apply(abs(V), 2, sum)
      norms <- as.matrix(norms)
      V <- V / (matrix(t(norms), nrow = nSmp, ncol = length(norms), byrow = T))
      U <- U * (matrix(t(norms), nrow = mFea, ncol = length(norms), byrow = T))
    } else {
      norms <- apply(abs(U), 2, sum)
      norms <- as.matrix(norms)
      U <- U / (matrix(t(norms), nrow = mFea, ncol = length(norms), byrow = T))
      V <- V * (matrix(t(norms), nrow = nSmp, ncol = length(norms), byrow = T))
    }

  Normalize_Results <- list(U = U, V = V)
  #Normalize_Results <- list(NormalizeUV_Results = NormalizeUV_Results)

  return(Normalize_Results)
  #return(V)
}

setSimilarity = function(candidateModules) {

  moduleCounts <- length(candidateModules)
  HPI <- matrix(0, moduleCounts, moduleCounts)

  for (i in 1:(moduleCounts - 1)) {
    module_i <- candidateModules[[i]]
    for (j in (i + 1):moduleCounts) {
      module_j <- candidateModules[[j]]
      if ((!(is.null(module_i)) && (!(is.null(module_j))))) {
        HPI[i, j] <- length(intersect(module_i, module_j)) / min(length(module_i), length(module_j))
      }
    }
  }
  #HPI[is.na(HPI)] <- 0
  return(HPI)
}

#' Assign genes to different clusters
#' @param moduleparams  The input params.
#' @export
#'
moduleNodesSelection = function(
  U_final,
  xita = 2
) {

  N <- dim(U_final)[1]
  K <- dim(U_final)[2]

  candidateModules <- vector('list', K)
  moduleSignal <- numeric(K)

  H_mean <- apply(U_final, 2, mean)
  H_std <- apply(U_final, 2, sd)

  for (k in 1:K) {

    candidateModules[[k]] <- which(U_final[, k] > H_mean[k] + xita * H_std[k])
    moduleSignal[k] <- mean(U_final[candidateModules[[k]], k])
  }

  HPI <- setSimilarity(candidateModules)
  modulesFinal <- candidateModules
  HPI[is.na(HPI)] <- 0
  mHPI <- sum(t(sum(HPI))) / sum(sum(HPI != 0))
  module_result <- list(modulesFinal = modulesFinal, mHPI = mHPI)

  return(module_result)

}