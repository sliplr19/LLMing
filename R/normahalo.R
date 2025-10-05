normahalo <- function(z, rs, S) {
  require(pracma)

  Sij <- matrix(
    rep(NA, length(rs) * ncol(rs[[1]])),
    nrow = length(rs),
    ncol = ncol(rs[[1]])
  )

  for (i in 1:length(rs)) {
    Xstar <- as.numeric(z[i, ])
    RPi <- rs[[i]]
    mean_RPi <- colMeans(RPi, na.rm = TRUE)
    sigma <- cov(RPi)
    sigma <- sigma + diag(1e-6, ncol(sigma))
    inv_sigma <- solve(sigma)
    d    <- sum(!is.na(S[,i] & S[,i] != 0))

    numpart   <- Xstar - mean_RPi
    num  <- drop(crossprod(numpart, solve(sigma, numpart)))

    LOSi <- sqrt(num)/ d

    for (j in 1:ncol(RPi)) {
      uj <- c(rep(0, j - 1), 1, rep(0, ncol(rs[[i]]) - j))
      diff     <- as.numeric(Xstar - mean_RPi)
      num   <- abs(sum(diff * uj))
      denom <- sqrt(sum(diff^2)) * sqrt(sum(uj^2))
      if(denom != 0){
        LOSij <- (num / denom) * LOSi
      }else{
        LOSij <- 0
      }
      if (S[j, i] != 0) {
        Sij[i, j] <- LOSij
      } else {
        Sij[i, j] <- 0
      }
    }
  }

  return(Sij)
}
