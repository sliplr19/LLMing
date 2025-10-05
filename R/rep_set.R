rep_set <- function(z, snn){
  rs <- list()
  for(i in 1:nrow(snn)){
    rs[[i]] <- data.frame(matrix(NA, nrow = ncol(snn), ncol = ncol(z)))
    for(j in 1:ncol(snn)){
      zind <- snn[i,j]
      zvec <- z[zind,]
      rs[[i]][j,] <- zvec
    }
  }
  return(rs)
}

