vector_SNN <- function(z, snn){
  vSNN <- list()
  for(i in 1:nrow(snn)){
    vSNN[[i]] = colMeans(z[snn[i,],])

  }
  vSNN = do.call("rbind", vSNN)
  return(vSNN)
}
