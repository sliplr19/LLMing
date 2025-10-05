sim_SNN <- function(z_dat, k, tops){
  require(dbscan)
  s <- tops
  SNN <- sNN(z_dat,k, jp = FALSE, sort = FALSE, search = "dist", bucketSize = 10, approx = 0)
  SNN_id <- SNN$id
  temp_neighbor <- matrix(0, ncol = nrow(SNN_id), nrow = nrow(SNN_id))
  for(i in 1:(nrow(SNN_id)-1)){
    for(j in (i+1):(nrow(SNN_id))){
      temp_neighbor[i,j] = sum(SNN_id[i,] %in% SNN_id[j,])
    }
  }
  temp_neighbor = temp_neighbor + t(temp_neighbor)
  output = t(apply(temp_neighbor,FUN =  function(x){
    return(order(x,decreasing = T)[1:s])
  }, MARGIN = 1))
  return(output)
}

