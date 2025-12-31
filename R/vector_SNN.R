<<<<<<< HEAD
#' Aggregate dataframe into mean feature vectors
=======
#' Aggregrate dataframe into mean feature vectors
>>>>>>> 2f77f8390366cb0a260d35d8b61399470dff6de6
#'
#' For each row of the SNN index matrix, this function takes the rows of
#' reference dataframe, z, and computes their column means, yielding one
#' mean vector per observation.
#'
#' @param z Numeric dataframe
#' @param snn Dataframe of shared nearest neighbors indices
#'
#' @returns Dataframe of same dimensions as z
#' @export
vector_SNN <- function(z, snn){
  vSNN <- list()
  for(i in 1:nrow(snn)){
    vSNN[[i]] = colMeans(z[snn[i,],])

  }
  vSNN = do.call("rbind", vSNN)
  return(vSNN)
}
