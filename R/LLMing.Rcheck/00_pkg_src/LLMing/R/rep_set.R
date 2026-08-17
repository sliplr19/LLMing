#' The vectors of the shared nearest neighbors
#'
#' Creates a list of the vectors of the top shared nearest neighbors for each row of the z dataframe
#'
#' @param z Dataframe of values of reference set
#' @param snn Dataframe of shared nearest neighbors indices
#'
#' @returns A list of dataframes where each row of the dataframe is the vector representation of a given shared nearest
#' neighbor
#' @export
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

