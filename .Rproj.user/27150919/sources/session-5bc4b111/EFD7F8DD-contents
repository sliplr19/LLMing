#' Pairwise cosine-style row score
#'
#' Given two numeric vectors, computes an average
#' cosine-based similarity.
#'
#' @param z Numeric vector
#' @param v_SNN Numeric vector, same size as z
#'
#' @returns A numeric vector
#' @export
pCOS_row = function(z, v_SNN){
  l = as.numeric(z) - as.numeric(v_SNN)
  l[l == 0] = 10^-5
  temp_mat = sqrt(pracma::kron(l^2, rep(1, length(l))) + pracma::kron(rep(1, length(l)), l^2))
  result <- abs(l) / temp_mat
  rmat = matrix(result, nrow = length(l), ncol = length(l))
  diag(rmat) = 0

  toReturn = rowSums(rmat) / (nrow(rmat) - 1)
  return(toReturn)
}

