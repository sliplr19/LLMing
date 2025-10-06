#' Thresholding of pCOS dataframe
#'
#' Converts each column of a pCOS score matrix into binary indicators
#'
#' @param pCOS_mat Dataframe of pCOS values
#' @param theta Numeric threshold
#'
#' @returns A matrix of 0s and 1s of which cells meet the threshold
#'
#' @examples
#' z_dat <- data.frame("A" = rnorm(500,0,1), "B" = rnorm(500,0,1), "C" = rnorm(500,0,1))
#' snn <- sim_SNN(z_dat, 10, 5)
#' vec_snn <- vector_SNN(z_dat, snn)
#' pCOSdat <- pCOS(z_dat, vec_snn)
#' G <- G_thres(pCOSdat, theta = 0.1)
#'
#' @export
G_thres = function(pCOS_mat,theta){

  thres = colMeans(pCOS_mat)*(1+theta)

  for(i in 1:ncol(pCOS_mat)){
    pCOS_mat[,i] = 1*(pCOS_mat[,i] > thres[i])

  }
  return(pCOS_mat)

}
