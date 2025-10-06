#' Text anomaly score
#'
#' Text anomaly detection method adapted from \insertCite{zhang_angle-based_2015}{LLMing}.
#'
#' @param dat A dataframe with text data, one text per row
#' @param k An integer representing number of nearest neighbors
#' @param tops An integer representing how many of shared nearest neighbors to return
#' @param theta Numeric threshold
#'
#' @returns Dataframe of local outlier score
#'
#' @references
#' \insertRef{zhang_angle-based_2015}{LLMing}
#' @export
textanomaly <- function(dat, k, tops, theta){
  dat_embed <- embed(dat, 1, keep_tokens = TRUE, tokens_method = NULL)
  z_dat <- z_score(dat_embed)
  snn <- sim_SNN(z_dat, k, tops)
  vecsnn <- vector_SNN(z_dat, snn)
  pCOSout <- pCOS(z_dat, vecsnn)
  S <- G_thres(pCOSout,theta)
  rs <- rep_set(z_dat, snn)
  nm <- normahalo(z_dat, rs, S)
  return(nm)
}
