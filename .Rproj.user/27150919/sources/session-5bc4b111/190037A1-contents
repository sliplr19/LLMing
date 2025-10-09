#' pCOS scores for every row of dataframe
#'
#' Applies pCOS_row() to corresponding rows of two data frames,
#' returning one pCOS value per row.
#'
#' @param z_dat Numeric dataframe, typically z-scores
#' @param vec_SNN Numeric dataframe, typically the output of vector_SNN
#'
#' @returns A dataframe with same dimensions as z_dat
#' @export
pCOS = function(z_dat, vec_SNN){

  return(mapply(FUN = pCOS_row ,
                split(as.matrix(z_dat), row(as.matrix(z_dat))),
                split(as.matrix(vec_SNN), row(as.matrix(vec_SNN)))))
}
