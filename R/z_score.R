z_score <- function(dat){
  z_dat <- data.frame(matrix(NA, nrow = nrow(dat), ncol = ncol(dat)))
  for(i in 1:ncol(dat)){
    z_dat[,i] <- scale(dat[,i], center = TRUE, scale = TRUE)
  }
  return(z_dat)
}
