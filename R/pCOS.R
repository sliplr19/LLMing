pCOS = function(z_dat, vec_SNN){

  return(mapply(FUN = pCOS_row ,
                split(as.matrix(z_dat), row(as.matrix(z_dat))),
                split(as.matrix(vec_SNN), row(as.matrix(vec_SNN)))))
}
