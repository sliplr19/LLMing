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
