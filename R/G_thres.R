G_thres = function(pCOS_mat,theta){

  thres = colMeans(pCOS_mat)*(1+theta)

  for(i in 1:ncol(pCOS_mat)){
    pCOS_mat[,i] = 1*(pCOS_mat[,i] > thres[i])

  }
  return(pCOS_mat)

}
