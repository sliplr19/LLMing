#' Evaluate Construct Validity of Text Embeddings
#'
#' Evaluates the construct validity of text embeddings by testing how well
#' embeddings predict a continuous construct score in held-out data using
#' an artificial neural network.
#'
#' The function splits observations into training and test sets, generates
#' text embeddings, standardizes the embeddings using training-set statistics,
#' fits a neural network regression model, and predicts construct scores in
#' the held-out test set.
#'
#' @param dat Dataframe containing the text column.
#' @param embed_method Embedding method: "Qwen", "NV", "e5", or "word2vec".
#' @param text_col Name of the text column in dat.
#' @param severity_col Name of the column that contains scores
#' @param seed Seed for train/test split
#' @param p Proportion in training set
#' @param embed_method Embedding method to use: "Qwen", "NV", "e5", or
#'   "word2vec".
#' @param batch_size Positive integer batch size. If NULL, method-specific
#'   defaults are used.
#' @param python_path Path to a Python executable. Ignored for word2vec.
#' @param cache_dir Hugging Face cache directory. Ignored for word2vec.
#' @param model Optional Hugging Face model ID. If NULL, uses the package
#'   default for the selected embedding method.
#' @param python_script Optional path to the Python.
#' @param clean Logical. If TRUE, apply the package's `clean_texts()` function
#'   before embedding.
#' @param word2vec_dim Embedding dimension for word2vec.
#' @param word2vec_iter Number of word2vec training iterations.
#' @param word2vec_window Context window for word2vec.
#' @param word2vec_threads Number of threads for word2vec.
#' @param verbose Logical. Print Python subprocess output.
#'
#' @return The test dataframe with an added column for predicted scores
#' @export

construct_validity <- function(
    dat,
    severity_col,
    text_col,
    seed = 973,
    p = 0.80,
    embed_method = c("Qwen", "NV", "e5", "word2vec"),
    batch_size = NULL,
    python_path = NULL,
    cache_dir = NULL,
    model = NULL,
    python_script = NULL,
    clean = TRUE,
    word2vec_dim = 50L,
    word2vec_iter = 20L,
    word2vec_window = 5L,
    word2vec_threads = 1L,
    verbose = TRUE
) {


  embed_method <- match.arg(embed_method)

  dat[[severity_col]] <- as.numeric(
    dat[[severity_col]]
  )

  dat[[text_col]] <- as.character(
    dat[[text_col]]
  )

  dat <- dat[
    !is.na(dat[[severity_col]]) &
      is.finite(dat[[severity_col]]) &
      !is.na(dat[[text_col]]) &
      nzchar(trimws(dat[[text_col]])),
    ,
    drop = FALSE
  ]

  set.seed(seed)

  idx <- caret::createDataPartition(
    y = dat[[severity_col]],
    p = p,
    list = FALSE
  )

  traindat <- as.data.frame(
    dat[idx, , drop = FALSE]
  )

  testdat <- as.data.frame(
    dat[-idx, , drop = FALSE]
  )

  traindat$doc_id <- seq_len(
    nrow(traindat)
  )

  testdat$doc_id <- seq_len(
    nrow(testdat)
  )

  x_train <- embed(
    dat = traindat,
    embed_method = embed_method,
    text_col = text_col,
    batch_size = batch_size,
    python_path = python_path,
    cache_dir = cache_dir,
    model = model,
    python_script = python_script,
    clean = clean,
    word2vec_dim = word2vec_dim,
    word2vec_iter = word2vec_iter,
    word2vec_window = word2vec_window,
    word2vec_threads = word2vec_threads,
    verbose = verbose
  )

  x_test <- embed(
    dat = testdat,
    embed_method = embed_method,
    text_col = text_col,
    batch_size = batch_size,
    python_path = python_path,
    cache_dir = cache_dir,
    model = model,
    python_script = python_script,
    clean = clean,
    word2vec_dim = word2vec_dim,
    word2vec_iter = word2vec_iter,
    word2vec_window = word2vec_window,
    word2vec_threads = word2vec_threads,
    verbose = verbose
  )

  x_train <- as.matrix(
    x_train
  )

  x_test <- as.matrix(
    x_test
  )

  x_center <- colMeans(
    x_train,
    na.rm = TRUE
  )

  x_scale <- apply(
    x_train,
    2,
    stats::sd,
    na.rm = TRUE
  )

  x_scale[
    x_scale == 0 |
      is.na(x_scale)
  ] <- 1

  x_train <- scale(
    x_train,
    center = x_center,
    scale = x_scale
  )

  x_test <- scale(
    x_test,
    center = x_center,
    scale = x_scale
  )

  y_train <- as.numeric(
    traindat[[severity_col]]
  )

  set.seed(seed)

  keras3::set_random_seed(
    seed
  )

  model_fit <- keras3::keras_model_sequential() |>
    keras3::layer_dense(
      units = 384,
      activation = "relu",
      input_shape = ncol(x_train)
    ) |>
    keras3::layer_dropout(
      rate = 0.1
    ) |>
    keras3::layer_dense(
      units = 384,
      activation = "relu"
    ) |>
    keras3::layer_dropout(
      rate = 0.1
    ) |>
    keras3::layer_dense(
      units = 1
    )

  model_fit |>
    keras3::compile(
      loss = "mse",
      optimizer = keras3::optimizer_adam(
        learning_rate = 1e-3
      ),
      metrics = "mean_absolute_error"
    )

  model_fit |>
    keras3::fit(
      x = x_train,
      y = y_train,
      epochs = 100,
      validation_split = 0.2,
      verbose = 1
    )

  pred <- model_fit |>
    stats::predict(
      x_test,
      verbose = 0
    ) |>
    drop()

  testdat$pred_severity <- as.numeric(
    pred
  )

  testdat
}
