#' Generate document embeddings
#'
#' Generate one embedding vector per row of a data frame using Qwen,
#' NV-Embed, E5, or a locally trained word2vec model.
#'
#' @param dat Dataframe containing the text column.
#' @param embed_method Embedding method: "Qwen", "NV", "e5", or "word2vec".
#' @param text_col Name of the text column in dat.
#' @param batch_size Positive integer batch size. If NULL, method-specific
#'   defaults are used.
#' @param python_path Path to a Python executable. If NULL, searches for
#'   "python3" or "python". Ignored for word2vec.
#' @param cache_dir Hugging Face cache directory. If NULL, uses the package
#'   user cache directory. Ignored for word2vec.
#' @param model Optional Hugging Face model ID. If NULL, uses the package
#'   default for the selected embedding method.
#' @param clean Logical. If TRUE, apply the package's `clean_texts()` function
#'   before embedding.
#' @param word2vec_dim Embedding dimension for word2vec.
#' @param word2vec_iter Number of word2vec training iterations.
#' @param word2vec_window Context window for word2vec.
#' @param word2vec_threads Number of threads for word2vec.
#' @param python_script Path to Python embedding script. Defaults to package script.
#' @param verbose Logical. Print Python subprocess output.
#'
#' @return A dataframe with one row per input row and one column per
#'   embedding dimension.
#' @export

embed <- function(
    dat,
    embed_method = c("Qwen", "NV", "e5", "word2vec"),
    text_col,
    batch_size = NULL,
    python_path = NULL,
    cache_dir = NULL,
    model = NULL,
    python_script = "embed.py",
    clean = TRUE,
    word2vec_dim = 50L,
    word2vec_iter = 20L,
    word2vec_window = 5L,
    word2vec_threads = 1L,
    verbose = TRUE
)  {



  if (!is.data.frame(dat)) {
    stop("`dat` must be a data.frame.", call. = FALSE)
  }

  if (!is.character(text_col) || length(text_col) != 1L ||
      is.na(text_col) || !nzchar(text_col)) {
    stop("`text_col` must be one non-empty character string.", call. = FALSE)
  }

  if (!text_col %in% names(dat)) {
    stop("Missing text column: ", text_col, call. = FALSE)
  }

  embed <- match.arg(embed)

  if (isTRUE(clean)) {
    texts_clean <- clean_texts(dat = dat, text_col = text_col)
  } else {
    texts_clean <- dat[[text_col]]
  }

  texts_clean <- as.character(texts_clean)
  texts_clean[is.na(texts_clean)] <- ""
  texts_clean <- trimws(texts_clean)
  texts_clean[texts_clean == ""] <- "empty text"

  if (embed == "word2vec") {


    int_args <- list(
      word2vec_dim = word2vec_dim,
      word2vec_iter = word2vec_iter,
      word2vec_window = word2vec_window,
      word2vec_threads = word2vec_threads
    )

    for (nm in names(int_args)) {
      x <- int_args[[nm]]
      if (length(x) != 1L || is.na(x) || x < 1 || x != as.integer(x)) {
        stop("`", nm, "` must be one positive integer.", call. = FALSE)
      }
    }

    texts_clean[texts_clean == "empty text"] <- "emptytext"

    word_model <- word2vec::word2vec(
      x = texts_clean,
      type = "skip-gram",
      dim = as.integer(word2vec_dim),
      iter = as.integer(word2vec_iter),
      min_count = 1L,
      window = as.integer(word2vec_window),
      threads = as.integer(word2vec_threads)
    )

    word_matrix <- as.matrix(word_model)

    if (is.null(dim(word_matrix)) ||
        nrow(word_matrix) == 0L ||
        ncol(word_matrix) == 0L) {
      stop("word2vec produced an empty word matrix.", call. = FALSE)
    }

    if (is.null(rownames(word_matrix))) {
      stop(
        "word2vec matrix has no row names; tokens cannot be matched to vectors.",
        call. = FALSE
      )
    }

    storage.mode(word_matrix) <- "double"
    word_matrix[!is.finite(word_matrix)] <- 0

    document_tokens <- strsplit(texts_clean, "\\s+")

    document_embeddings <- lapply(document_tokens, function(tokens) {
      tokens <- tokens[
        !is.na(tokens) &
          tokens != "" &
          tokens %in% rownames(word_matrix)
      ]

      if (length(tokens) == 0L) {
        return(rep(0, ncol(word_matrix)))
      }

      vector <- colMeans(
        word_matrix[tokens, , drop = FALSE]
      )

      vector[!is.finite(vector)] <- 0
      vector
    })

    output <- do.call(rbind, document_embeddings)
    storage.mode(output) <- "double"
    output[!is.finite(output)] <- 0

    if (nrow(output) != nrow(dat)) {
      stop(
        "word2vec row mismatch: expected ", nrow(dat),
        " rows but received ", nrow(output), ".",
        call. = FALSE
      )
    }

    output <- as.data.frame(output)
    names(output) <- paste0("word2vec_", seq_len(ncol(output)))

    return(output)
  }

  defaults <- list(
    Qwen = list(
      batch_size = 1L,
      model = "Qwen/Qwen3-Embedding-8B"
    ),
    NV = list(
      batch_size = 1L,
      model = "nvidia/NV-Embed-v2"
    ),
    e5 = list(
      batch_size = 4L,
      model = "intfloat/e5-large"
    )
  )

  if (is.null(batch_size)) {
    batch_size <- defaults[[embed]]$batch_size
  }

  batch_size <- as.integer(batch_size)

  if (length(batch_size) != 1L ||
      is.na(batch_size) ||
      batch_size < 1L) {
    stop("`batch_size` must be one positive integer.", call. = FALSE)
  }

  if (is.null(model)) {
    model <- defaults[[embed]]$model
  }

  if (!is.character(model) || length(model) != 1L ||
      is.na(model) || !nzchar(model)) {
    stop("`model` must be one non-empty character string.", call. = FALSE)
  }

  if (is.null(python_path)) {
    candidates <- c(
      Sys.which("python3"),
      Sys.which("python")
    )

    candidates <- unname(
      candidates[nzchar(candidates)]
    )

    if (length(candidates) == 0L) {
      stop(
        "No Python executable was found. Supply `python_path` or add Python to PATH.",
        call. = FALSE
      )
    }

    python_path <- candidates[[1L]]
  }

  python_path <- path.expand(python_path)

  if (!file.exists(python_path)) {
    stop(
      "Python executable does not exist: ",
      python_path,
      call. = FALSE
    )
  }

  package_name <- utils::packageName(
    env = environment()
  )

  if (is.null(package_name) || !nzchar(package_name)) {
    package_name <- "textembeddings"
  }

  if (is.null(cache_dir)) {
    cache_dir <- tools::R_user_dir(
      package_name,
      which = "cache"
    )

    cache_dir <- file.path(
      cache_dir,
      "huggingface",
      tolower(embed)
    )
  }

  cache_dir <- path.expand(cache_dir)

  hf_hub_cache <- file.path(
    cache_dir,
    "hub"
  )

  hf_modules_cache <- file.path(
    cache_dir,
    "modules"
  )

  xdg_cache_home <- file.path(
    cache_dir,
    "xdg"
  )

  dirs <- c(
    cache_dir,
    hf_hub_cache,
    hf_modules_cache,
    xdg_cache_home
  )

  invisible(
    lapply(
      dirs,
      dir.create,
      recursive = TRUE,
      showWarnings = FALSE
    )
  )

  if (is.null(python_script)) {

    installed_package <- utils::packageName(
      env = environment()
    )

    if (is.null(installed_package) || !nzchar(installed_package)) {
      stop(
        paste(
          "Could not determine the installed package name.",
          "During package development, supply `python_script` explicitly."
        ),
        call. = FALSE
      )
    }

    python_script <- system.file(
      "python",
      "embed_texts.py",
      package = installed_package
    )
  }

  if (!nzchar(python_script) || !file.exists(python_script)) {
    stop(
      paste(
        "Python embedding worker was not found.",
        "Install it at inst/python/embed_texts.py",
        "or supply `python_script`."
      ),
      call. = FALSE
    )
  }

  task_dir <- tempfile(
    pattern = paste0(
      "embedding_",
      tolower(embed),
      "_"
    )
  )

  dir.create(
    task_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  on.exit(
    unlink(
      task_dir,
      recursive = TRUE,
      force = TRUE
    ),
    add = TRUE
  )

  input_file <- file.path(
    task_dir,
    "texts.csv"
  )

  output_file <- file.path(
    task_dir,
    "embeddings.csv"
  )

  input_data <- data.frame(
    row_id = seq_along(texts_clean),
    text = texts_clean,
    stringsAsFactors = FALSE
  )

  utils::write.csv(
    input_data,
    input_file,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  command_environment <- c(
    paste0("HF_HOME=", cache_dir),
    paste0("HF_HUB_CACHE=", hf_hub_cache),
    paste0("HUGGINGFACE_HUB_CACHE=", hf_hub_cache),
    paste0("TRANSFORMERS_CACHE=", hf_hub_cache),
    paste0("HF_MODULES_CACHE=", hf_modules_cache),
    paste0("XDG_CACHE_HOME=", xdg_cache_home),
    "TOKENIZERS_PARALLELISM=false"
  )

  command_output <- system2(
    command = python_path,
    args = c(
      shQuote(python_script),
      shQuote(input_file),
      shQuote(output_file),
      "--embed",
      shQuote(embed),
      "--model",
      shQuote(model),
      "--batch_size",
      batch_size,
      "--cache_dir",
      shQuote(cache_dir)
    ),
    stdout = TRUE,
    stderr = TRUE,
    env = command_environment
  )

  command_status <- attr(
    command_output,
    "status"
  )

  if (is.null(command_status)) {
    command_status <- 0L
  }

  if (isTRUE(verbose) && length(command_output) > 0L) {
    cat(
      paste(
        command_output,
        collapse = "\n"
      ),
      "\n",
      file = stderr()
    )
  }

  if (
    command_status != 0L ||
    !file.exists(output_file)
  ) {

    details <- if (length(command_output)) {
      paste0(
        "\n",
        paste(
          command_output,
          collapse = "\n"
        )
      )
    } else {
      ""
    }

    stop(
      embed,
      " Python subprocess failed with status ",
      command_status,
      ".",
      details,
      call. = FALSE
    )
  }

  embeddings <- utils::read.csv(
    output_file,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  if (!"row_id" %in% names(embeddings)) {
    stop(
      embed,
      " output is missing `row_id`.",
      call. = FALSE
    )
  }

  embeddings <- embeddings[
    order(embeddings$row_id),
    ,
    drop = FALSE
  ]

  embeddings$row_id <- NULL

  embeddings <- as.matrix(
    embeddings
  )

  storage.mode(embeddings) <- "double"

  if (nrow(embeddings) != nrow(dat)) {
    stop(
      "Embedding row mismatch for ",
      embed,
      ": expected ",
      nrow(dat),
      " rows but received ",
      nrow(embeddings),
      ".",
      call. = FALSE
    )
  }

  if (ncol(embeddings) < 1L) {
    stop(
      embed,
      " returned no embedding dimensions.",
      call. = FALSE
    )
  }

  invalid_rows <- !apply(
    embeddings,
    1L,
    function(row) {
      all(is.finite(row))
    }
  )

  if (any(invalid_rows)) {
    warning(
      embed,
      " returned invalid rows: ",
      paste(
        which(invalid_rows),
        collapse = ", "
      ),
      ". Replacing them with zeros.",
      call. = FALSE
    )

    embeddings[
      invalid_rows,
      ,
      drop = FALSE
    ] <- 0
  }

  embeddings <- as.data.frame(
    embeddings
  )

  names(embeddings) <- paste0(
    embed,
    "_",
    seq_len(ncol(embeddings))
  )

  embeddings
}
