#' Embed texts with a selected embedding model
#'
#' Cleans a text column and converts it to a numeric embedding matrix or
#' dataframe, with one row per input text and one column per embedding
#' dimension. Supports pure R and Python-backed embedding methods.
#'
#' @param dat A dataframe containing one text per row.
#' @param method Character scalar specifying the embedding method. One of
#'   `"E5"`, `"Qwen3"`, `"NV-Embed"`, `"BERT"`, or `"GloVe"`.
#' @param text_col Character scalar giving the name of the text column in
#'   `dat`. Defaults to `"text"`.
#' @param py_e5_qwen Optional path to the Python executable to use for the
#'   `"E5"` and `"Qwen3"` methods.
#' @param py_nv Optional path to the Python executable to use for the
#'   `"NV-Embed"` method.
#' @param hf_cache_root Optional path to a cache directory for Hugging Face
#'   models and related files. Required for Python-backed methods and used as a
#'   temporary cache for BERT when not otherwise supplied.
#' @param max_length Integer maximum sequence length passed to the tokenizer for
#'   the `"NV-Embed"` method. Defaults to `256`.
#' @param prefer_gpu Logical; if `TRUE`, Python-backed methods will try to use a
#'   CUDA-enabled GPU when available. Defaults to `FALSE`.
#' @param auto_install_sentence_transformers Logical; if `TRUE`, attempts to
#'   install or update the `sentence-transformers` Python package for the
#'   `"E5"` and `"Qwen3"` methods. Defaults to `FALSE`.
#'
#' @returns A numeric matrix or dataframe with one row per input text and one
#'   column per embedding dimension.
#'
#' @details
#' The input text is lightly preprocessed before embedding. This preprocessing:
#' \itemize{
#'   \item removes punctuation, symbols, numbers, and URLs,
#'   \item converts text to lowercase,
#'   \item removes English stopwords,
#'   \item transliterates text to ASCII.
#' }
#'
#' Method-specific behavior:
#' \itemize{
#'   \item `"GloVe"` uses a pure R workflow via `text2vec` and returns
#'   document-level mean embeddings.
#'   \item `"BERT"` uses `text::textEmbed()` with the `"bert-base-uncased"`
#'   model on CPU.
#'   \item `"E5"` and `"Qwen3"` use Python and `sentence-transformers`.
#'   \item `"NV-Embed"` uses Python and `transformers`.
#' }
#'
#' @examples
#' df <- data.frame(
#'   text = c(
#'     "I slept well and feel great today!",
#'     "I saw friends and it went well.",
#'     "I think I failed that exam. I'm such a disappointment."
#'   )
#' )
#'
#' emb_dat <- embed(
#'   dat = df,
#'   method = "BERT",
#'   text_col = "text"
#' )
#'
#' @export

embed <- function(dat,
                  method = c("E5", "Qwen3", "NV-Embed", "BERT", "GloVe"),
                  text_col = "text",
                  py_e5_qwen = NULL,
                  py_nv = NULL,
                  hf_cache_root = NULL,
                  max_length = 256,
                  prefer_gpu = FALSE,
                  auto_install_sentence_transformers = FALSE) {

  Sys.setlocale("LC_NUMERIC", "C")
  method <- match.arg(method)

  if (!is.data.frame(dat)) stop("dat must be a data.frame")
  if (!text_col %in% names(dat)) stop("Missing text column: ", text_col)

  # ---- one row per text ----
  texts <- as.character(dat[[text_col]])
  texts[is.na(texts)] <- ""

  # ---- optional cleaning (keeps rows aligned) ----
  toks <- quanteda::tokens(
    texts,
    remove_punct = TRUE,
    remove_symbols = TRUE,
    remove_numbers = TRUE,
    remove_url = TRUE
  )
  toks <- quanteda::tokens_tolower(toks)
  toks <- quanteda::tokens_remove(toks, stopwords::stopwords("en", source = "smart"))

  texts_clean <- vapply(quanteda::as.list(toks), paste, collapse = " ", FUN.VALUE = character(1))
  texts_clean <- stringi::stri_trans_general(texts_clean, "Latin-ASCII")

  # Avoid inherited caches confusing transformers
  Sys.unsetenv("TRANSFORMERS_CACHE")
  Sys.unsetenv("HF_HUB_CACHE")
  Sys.unsetenv("HUGGINGFACE_HUB_CACHE")

  # hf_cache_root only required for python-based methods
  if (method %in% c("E5", "Qwen3", "NV-Embed")) {
    if (is.null(hf_cache_root) || !nzchar(hf_cache_root)) stop("hf_cache_root must be set")
    dir.create(hf_cache_root, recursive = TRUE, showWarnings = FALSE)
  }

  # ------------------------------
  # Pure R: GloVe (document mean)
  # ------------------------------
  if (method == "GloVe") {
    if (!requireNamespace("text2vec", quietly = TRUE)) stop("Install text2vec for GloVe")
    if (!requireNamespace("Matrix", quietly = TRUE)) stop("Install Matrix for GloVe")

    tokens <- text2vec::space_tokenizer(texts_clean)
    it <- text2vec::itoken(tokens, progressbar = FALSE)

    vocab <- text2vec::create_vocabulary(it)
    vectorizer <- text2vec::vocab_vectorizer(vocab)

    dtm <- text2vec::create_dtm(it, vectorizer)
    tcm <- text2vec::create_tcm(it, vectorizer, skip_grams_window = 5L)

    glove <- text2vec::GlobalVectors$new(rank = 50, x_max = 10)
    wv_main <- glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01, n_threads = 4)
    wv <- wv_main + t(glove$components)

    row_sums <- Matrix::rowSums(dtm)
    row_sums[row_sums == 0] <- 1
    dtm_norm <- dtm / row_sums
    return(as.matrix(dtm_norm %*% wv))
  }
  # ------------------------------
  # BERT via text::textEmbed (CPU forced) + fix NLTK punkt_tab
  # ------------------------------
  if (method == "BERT") {
    if (!requireNamespace("text", quietly = TRUE)) stop("Install text for BERT")
    if (!requireNamespace("reticulate", quietly = TRUE)) stop("Install reticulate for BERT")

    # Writable cache for NLTK downloads (avoid home dir on clusters)
    if (is.null(hf_cache_root) || !nzchar(hf_cache_root)) {
      hf_cache_root <- file.path(tempdir(), "hf_cache")
    }
    dir.create(hf_cache_root, recursive = TRUE, showWarnings = FALSE)

    nltk_dir <- file.path(hf_cache_root, "nltk_data")
    dir.create(nltk_dir, recursive = TRUE, showWarnings = FALSE)
    Sys.setenv(NLTK_DATA = nltk_dir)

    # Force CPU always
    device <- "cpu"

    # Ensure NLTK resources exist
    try({
      nltk <- reticulate::import("nltk", delay_load = TRUE)
      suppressWarnings(nltk$download("punkt", download_dir = nltk_dir, quiet = TRUE))
      suppressWarnings(nltk$download("punkt_tab", download_dir = nltk_dir, quiet = TRUE))
    }, silent = TRUE)

    emb <- text::textEmbed(
      texts = texts_clean,
      model = "bert-base-uncased",
      layers = 1,
      aggregation_from_layers_to_tokens = "mean",
      aggregation_from_tokens_to_texts  = "mean",
      keep_token_embeddings = FALSE,
      device = device
    )

    # ---- Robustly locate embeddings in textEmbed output (version-proof) ----
    find_emb <- function(x, n_texts) {
      # direct hits
      if (is.data.frame(x) && nrow(x) %in% c(n_texts, 1L) && ncol(x) > 1) return(x)
      if (is.matrix(x)     && nrow(x) %in% c(n_texts, 1L) && ncol(x) > 1) return(as.data.frame(x))

      # some versions store as list-of-vectors (one per text)
      if (is.list(x) && length(x) %in% c(n_texts, 1L) && length(x) > 0 && is.numeric(x[[1]])) {
        m <- do.call(rbind, x)
        return(as.data.frame(m))
      }

      # recurse through lists
      if (is.list(x)) {
        for (nm in names(x)) {
          res <- find_emb(x[[nm]], n_texts)
          if (!is.null(res)) return(res)
        }
      }
      NULL
    }

    out <- find_emb(emb, length(texts_clean))

    # Helpful debug if nothing found (prints top-level structure once)
    if (is.null(out)) {
      cat("textEmbed returned names:\n")
      print(names(emb))
      if (!is.null(emb$texts)) {
        cat("\ntextEmbed$texts names:\n")
        print(names(emb$texts))
      }
      stop("Could not find embeddings in textEmbed output.")
    }

    return(out)
  }
  # ------------------------------
  # Helper: run python subprocess
  # ------------------------------
  py_run <- function(python_exe, py_lines, env_vars) {
    if (is.null(python_exe) || !file.exists(python_exe)) {
      stop("Python executable not found: ", python_exe)
    }
    python_exe <- trimws(python_exe)
    message("Running python executable: ", python_exe)

    in_csv  <- tempfile(fileext = ".csv")
    out_csv <- tempfile(fileext = ".csv")
    py_file <- tempfile(fileext = ".py")

    utils::write.csv(
      data.frame(text = texts_clean),
      in_csv,
      row.names = FALSE,
      fileEncoding = "UTF-8"
    )
    writeLines(py_lines, py_file)

    out_log <- tempfile()
    err_log <- tempfile()

    # save old env and restore on exit
    env_names <- sub("=.*$", "", env_vars)
    old_env <- Sys.getenv(env_names, unset = NA_character_)

    env_list <- as.list(sub("^[^=]*=", "", env_vars))
    names(env_list) <- env_names
    do.call(Sys.setenv, env_list)

    on.exit({
      for (nm in env_names) {
        old_val <- old_env[[nm]]
        if (is.na(old_val)) {
          Sys.unsetenv(nm)
        } else {
          do.call(Sys.setenv, stats::setNames(list(old_val), nm))
        }
      }
    }, add = TRUE)

    status <- system2(
      python_exe,
      c(py_file, in_csv, out_csv),
      stdout = out_log,
      stderr = err_log
    )

    py_out <- readLines(out_log, warn = FALSE)
    py_err <- readLines(err_log, warn = FALSE)

    cat("\n--- python stdout (tail) ---\n",
        paste(utils::tail(py_out, 80), collapse = "\n"),
        "\n", sep = "")
    cat("\n--- python stderr (tail) ---\n",
        paste(utils::tail(py_err, 200), collapse = "\n"),
        "\n", sep = "")

    if (!identical(status, 0L)) {
      stop(
        "Python subprocess failed (exit status ", status, ")\n",
        "stdout log: ", out_log, "\n",
        "stderr log: ", err_log, "\n",
        "---- stderr tail (last 200 lines) ----\n",
        paste(utils::tail(py_err, 200), collapse = "\n"),
        "\n"
      )
    }

    if (!file.exists(out_csv) || file.info(out_csv)$size < 10) {
      stop("Python produced an empty output CSV: ", out_csv)
    }

    df <- utils::read.csv(out_csv, check.names = FALSE)
    emb_mat <- as.matrix(df)

    if (nrow(emb_mat) != length(texts_clean)) {
      stop(
        "Row mismatch: got ", nrow(emb_mat),
        " embeddings but had ", length(texts_clean), " input texts."
      )
    }

    emb_mat
  }
  # ------------------------------
  # E5 / Qwen3 via sentence-transformers
  # ------------------------------
  if (method %in% c("E5", "Qwen3")) {
    if (is.null(py_e5_qwen) || !file.exists(py_e5_qwen)) stop("py_e5_qwen not found: ", py_e5_qwen)

    hf_cache_subdir <- if (method == "E5") "st_e5-base-v2" else "st_qwen3-embed"
    hf_cache <- file.path(hf_cache_root, hf_cache_subdir)
    dir.create(hf_cache, recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(hf_cache, "tmp"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(hf_cache, "hub"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(hf_cache, "datasets"), recursive = TRUE, showWarnings = FALSE)

    if (isTRUE(auto_install_sentence_transformers)) {
      suppressWarnings(system2(py_e5_qwen, c("-m", "pip", "install", "-U", "sentence-transformers"),
                               stdout = TRUE, stderr = TRUE))
    }

    model_id <- if (method == "E5") "intfloat/e5-base-v2" else "Qwen/Qwen3-Embedding-4B"

    py_lines <- c(
      "import os, sys, traceback",
      "os.environ['TOKENIZERS_PARALLELISM'] = 'false'",
      "in_path  = sys.argv[1]",
      "out_path = sys.argv[2]",
      sprintf("model_id = %s", shQuote(model_id)),
      sprintf("method = %s", shQuote(method)),
      sprintf("prefer_gpu = %s", if (isTRUE(prefer_gpu)) "True" else "False"),
      "",
      "try:",
      "    import pandas as pd",
      "    df = pd.read_csv(in_path)",
      "    texts = df['text'].fillna('').astype(str).tolist()",
      "",
      "    from huggingface_hub import snapshot_download",
      "    cache_dir = os.environ.get('HF_HOME', None)",
      "    print('HF_HOME:', cache_dir, flush=True)",
      "    local_dir = snapshot_download(repo_id=model_id, cache_dir=cache_dir)",
      "    print('Snapshot ready at:', local_dir, flush=True)",
      "",
      "    from sentence_transformers import SentenceTransformer",
      "    import torch",
      "    device = 'cuda' if (prefer_gpu and torch.cuda.is_available()) else 'cpu'",
      "    print('ST device:', device, flush=True)",
      "    prefix = 'passage: ' if method == 'E5' else ''",
      "    texts2 = [prefix + t for t in texts]",
      "    mdl = SentenceTransformer(local_dir, device=device)",
      "    emb = mdl.encode(texts2, batch_size=8, convert_to_numpy=True, normalize_embeddings=True)",
      "",
      "    out_df = pd.DataFrame(emb)",
      "    out_df.to_csv(out_path, index=False)",
      "except Exception as e:",
      "    print('\\nPYTHON_EXCEPTION:', repr(e), file=sys.stderr, flush=True)",
      "    traceback.print_exc()",
      "    raise"
    )

    env_vars <- c(
      paste0("HF_HOME=", hf_cache),
      paste0("HF_HUB_CACHE=", file.path(hf_cache, "hub")),
      paste0("HF_DATASETS_CACHE=", file.path(hf_cache, "datasets")),
      paste0("XDG_CACHE_HOME=", hf_cache),
      paste0("TMPDIR=", file.path(hf_cache, "tmp")),
      paste0("HOME=", hf_cache),
      "HF_HUB_DISABLE_XET=1",
      "HF_HUB_DISABLE_TELEMETRY=1",
      "TOKENIZERS_PARALLELISM=false"
    )

    return(py_run(py_e5_qwen, py_lines, env_vars))
  }

  # ------------------------------
  # NV-Embed v2 (handles 2D OR 3D outputs)
  # ------------------------------
  if (method == "NV-Embed") {
    if (is.null(py_nv) || !file.exists(py_nv)) stop("py_nv not found: ", py_nv)

    hf_cache_subdir <- "nv_NV-Embed-v2"
    hf_cache <- file.path(hf_cache_root, hf_cache_subdir)
    dir.create(hf_cache, recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(hf_cache, "tmp"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(hf_cache, "hub"), recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(hf_cache, "datasets"), recursive = TRUE, showWarnings = FALSE)

    py_lines <- c(
      "import os, sys, traceback",
      "os.environ['TOKENIZERS_PARALLELISM'] = 'false'",
      "os.environ['HF_HUB_DISABLE_XET'] = '1'",
      "",
      "in_path  = sys.argv[1]",
      "out_path = sys.argv[2]",
      "model_id = 'nvidia/NV-Embed-v2'",
      sprintf("prefer_gpu = %s", if (isTRUE(prefer_gpu)) "True" else "False"),
      sprintf("max_length = %d", as.integer(max_length)),
      "",
      "try:",
      "    import pandas as pd",
      "    df = pd.read_csv(in_path)",
      "    texts = df['text'].fillna('').astype(str).tolist()",
      "",
      "    from huggingface_hub import snapshot_download",
      "    cache_dir = os.environ.get('HF_HOME', None)",
      "    print('HF_HOME:', cache_dir, flush=True)",
      "    local_dir = snapshot_download(repo_id=model_id, cache_dir=cache_dir)",
      "    print('Snapshot at:', local_dir, flush=True)",
      "",
      "    import torch",
      "    from transformers import AutoTokenizer, AutoModel",
      "    device = 'cuda' if (prefer_gpu and torch.cuda.is_available()) else 'cpu'",
      "    print('CUDA available:', torch.cuda.is_available(), flush=True)",
      "    print('Using device:', device, flush=True)",
      "    dtype = torch.float16 if device == 'cuda' else torch.float32",
      "",
      "    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)",
      "    mdl = AutoModel.from_pretrained(",
      "        local_dir, trust_remote_code=True, local_files_only=True,",
      "        torch_dtype=dtype, low_cpu_mem_usage=True",
      "    )",
      "    mdl = mdl.to(device)",
      "    mdl.eval()",
      "",
      "    batch = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')",
      "    batch = {k: v.to(device) for k, v in batch.items()}",
      "",
      "    with torch.no_grad():",
      "        out = mdl(**batch)",
      "",
      "    # ---- Extract something embedding-like ----",
      "    emb = None",
      "    if hasattr(out, 'sentence_embedding') and out.sentence_embedding is not None:",
      "        emb = out.sentence_embedding",
      "    elif hasattr(out, 'sentence_embeddings') and out.sentence_embeddings is not None:",
      "        emb = out.sentence_embeddings",
      "    elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:",
      "        emb = out.last_hidden_state",
      "    elif isinstance(out, dict):",
      "        for k in ('sentence_embedding','sentence_embeddings','text_embeddings','embeddings','embedding'):",
      "            if k in out and out[k] is not None:",
      "                emb = out[k]",
      "                break",
      "        if emb is None:",
      "            for k in ('last_hidden_state','token_embeddings','encoder_last_hidden_state'):",
      "                if k in out and out[k] is not None:",
      "                    emb = out[k]",
      "                    break",
      "    elif isinstance(out, (tuple, list)) and len(out) > 0:",
      "        emb = out[0]",
      "",
      "    if emb is None:",
      "        raise RuntimeError('Could not extract embeddings from model output.')",
      "",
      "    # Move to CPU tensor if needed",
      "    if not torch.is_tensor(emb):",
      "        emb = torch.tensor(emb)",
      "    emb = emb.to(device)",
      "",
      "    # If emb is 3D (N,T,D), mean-pool across T using attention mask",
      "    if emb.dim() == 3:",
      "        # emb: (N,T,D)",
      "        attn = batch.get('attention_mask', None)",
      "        if attn is None:",
      "            emb = emb.mean(dim=1)",
      "        else:",
      "            attn = attn.unsqueeze(-1).to(dtype=emb.dtype)  # (N,T,1)",
      "            emb = (emb * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)",
      "",
      "    # If emb is 1D (D,), add batch dim",
      "    if emb.dim() == 1:",
      "        emb = emb.unsqueeze(0)",
      "",
      "    # Now emb must be 2D (N,D)",
      "    if emb.dim() != 2:",
      "        raise RuntimeError(f'Expected 2D embeddings after pooling, got shape={tuple(emb.shape)}')",
      "",
      "    emb = torch.nn.functional.normalize(emb, p=2, dim=1)",
      "    emb = emb.detach().cpu().numpy()",
      "",
      "    out_df = pd.DataFrame(emb)",
      "    out_df.to_csv(out_path, index=False)",
      "except Exception as e:",
      "    print('\\nPYTHON_EXCEPTION:', repr(e), file=sys.stderr, flush=True)",
      "    traceback.print_exc()",
      "    raise"
    )

    env_vars <- c(
      paste0("HF_HOME=", hf_cache),
      paste0("HF_HUB_CACHE=", file.path(hf_cache, "hub")),
      paste0("HF_DATASETS_CACHE=", file.path(hf_cache, "datasets")),
      paste0("XDG_CACHE_HOME=", hf_cache),
      paste0("TMPDIR=", file.path(hf_cache, "tmp")),
      paste0("HOME=", hf_cache),
      "HF_HUB_DISABLE_XET=1",
      "HF_HUB_DISABLE_TELEMETRY=1",
      "TOKENIZERS_PARALLELISM=false"
    )

    return(py_run(py_nv, py_lines, env_vars))
  }

  stop("Unhandled method: ", method)
}

