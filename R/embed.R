embed <- function(dat, layers, keep_tokens = TRUE, tokens_method = NULL){
  nltk  <- reticulate::import("nltk")
  torch <- reticulate::import("torch", delay_load = TRUE)

  device_choice <- if (torch$cuda$is_available()) "gpu" else "cpu"
  message("Using device: ", device_choice)

  txt_col <- names(dat)[sapply(dat, is.character)][1]
  dat <- quanteda::corpus(dat, text_field = txt_col)
  dat <- dat |>
    quanteda::tokens(remove_punct = TRUE, remove_symbols = TRUE,
                     remove_numbers = TRUE, remove_url = TRUE) |>
    quanteda::tokens_tolower() |>
    quanteda::tokens_remove(stopwords::stopwords("en", source = "smart"))

  texts_clean <- vapply(quanteda::as.list(dat), paste, collapse = " ", FUN.VALUE = character(1))
  texts_clean <- stringi::stri_trans_general(texts_clean, "Latin-ASCII")

  wordembeddings <- text::textEmbed(
    texts = texts_clean,
    model = 'bert-base-uncased',
    layers = layers,
    aggregation_from_layers_to_tokens = "mean",
    aggregation_from_tokens_to_texts = "mean",
    aggregation_from_tokens_to_word_types = tokens_method,
    keep_token_embeddings = keep_tokens,
    device = "gpu"
  )

  return(as.data.frame(wordembeddings$texts$texts))
}

