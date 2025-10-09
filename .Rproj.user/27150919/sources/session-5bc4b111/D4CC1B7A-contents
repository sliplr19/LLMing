#' Embed texts with a Transformer model
#'
#' Cleans a text column and converts it to a dataframe of numeric vectors via
#' BERT embeddings. For the input dataframe, each row
#' is one text entry.
#'
#' @param dat A dataframe with text data, one text per row
#' @param layers Integer vector specifying which model layers to aggregate from.
#' @param keep_tokens Logical, keep token-level embeddings in the returned
#'   object or discard them to save memory
#' @param tokens_method  Character scalar controlling how token-level
#'   embeddings are aggregated to word types
#'
#' @returns A dataframe where each row corresponds to one input text and each
#'   column is an embedding dimension
#'
#'   @examples df <- data.frame(
#'   text = c(
#'     "I slept well and feel great today!",
#'     "I saw from friends and it went well.",
#'     "I think I failed that exam. I'm such a disapointment."
#'   )
#' )
#'
#' emb_dat <- embed(
#'   dat = df,
#'   layers = 1,
#'   keep_tokens = FALSE,
#'   tokens_method = "mean"
#' )
#' @export
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

