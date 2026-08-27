#' Clean texts before embeddings
#'
#' Converted to lowercase with numbers,
#' punctuations, slashes, extra whitespace,
#' and stopwords removed. Stemming is also applied
#'
#' @param dat Dataframe containing a text column
#' @param text_col Name of text column in dat
#'
#' @returns Vector of clean text
#' @export

clean_texts <- function(
    dat,
    text_col
) {


  if (!is.data.frame(dat)) {
    stop("dat must be a data.frame.")
  }

  if (!text_col %in% names(dat)) {
    stop(
      "Missing text column: ",
      text_col
    )
  }

  texts <- as.character(
    dat[[text_col]]
  )

  texts[is.na(texts)] <- ""

  tokens <- quanteda::tokens(
    texts,
    remove_punct = TRUE,
    remove_symbols = TRUE,
    remove_numbers = TRUE,
    remove_url = TRUE
  )

  tokens <- quanteda::tokens_tolower(
    tokens
  )

  tokens <- quanteda::tokens_remove(
    tokens,
    stopwords::stopwords(
      "en",
      source = "smart"
    )
  )

  texts_clean <- vapply(
    quanteda::as.list(tokens),
    paste,
    collapse = " ",
    FUN.VALUE = character(1)
  )

  texts_clean <- stringi::stri_trim_both(
    texts_clean
  )

  texts_clean[is.na(texts_clean)] <- ""
  texts_clean[texts_clean == ""] <- "empty text"

  texts_clean
}

