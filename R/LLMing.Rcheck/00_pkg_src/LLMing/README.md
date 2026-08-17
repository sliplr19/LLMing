
# LLMing

<!-- badges: start -->

<!-- badges: end -->

The goal of LLMing is to generate and assess psychological text data.

## Installation

You can install the development version of LLMing from GitHub with:

``` r
# install.packages("pak")
pak::pak("sliplr19/LLMing")
```

## Example

This is a basic example which shows you how to get BERT embeddings:

``` r
library(LLMing)
#> 
#> Attaching package: 'LLMing'
#> The following object is masked from 'package:stats':
#> 
#>     embed

df <- data.frame(
  text = c(
    "I slept well and feel great today!",
    "I saw friends and it went well.",
    "I think I failed that exam. I'm such a disapointment."
  )
)

emb_dat <- 
  LLMing::embed(
    df,
    embed = "word2vec",
    text_col = "text")
```
