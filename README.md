
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
   dat = df,
   method = "BERT")
#> [0;34mProcessing batch 1/1
#> [0m
#> [0;32mCompleted layers output for texts (variable: 1/1, duration: 5.404953 secs).
#> [0m
#> [0;32mCompleted layers aggregation for word_type_embeddings. 
#> [0m
#> [0;34mCompleted layers aggregation (variable 1/1, duration: 0.317711 secs).
#> [0m
#> [0;32mMinutes from start:  1.704[0m
#> [0;30mEstimated embedding time left = 0 minutes[0m
```
