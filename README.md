
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

emb_dat <- embed(
  dat = df,
  layers = 1,
  keep_tokens = FALSE,
  tokens_method = "mean"
)
#> Using device: cpu
#> [0;34mProcessing batch 1/1
#> [0m
#> MPS for Mac available: False
#> Unable to use MPS (Mac M1+), CUDA (GPU), using CPU
#> [0;32mCompleted layers output for texts (variable: 1/1, duration: 0.783682 secs).
#> [0m
#> [0;32mCompleted layers aggregation for word_type_embeddings. 
#> [0m
#> [0;34mCompleted layers aggregation (variable 1/1, duration: 0.228859 secs).
#> [0m
#> [0;35mEmbedding single context embeddings.
#> [0m
#> [0;34mCompleted layers aggregation (variable 1/13, duration: 0.020877 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 2/13, duration: 0.019911 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 3/13, duration: 0.019728 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 4/13, duration: 0.035636 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 5/13, duration: 0.035478 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 6/13, duration: 0.068546 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 7/13, duration: 0.017823 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 8/13, duration: 0.016648 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 9/13, duration: 0.021572 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 10/13, duration: 0.024744 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 11/13, duration: 0.019792 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 12/13, duration: 0.018830 secs).
#> [0m
#> [0;34mCompleted layers aggregation (variable 13/13, duration: 0.017301 secs).
#> [0m
#> [0;35mDone! 
#> [0m
#> [0;32mMinutes from start:  0.138[0m
#> [0;30mEstimated embedding time left = 0 minutes[0m
```
