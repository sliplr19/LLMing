#' Generate text data via Python LLM
#'
#' All prompt components and example texts are provided by the user as
#' function arguments. This function generates text data based on
#' severity score from a given questionnaire.
#'
#' @param prompts A data.frame with one row per diary to generate.
#'   Must contain at least a column indicating severity level.
#' @param examples A data.frame of example diary texts with columns
#'   `text` and `label`.
#' @param scenario Character string used in the SCENARIO section.
#' @param overall_rules Character string describing global writing rules.
#' @param percentile_scaffold Character string describing how percentiles
#'   map onto severity.
#' @param item_rules Character string describing how to internally choose
#'   symptom patterns.
#' @param items Character string of the battery under study.
#' @param structure_rules Character string describing structural rules.
#' @param percentile_specification Character string describing what the
#'   severity percentile means.
#' @param band_specification Character string describing severity bands.
#' @param example_instruction Character string introducing the example texts.
#' @param what_to_write Character string describing what the model should
#'   write about.
#' @param task_desc Character string for the system-level role description.
#' @param target_min Integer minimum number of tokens to generate.
#' @param target_max Integer maximum number of tokens to generate.
#' @param temperature Numeric temperature for sampling.
#' @param top_p Numeric top-p nucleus sampling value.
#' @param repetition_penalty Numeric repetition penalty.
#' @param model_name Model identifier string to pass to transformers.
#' @param batch_size Integer passed through to the Python script.
#' @param python Path to the Python executable.
#' @param env Optional named character vector or list of environment variables.
#' @param output_file Optional path to save the output CSV.
#'
#' @return A data.frame with columns `id`, `severity`, and `response`.
#'
#' @examples
#' prompts <- data.frame(
#'   id = 1:2,
#'   severity = c(10, 80),
#'   num_examples = c(1, 1)
#' )
#'
#' examples <- data.frame(
#'   text = c("Example A", "Example B"),
#'   label = c("group1", "group2"),
#'   stringsAsFactors = FALSE
#' )
#'
#' @export


text_datagen <- function(prompts,
                         examples,
                         scenario = NULL,
                         overall_rules = NULL,
                         percentile_scaffold = NULL,
                         item_rules= NULL,
                         items = NULL,
                         structure_rules = NULL,
                         percentile_specification = NULL,
                         band_specification = NULL,
                         example_instruction = NULL,
                         what_to_write = NULL,
                         task_desc = NULL,
                         target_min = 90L,
                         target_max = 100L,
                         temperature = 0.4,
                         top_p = 0.9,
                         repetition_penalty = 1.1,
                         model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                         batch_size = 2L,
                         python = Sys.getenv("RETICULATE_PYTHON", "python"),
                         env = NULL,
                         output_file = NULL) {
  # ---- basic checks ----
  if (!is.data.frame(prompts)) {
    stop("'prompts' must be a data.frame.")
  }
  if (!"severity" %in% names(prompts)) {
    stop("'prompts' must contain a 'severity' column.")
  }
  if (!is.data.frame(examples)) {
    stop("'examples' must be a data.frame.")
  }
  if (!all(c("text", "label") %in% names(examples))) {
    stop("'examples' must contain columns 'text' and 'label'.")
  }

  # ---- locate python script in package ----
  pkg <- utils::packageName()
  script_path <- system.file("python", "text_datagen.py", package = pkg)
  if (!nzchar(script_path) || !file.exists(script_path)) {
    stop("Could not find 'text_datagen.py' in inst/python/ of the package.")
  }

  # ---- write temporary files ----
  tmp_dir <- tempdir()
  input_csv    <- file.path(tmp_dir, paste0("prompts_",  Sys.getpid(), ".csv"))
  examples_csv <- file.path(tmp_dir, paste0("examples_", Sys.getpid(), ".csv"))
  config_json  <- file.path(tmp_dir, paste0("prompt_cfg_", Sys.getpid(), ".json"))

  utils::write.csv(prompts,  input_csv,    row.names = FALSE)
  utils::write.csv(examples, examples_csv, row.names = FALSE)

  cfg <- list(
    scenario                = scenario,
    overall_rules           = overall_rules,
    percentile_scaffold     = percentile_scaffold,
    item_rules              = item_rules,
    items                   = items,
    structure_rules         = structure_rules,
    percentile_specification = percentile_specification,
    band_specification      = band_specification,
    example_instruction     = example_instruction,
    what_to_write           = what_to_write,
    task_desc               = task_desc,
    target_min              = as.integer(target_min),
    target_max              = as.integer(target_max),
    temperature             = as.numeric(temperature),
    top_p                   = as.numeric(top_p),
    repetition_penalty      = as.numeric(repetition_penalty)
  )

  jsonlite::write_json(cfg, config_json, auto_unbox = TRUE, pretty = TRUE)

  if (is.null(output_file)) {
    output_file <- file.path(tmp_dir, paste0("diaries_", Sys.getpid(), ".csv"))
  }
  output_file <- normalizePath(output_file, mustWork = FALSE)

  # ---- handle env vars (generic) ----
  if (!is.null(env)) {
    if (is.list(env)) {
      env <- unlist(env, use.names = TRUE)
    }
    if (!is.character(env) || is.null(names(env)) || any(names(env) == "")) {
      stop("'env' must be a named character vector or list.")
    }

    env_names <- names(env)
    old_env <- stats::setNames(
      vapply(env_names, Sys.getenv, FUN.VALUE = character(1), unset = NA_character_),
      env_names
    )

    on.exit({
      for (nm in env_names) {
        old_val <- old_env[[nm]]
        if (is.na(old_val)) {
          Sys.unsetenv(nm)
        } else {
          Sys.setenv(structure(old_val, names = nm))
        }
      }
    }, add = TRUE)

    for (nm in env_names) {
      Sys.setenv(structure(env[[nm]], names = nm))
    }
  }

  # ---- normalize paths ----
  input_csv    <- normalizePath(input_csv,    mustWork = TRUE)
  examples_csv <- normalizePath(examples_csv, mustWork = TRUE)
  config_json  <- normalizePath(config_json,  mustWork = TRUE)

  # ---- build args for python ----
  args <- c(
    script_path,
    input_csv,
    output_file,
    "--batch_size",   as.character(as.integer(batch_size)),
    "--example_file", examples_csv,
    "--config_file",  config_json,
    "--model_name",   model_name
  )

  status <- system2(
    command = python,
    args    = args
  )

  if (!identical(status, 0L)) {
    stop("Python text_datagen.py failed with status code: ", status)
  }

  res <- utils::read.csv(output_file, stringsAsFactors = FALSE)
  res
}
