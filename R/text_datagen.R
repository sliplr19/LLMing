#' Generate Synthetic Text Using an Ollama Language Model
#'
#' Generates synthetic text using a locally available Ollama language model.
#' Generation conditions are supplied in an input CSV, and few-shot examples
#' are supplied in a separate example CSV. Prompt content, severity
#' instructions, model settings, and generation settings can be customized.
#'
#' The Python generation script `text_datagen.py` is included with the package
#' and is located automatically. Users do not need to provide the path to the
#' Python script.
#'
#' @param prompt_info_csv Character string. Path to the CSV containing the
#'   generation conditions. The file must contain a `severity` column and may
#'   contain `num` and `seed` columns. `num` controls the number of examples
#'   selected from each example category, and `seed` controls random example
#'   selection.
#'
#' @param examples_csv Character string. Path to the CSV containing few-shot
#'   text examples. The file must contain `label` and `text` columns.
#'
#' @param output_csv Character string. Path where the CSV containing the
#'   generated text will be written.
#'
#' @param model Character string. Name of the Ollama model used for text
#'   generation, for example `"llama3:8b"`.
#'
#' @param python_script Character. Path to the Python script used to generate
#'   the synthetic text.
#'
#' @param python_path Character string. Python executable used to run the
#'   package's Python generation script. Defaults to `"python"`.
#'
#' @param system_prompt Character string. System-level instruction supplied
#'   to the language model.
#'
#' @param prompt_template Character string. Template used to construct the
#'   user prompt. The Python worker should replace `{severity}`,
#'   `{severity_instructions}`, `{items}`, and `{example_section}` with the
#'   corresponding information for each generation.
#'
#' @param items Character string. Questionnaire items, construct description,
#'   scoring information, or other measurement information supplied to the
#'   language model.
#'
#' @param severity_instructions A data.frame with columns `min`, `max`, and
#'   `instructions`. Each row defines the instructions associated with a
#'   range of severity values.
#'
#' @param severity_min Numeric. Minimum valid severity value. Defaults to 10.
#'
#' @param severity_max Numeric. Maximum valid severity value. Defaults to 90.
#'
#' @param label_order Character vector. Order in which example categories are
#'   selected when `num = 1`.
#'
#' @param label_order_double Character vector. Order in which example
#'   categories are selected when `num = 2`.
#'
#' @param batch_size Integer. Batch-size argument supplied to the Python
#'   generation script. Defaults to 2.
#'
#' @param max_retries Integer. Number of retries permitted after an invalid
#'   or failed model response. Defaults to 2.
#'
#' @param temperature Numeric. Ollama sampling temperature. Higher values
#'   generally produce more variable output. Defaults to 0.7.
#'
#' @param top_p Numeric. Nucleus-sampling probability passed to Ollama.
#'   Defaults to 0.9.
#'
#' @param repeat_penalty Numeric. Repetition penalty passed to Ollama.
#'   Defaults to 1.05.
#'
#' @param num_predict Integer. Maximum number of tokens that Ollama may
#'   generate for each response. Defaults to 1200.
#'
#' @param num_ctx Integer. Context-window size supplied to Ollama. Defaults
#'   to 8192.
#'
#' @param min_words Integer. Minimum number of words required for a generated
#'   response to be considered valid. Defaults to 80.
#'
#' @param max_words Integer. Maximum number of words permitted for a generated
#'   response to be considered valid. Defaults to 400.
#'
#' @param require_single_paragraph Logical. If `TRUE`, responses containing
#'   multiple paragraphs are considered invalid. Defaults to `TRUE`.
#'
#' @param verbose Logical. If `TRUE`, output from the Python subprocess is
#'   printed to the R console. Defaults to `TRUE`.
#'
#' @return A data.frame containing the original generation-condition columns
#'   and a `response` column containing the generated text.
#'
#' @export
text_datagen <- function(
    prompt_info_csv,
    examples_csv,
    output_csv,
    model,
    system_prompt,
    prompt_template,
    items,
    severity_instructions,
    python_script = "text_datagen.py",
    python_path = "python",
    severity_min = 10,
    severity_max = 90,
    label_order = c("minimum", "moderate", "severe"),
    label_order_double = c(
      "minimum", "moderate", "severe",
      "severe", "moderate", "minimum"
    ),
    batch_size = 2L,
    max_retries = 2L,
    temperature = 0.7,
    top_p = 0.9,
    repeat_penalty = 1.05,
    num_predict = 1200L,
    num_ctx = 8192L,
    min_words = 80L,
    max_words = 400L,
    require_single_paragraph = TRUE,
    verbose = TRUE
) {


  if (!file.exists(prompt_info_csv)) {
    stop(
      "Could not find `prompt_info_csv`: ",
      prompt_info_csv,
      call. = FALSE
    )
  }

  if (!file.exists(examples_csv)) {
    stop(
      "Could not find `examples_csv`: ",
      examples_csv,
      call. = FALSE
    )
  }

  if (!file.exists(python_script)) {
    stop(
      "Could not find `python_script`: ",
      python_script,
      call. = FALSE
    )
  }


  prompt_info <- utils::read.csv(
    prompt_info_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  if (!"severity" %in% names(prompt_info)) {
    stop(
      "`prompt_info_csv` must contain a `severity` column.",
      call. = FALSE
    )
  }

  severity <- suppressWarnings(
    as.numeric(prompt_info$severity)
  )

  if (
    anyNA(severity) ||
    any(!is.finite(severity))
  ) {
    stop(
      "All values in `severity` must be finite numeric values.",
      call. = FALSE
    )
  }

  invalid_severity <- (
    severity < severity_min |
      severity > severity_max
  )

  if (any(invalid_severity)) {
    stop(
      "`severity` values must be between ",
      severity_min,
      " and ",
      severity_max,
      ".",
      call. = FALSE
    )
  }


  if ("num" %in% names(prompt_info)) {

    num <- suppressWarnings(
      as.integer(prompt_info$num)
    )

    if (
      anyNA(num) ||
      any(!num %in% c(1L, 2L))
    ) {
      stop(
        "If present, `num` must contain only 1 or 2.",
        call. = FALSE
      )
    }
  }


  examples <- utils::read.csv(
    examples_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  required_example_columns <- c(
    "label",
    "text"
  )

  missing_example_columns <- setdiff(
    required_example_columns,
    names(examples)
  )

  if (length(missing_example_columns) > 0L) {
    stop(
      "`examples_csv` is missing required column(s): ",
      paste(
        missing_example_columns,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  example_labels <- unique(
    trimws(
      tolower(
        as.character(examples$label)
      )
    )
  )

  requested_labels <- unique(
    tolower(
      c(
        label_order,
        label_order_double
      )
    )
  )

  missing_labels <- setdiff(
    requested_labels,
    example_labels
  )

  if (length(missing_labels) > 0L) {
    stop(
      "`examples_csv` does not contain label(s): ",
      paste(
        missing_labels,
        collapse = ", "
      ),
      call. = FALSE
    )
  }


  if (!is.data.frame(severity_instructions)) {
    stop(
      "`severity_instructions` must be a data.frame.",
      call. = FALSE
    )
  }

  required_severity_columns <- c(
    "min",
    "max",
    "instructions"
  )

  missing_severity_columns <- setdiff(
    required_severity_columns,
    names(severity_instructions)
  )

  if (length(missing_severity_columns) > 0L) {
    stop(
      "`severity_instructions` is missing required column(s): ",
      paste(
        missing_severity_columns,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  severity_instructions$min <- as.numeric(
    severity_instructions$min
  )

  severity_instructions$max <- as.numeric(
    severity_instructions$max
  )

  severity_instructions$instructions <- as.character(
    severity_instructions$instructions
  )

  if (
    anyNA(severity_instructions$min) ||
    anyNA(severity_instructions$max) ||
    any(!is.finite(severity_instructions$min)) ||
    any(!is.finite(severity_instructions$max))
  ) {
    stop(
      "Severity instruction ranges must be finite numeric values.",
      call. = FALSE
    )
  }

  if (
    any(
      severity_instructions$min >
      severity_instructions$max
    )
  ) {
    stop(
      "Each `min` must be less than or equal to its corresponding `max`.",
      call. = FALSE
    )
  }

  has_instructions <- vapply(
    severity,
    function(x) {
      any(
        x >= severity_instructions$min &
          x <= severity_instructions$max
      )
    },
    logical(1)
  )

  if (!all(has_instructions)) {
    stop(
      "Some severity values do not have corresponding severity instructions.",
      call. = FALSE
    )
  }

  if (nrow(severity_instructions) > 1L) {

    rules <- severity_instructions[
      order(
        severity_instructions$min,
        severity_instructions$max
      ),
      ,
      drop = FALSE
    ]

    overlap <- (
      rules$min[-1L] <=
        rules$max[-nrow(rules)]
    )

    if (any(overlap)) {
      stop(
        "`severity_instructions` contains overlapping ranges.",
        call. = FALSE
      )
    }
  }


  if (severity_min >= severity_max) {
    stop(
      "`severity_min` must be smaller than `severity_max`.",
      call. = FALSE
    )
  }

  if (
    batch_size < 1 ||
    batch_size != as.integer(batch_size)
  ) {
    stop(
      "`batch_size` must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    max_retries < 0 ||
    max_retries != as.integer(max_retries)
  ) {
    stop(
      "`max_retries` must be a non-negative integer.",
      call. = FALSE
    )
  }

  if (
    num_predict < 1 ||
    num_predict != as.integer(num_predict)
  ) {
    stop(
      "`num_predict` must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    num_ctx < 1 ||
    num_ctx != as.integer(num_ctx)
  ) {
    stop(
      "`num_ctx` must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    min_words < 0 ||
    min_words != as.integer(min_words)
  ) {
    stop(
      "`min_words` must be a non-negative integer.",
      call. = FALSE
    )
  }

  if (
    max_words < 1 ||
    max_words != as.integer(max_words)
  ) {
    stop(
      "`max_words` must be a positive integer.",
      call. = FALSE
    )
  }

  if (min_words > max_words) {
    stop(
      "`min_words` cannot exceed `max_words`.",
      call. = FALSE
    )
  }

  if (
    temperature < 0 ||
    !is.finite(temperature)
  ) {
    stop(
      "`temperature` must be non-negative.",
      call. = FALSE
    )
  }

  if (
    top_p <= 0 ||
    top_p > 1 ||
    !is.finite(top_p)
  ) {
    stop(
      "`top_p` must be greater than 0 and no greater than 1.",
      call. = FALSE
    )
  }

  if (
    repeat_penalty <= 0 ||
    !is.finite(repeat_penalty)
  ) {
    stop(
      "`repeat_penalty` must be greater than 0.",
      call. = FALSE
    )
  }

  python_executable <- Sys.which(
    python_path
  )

  if (nzchar(python_executable)) {

    python_path <- unname(
      python_executable
    )

  } else {

    python_path <- path.expand(
      python_path
    )

    if (!file.exists(python_path)) {
      stop(
        "Could not find Python executable: ",
        python_path,
        call. = FALSE
      )
    }
  }

  python_script <- normalizePath(
    python_script,
    mustWork = TRUE
  )


  task_dir <- tempfile(
    pattern = "text_datagen_"
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

  system_prompt_file <- file.path(
    task_dir,
    "system_prompt.txt"
  )

  prompt_template_file <- file.path(
    task_dir,
    "prompt_template.txt"
  )

  items_file <- file.path(
    task_dir,
    "items.txt"
  )

  severity_instructions_file <- file.path(
    task_dir,
    "severity_instructions.csv"
  )

  label_order_file <- file.path(
    task_dir,
    "label_order.txt"
  )

  label_order_double_file <- file.path(
    task_dir,
    "label_order_double.txt"
  )


  writeLines(
    system_prompt,
    system_prompt_file,
    useBytes = TRUE
  )

  writeLines(
    prompt_template,
    prompt_template_file,
    useBytes = TRUE
  )

  writeLines(
    items,
    items_file,
    useBytes = TRUE
  )

  utils::write.csv(
    severity_instructions,
    severity_instructions_file,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  writeLines(
    as.character(label_order),
    label_order_file,
    useBytes = TRUE
  )

  writeLines(
    as.character(label_order_double),
    label_order_double_file,
    useBytes = TRUE
  )


  output_directory <- dirname(
    normalizePath(
      output_csv,
      mustWork = FALSE
    )
  )

  dir.create(
    output_directory,
    recursive = TRUE,
    showWarnings = FALSE
  )


  python_args <- c(
    shQuote(python_script),

    shQuote(
      normalizePath(
        prompt_info_csv,
        mustWork = TRUE
      )
    ),

    shQuote(
      normalizePath(
        output_csv,
        mustWork = FALSE
      )
    ),

    "--examples",
    shQuote(
      normalizePath(
        examples_csv,
        mustWork = TRUE
      )
    ),

    "--model",
    shQuote(model),

    "--system_prompt",
    shQuote(
      normalizePath(
        system_prompt_file,
        mustWork = TRUE
      )
    ),

    "--prompt_template",
    shQuote(
      normalizePath(
        prompt_template_file,
        mustWork = TRUE
      )
    ),

    "--items",
    shQuote(
      normalizePath(
        items_file,
        mustWork = TRUE
      )
    ),

    "--severity_instructions",
    shQuote(
      normalizePath(
        severity_instructions_file,
        mustWork = TRUE
      )
    ),

    "--label_order",
    shQuote(
      normalizePath(
        label_order_file,
        mustWork = TRUE
      )
    ),

    "--label_order_double",
    shQuote(
      normalizePath(
        label_order_double_file,
        mustWork = TRUE
      )
    ),

    "--severity_min",
    as.character(severity_min),

    "--severity_max",
    as.character(severity_max),

    "--batch_size",
    as.character(
      as.integer(batch_size)
    ),

    "--max_retries",
    as.character(
      as.integer(max_retries)
    ),

    "--temperature",
    as.character(temperature),

    "--top_p",
    as.character(top_p),

    "--repeat_penalty",
    as.character(repeat_penalty),

    "--num_predict",
    as.character(
      as.integer(num_predict)
    ),

    "--num_ctx",
    as.character(
      as.integer(num_ctx)
    ),

    "--min_words",
    as.character(
      as.integer(min_words)
    ),

    "--max_words",
    as.character(
      as.integer(max_words)
    )
  )

  if (isTRUE(require_single_paragraph)) {

    python_args <- c(
      python_args,
      "--require_single_paragraph"
    )
  }


  command_output <- system2(
    command = python_path,
    args = python_args,
    stdout = TRUE,
    stderr = TRUE
  )

  command_status <- attr(
    command_output,
    "status"
  )

  if (is.null(command_status)) {
    command_status <- 0L
  }

  if (
    isTRUE(verbose) &&
    length(command_output) > 0L
  ) {
    cat(
      paste(
        command_output,
        collapse = "\n"
      ),
      "\n"
    )
  }



  if (command_status != 0L) {

    stop(
      paste0(
        "`text_datagen.py` failed with status ",
        command_status,
        ".\n",
        paste(
          command_output,
          collapse = "\n"
        )
      ),
      call. = FALSE
    )
  }



  if (!file.exists(output_csv)) {
    stop(
      "`text_datagen.py` did not create the expected output CSV.",
      call. = FALSE
    )
  }

  results <- utils::read.csv(
    output_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  if (!"response" %in% names(results)) {
    stop(
      "Generated output does not contain a `response` column.",
      call. = FALSE
    )
  }



  return(results)
}
