# =============================================================================
# 02_training.R
# Step 2: define a calibrator and train/load a calibration method
#
# This file contains:
#   - the generic calibrator object/interface constructor
#   - the generic BiLSTM adapter
#   - compatibility with the original pretrained SIR BiLSTM
#   - training-data saving and generic BiLSTM training
#
# The calibration method is independent of the epidemic model. A future
# Transformer, CNN, Bayesian model, or other method can implement the same
# list(...) |> structure(class = "calibrator") interface.
# =============================================================================

# Generic calibrator interface -------------------------------------------------
#
# A calibrator is deliberately just a list of three functions plus a model
# label.  This is the public extension mechanism: users do not need to inherit
# from a complicated R6/S4 class or modify package internals.
#
# The required shape is:
#
# calibrator_x <- list(
#   pre_process = function(x, ...) { ... },
#   run          = function(x, ...) { ... },
#   post_process = function(x, ...) { ... },
#   type         = "MODEL"
# ) |> structure(class = "calibrator")

#' @export
print.calibrator <- function(x, ...) {
  cat("Calibrator object for", x$type, "models\n")
  invisible(x)
}


# Generic BiLSTM calibrator ---------------------------------------------------

# Post-processing used by the original pretrained SIR model: keep the predicted
# ptran and R0, and recompute crate from R0 = ptran * crate / recov. Models
# trained without a crate target get it added this way. Models without ptran
# and R0 targets or a known recov are left unchanged.
derive_crate <- function(x, recov = NULL, ...) {
  if (!is.null(recov) && all(c("ptran", "R0") %in% names(x))) {
    x["crate"] <- x["R0"] * recov / x["ptran"]
  }
  x
}

#' Create a BiLSTM calibrator from a saved model directory
#'
#' The saved model determines the number of inputs and outputs.  `known_names`
#' tells the R wrapper which named values from `...` should be sent to Python.
#' Nothing here assumes SIR or SEIR. Targets trained on the log scale, named
#' `log_<name>` (for example `log_preval`), are returned as `<name>` on the
#' original scale.
#'
#' @param model_dir Directory containing model.pt and the saved scalers.
#' @param known_names Names of known parameters expected by the model.
#' @param type Epidemiological model label.
#' @param name Calibrator label.
#' @param post_process Result transformation. By default, the same as the
#'   pretrained SIR model: when the model predicts `ptran` and `R0` and
#'   `recov` is known, `crate` is set to `R0 * recov / ptran` (replacing the
#'   network's own `crate`, or added when the model has no `crate` target).
#'   Use `function(x, ...) x` to keep the raw network output.
#' @param device Where to run the model: `"auto"` (a GPU when PyTorch can use
#'   one, otherwise the CPU), `"cuda"` or `"cpu"`.
#' @return A `calibrator` object.
#' @export
bilstm_calibrator <- function(
    model_dir,
    known_names,
    type = "generic",
    name = "BiLSTM",
    post_process = derive_crate,
    device = "auto"
) {
  model_dir <- normalizePath(model_dir, mustWork = TRUE)

  pre_process <- function(daily_cases, ...) {
    dots <- list(...)
    missing <- setdiff(known_names, names(dots))
    if (length(missing)) {
      stop("Missing known parameter(s): ", paste(missing, collapse = ", "))
    }

    list(
      epicurve = as.numeric(daily_cases),
      known = as.numeric(unlist(dots[known_names], use.names = FALSE))
    )
  }

  run <- function(x, ...) {
    py <- load_python_script("bilstm.py")
    ans <- py$predict_saved_model(
      x$epicurve,
      matrix(x$known, nrow = 1),
      model_dir,
      device
    )
    unlog_targets(unlist(ans))
  }

  list(
    pre_process = pre_process,
    run = run,
    post_process = post_process,
    type = type
  ) |>
    structure(class = "calibrator")
}

# Targets trained on the log scale are named log_<name>; return them as <name>.
unlog_targets <- function(x) {
  logs <- startsWith(names(x), "log_")
  if (is.data.frame(x)) x[logs] <- lapply(x[logs], exp) else x[logs] <- exp(x[logs])
  names(x)[logs] <- sub("^log_", "", names(x)[logs])
  x
}

# Estimates for many curves at once from a BiLSTM model: one row per curve
# (one row of `known` each), log_ targets converted as in bilstm_calibrator(),
# no post-processing. `curves` is a matrix, or, for a model trained on windows
# of any length, a list of curves of any lengths. Much faster than calling
# calibrate() curve by curve.
bilstm_predict_many <- function(model_dir, curves, known, device = "auto") {
  py <- load_python_script("bilstm.py")
  ans <- py$predict_saved_model_batch(
    if (is.list(curves) && !is.data.frame(curves)) lapply(curves, as.numeric) else as.matrix(curves),
    as.matrix(known),
    normalizePath(model_dir, mustWork = TRUE),
    device
  )
  unlog_targets(as.data.frame(lapply(ans, as.numeric), check.names = FALSE))
}

#' Create the original pretrained SIR BiLSTM calibrator
#'
#' This is a compatibility adapter for the already-trained 61-day SIR model.
#' It is separate from the generic BiLSTM engine because the original checkpoint
#' used its own architecture and saved scalers.
#'
#' @export
pretrained_sir_calibrator <- function(model_dir = NULL) {
  if (is.null(model_dir)) {
    model_dir <- package_file("models", "sir", "pretrained")
  }
  model_dir <- normalizePath(model_dir, mustWork = TRUE)

  pre_process <- function(daily_cases, n, recov, ...) {
    list(
      epicurve = as.numeric(daily_cases),
      n = as.numeric(n),
      recov = as.numeric(recov)
    )
  }

  run <- function(x, ...) {
    py <- load_python_script("legacy_sir.py")
    unlist(py$predict_pretrained_sir(
      x$epicurve,
      x$n,
      x$recov,
      model_dir
    ))
  }

  list(
    pre_process = pre_process,
    run = run,
    post_process = derive_crate,
    type = "SIR"
  ) |>
    structure(class = "calibrator")
}


# Generic training helpers ----------------------------------------------------

#' Save a training bank
#'
#' The Python trainer only needs a parameter table and an observed-curve matrix.
#' `name` can be "sir", "seir", "sirh", or any future model label.
#' @export
save_training_data <- function(x, name, output_dir = tempdir()) {
  if (is.null(x$theta) || !is.data.frame(x$theta)) {
    stop("x$theta must be a data.frame.")
  }

  observations <- x$observations
  if (is.null(observations)) observations <- x$incidence
  if (is.null(observations)) {
    stop("x must contain an `observations` matrix.")
  }
  observations <- as.matrix(observations)

  if (nrow(observations) != nrow(x$theta)) {
    stop("theta and observations must have the same number of rows.")
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  theta_path <- file.path(output_dir, paste0(name, "_theta.csv"))
  observations_path <- file.path(output_dir, paste0(name, "_observations.csv"))

  write.csv(x$theta, theta_path, row.names = FALSE)
  write.csv(observations, observations_path, row.names = FALSE)

  invisible(c(theta = theta_path, observations = observations_path))
}

#' Train the generic BiLSTM calibrator
#'
#' @param name Label used for input/output file names.
#' @param known_names Columns in the parameter table treated as known inputs.
#' @param target_names Columns the neural network should estimate.
#' @param type Epidemiological model label written to model_config.json.
#' @param data Optional `training_data` object. If supplied, it is saved to a
#'   temporary folder and used directly; no package simulator is required.
#' @param parameter_file Optional external CSV containing parameter rows.
#' @param observation_file Optional external CSV containing observation rows.
#' @param output_dir Directory the trained model is written to. Defaults to a
#'   temporary folder; pass your own folder to keep the model after the R
#'   session ends.
#' @param epochs Number of epochs.
#' @param batch_size Batch size.
#' @param learning_rate Adam learning rate.
#' @param seed Random seed.
#' @param hidden_size BiLSTM hidden dimension.
#' @param num_layers Number of BiLSTM layers.
#' @param dropout LSTM dropout.
#' @param physics_weight Optional SIR identity penalty. Set to zero for a fully
#'   generic model.  It is only used when the required columns are present.
#' @param incidence_max Observations are divided by this value before they
#'   enter the network. The original pretrained SIR model used 10000.
#' @param incidence_transform `"none"` (default) or `"log1p"`. With `"log1p"`
#'   the network reads `log(1 + cases) / log(1 + incidence_max)`, which keeps
#'   small counts readable; the saved model does the same to new curves.
#' @param output_activation Activation applied to the scaled outputs:
#'   `"sigmoid"` or `"softplus"`. Either one value for all targets or one per
#'   target, in the order of `target_names`. The original pretrained SIR model
#'   used `c("sigmoid", "softplus", "softplus")` for ptran, crate and R0.
#' @param device Where to train: `"auto"` (a GPU when PyTorch can use one,
#'   otherwise the CPU), `"cpu"` or `"cuda"`. The saved model always runs on
#'   the CPU.
#' @param window_min,window_max Optional. Train on random windows of each curve
#'   instead of whole curves: every batch gets a window length drawn uniformly
#'   between `window_min` and `window_max` days, and every curve a start drawn
#'   uniformly over the positions where the window fits. The saved model then
#'   accepts curves of any length. `window_max` defaults to the curve length.
#' @param window_stop_zeros Optional, with windows. Each curve's informative
#'   days end at its first run of this many consecutive zero days (for example
#'   7: a week without cases); windows are drawn only from the informative days
#'   and the rest of the curve is ignored. The saved model cuts new curves the
#'   same way.
#' @param state_files Optional, with windows. A named character vector of CSV
#'   files, one per state (same rows and days as the observations), holding the
#'   value of that state at the start of each day, e.g.
#'   `c(infected_share = "infected.csv")`. Each becomes an extra estimated
#'   output: its value at the start of the window. `output_activation` then
#'   needs one entry per target plus one per state.
#' @param grad_clip Optional maximum norm of the gradient in each update
#'   (gradient clipping). Steadies training on long sequences.
#' @param lr_schedule `"none"` (default, a constant learning rate) or
#'   `"cosine"`: the learning rate falls smoothly from `learning_rate` to 1% of
#'   it over the epochs.
#' @param resume If TRUE, a checkpoint is saved in `output_dir` after every
#'   epoch, and training continues from it when it is run again with the same
#'   settings (for long runs that may be interrupted).
#' @param observation_noise If TRUE, the observations are true daily cases and
#'   the parameter table has `report_rate`, `overdispersion` and `weekday_sd`
#'   (see [seir_simulator()]): every training batch gets newly drawn reported
#'   cases, so the network never sees the same noise twice (the validation
#'   curves get one fixed draw). The saved model reads reported cases.
#'   Fixed-length curves only.
#' @return Invisibly, the path to the trained model directory.
#' @export
train_bilstm_calibrator <- function(
    name,
    known_names,
    target_names,
    type = name,
    data = NULL,
    parameter_file = NULL,
    observation_file = NULL,
    output_dir = file.path(tempdir(), name),
    epochs = 100,
    batch_size = 64,
    learning_rate = 0.001,
    seed = 122,
    hidden_size = 160,
    num_layers = 3,
    dropout = 0.5,
    physics_weight = 0,
    incidence_max = 100000,
    incidence_transform = "none",
    output_activation = "sigmoid",
    device = "auto",
    window_min = NULL,
    window_max = NULL,
    window_stop_zeros = NULL,
    state_files = NULL,
    grad_clip = NULL,
    observation_noise = FALSE,
    lr_schedule = "none",
    resume = FALSE
) {
  script <- package_file("python", "train_bilstm.py")

  n_outputs <- length(target_names) + length(state_files)
  if (!length(output_activation) %in% c(1L, n_outputs)) {
    stop(
      "output_activation must have length 1 or one value per output (",
      n_outputs, ")."
    )
  }
  if (length(state_files) && (is.null(names(state_files)) || any(names(state_files) == ""))) {
    stop("state_files must be a named character vector.")
  }
  incidence_transform <- match.arg(incidence_transform, c("none", "log1p"))
  lr_schedule <- match.arg(lr_schedule, c("none", "cosine"))
  bad <- setdiff(output_activation, c("sigmoid", "softplus"))
  if (length(bad)) {
    stop("Unknown output_activation: ", paste(bad, collapse = ", "))
  }
  python <- reticulate::py_config()$python

  # Two supported input modes:
  #   1. `data=`: pass a training_data object directly;
  #   2. `parameter_file=` + `observation_file=`: use external CSV files.
  if (!is.null(data)) {
    if (is.null(data$theta) || is.null(data$observations)) {
      stop("data must contain `theta` and `observations`.")
    }
    paths <- save_training_data(
      data, name = name, output_dir = tempfile("training_data_")
    )
    theta <- paths[["theta"]]
    observations <- paths[["observations"]]
  } else if (!is.null(parameter_file) && !is.null(observation_file)) {
    theta <- normalizePath(parameter_file, mustWork = TRUE)
    observations <- normalizePath(observation_file, mustWork = TRUE)
  } else {
    stop(
      "Supply either `data`, or both `parameter_file` and `observation_file`."
    )
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir, mustWork = TRUE)

  if (!file.exists(theta)) stop("Missing training file: ", theta)
  if (!file.exists(observations)) {
    stop("Missing training observations: ", observations)
  }

  args <- c(
    script,
    "--theta", theta,
    "--curves", observations,
    "--output-dir", output_dir,
    "--known", paste(known_names, collapse = ","),
    "--targets", paste(target_names, collapse = ","),
    "--type", type,
    "--epochs", as.integer(epochs),
    "--batch-size", as.integer(batch_size),
    "--learning-rate", learning_rate,
    "--seed", as.integer(seed),
    "--hidden-size", as.integer(hidden_size),
    "--num-layers", as.integer(num_layers),
    "--dropout", dropout,
    "--physics-weight", physics_weight,
    "--incidence-max", incidence_max,
    "--incidence-transform", incidence_transform,
    "--output-activation", paste(output_activation, collapse = ","),
    "--device", device,
    if (!is.null(window_min)) c("--window-min", as.integer(window_min)),
    if (!is.null(window_max)) c("--window-max", as.integer(window_max)),
    if (!is.null(window_stop_zeros)) c("--window-stop-zeros", as.integer(window_stop_zeros)),
    if (length(state_files)) c("--state-files", paste0(
      names(state_files), "=", normalizePath(state_files, mustWork = TRUE), collapse = ","
    )),
    if (!is.null(grad_clip)) c("--grad-clip", grad_clip),
    if (isTRUE(observation_noise)) "--observation-noise",
    "--lr-schedule", lr_schedule,
    if (isTRUE(resume)) "--resume"
  )

  status <- system2(python, args = args)
  if (!identical(status, 0L)) stop("Python training failed with status ", status)

  invisible(output_dir)
}
