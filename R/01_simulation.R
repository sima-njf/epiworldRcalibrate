# =============================================================================
# 01_simulation.R
# Step 1: define/import a simulator and create training data
#
# This file contains:
#   - internal path/Python helpers used by the package
#   - generic simulator interface
#   - generic user-supplied training-data interface
#   - built-in epiworldR SIR and SEIR simulators
#   - SIR/SEIR training-bank generators
#
# A model does NOT have to exist in epiworldR. Users can supply their own
# parameter table + observation matrix, or wrap their own simulator with
# new_simulator().
# =============================================================================

# Internal path helpers -------------------------------------------------------

# Files under inst/ are installed at the top level of the package, so paths
# here leave out "inst" (e.g. inst/python/bilstm.py -> python/bilstm.py).
package_file <- function(...) {
  path <- system.file(..., package = "epiworldRcalibrate")
  if (!nzchar(path)) stop("Cannot find package file: ", file.path(...))
  path
}

# Each script is sourced once per session; later calls reuse it (and the
# models it has already loaded).
python_scripts <- new.env(parent = emptyenv())

load_python_script <- function(filename) {
  if (is.null(python_scripts[[filename]])) {
    path <- package_file("python", filename)
    env <- new.env(parent = globalenv())
    reticulate::source_python(path, envir = env, convert = TRUE)
    python_scripts[[filename]] <- env
  }
  python_scripts[[filename]]
}

setup_python_deps <- function(force = FALSE) {
  packages <- c("numpy", "pandas", "torch", "scikit-learn==1.6.1", "joblib", "optuna>=4")
  reticulate::py_require(packages, action = if (force) "set" else "add")
  invisible(TRUE)
}

# Declare the Python packages when the package loads, so reticulate sets them
# up the first time Python is used.
.onLoad <- function(libname, pkgname) {
  setup_python_deps()
}


# Generic simulation interface ------------------------------------------------
#
# Data generation is deliberately separated from both the epidemiological
# model and the calibration method.  A simulator object only needs two pieces:
#   1. a function that samples parameter sets;
#   2. a function that turns one parameter set into one numeric epidemic curve.
#
# The generic simulate_training_data() function does not know what SIR, SEIR,
# SIRH, hospitalization, or any other state means.

#' Create a simulator object
#'
#' @param sample_parameters Function with argument `n_sims` that returns a data
#'   frame with one row per simulated parameter set.
#' @param run Function with arguments `parameters` and `seed`. `parameters` is a
#'   named list containing one row of the sampled parameter table. The function
#'   must return one numeric epidemic curve.
#' @param type Label for the epidemiological model, e.g. "SIR", "SEIR", "SIRH".
#' @param name Human-readable simulator name.
#' @return An object of class `simulator`.
#' @export
new_simulator <- function(
    sample_parameters,
    run,
    type = "generic",
    name = NULL
) {
  if (!is.function(sample_parameters)) {
    stop("sample_parameters must be a function.")
  }
  if (!is.function(run)) {
    stop("run must be a function.")
  }

  structure(
    list(
      sample_parameters = sample_parameters,
      run = run,
      type = type,
      name = name
    ),
    class = "simulator"
  )
}

#' @export
print.simulator <- function(x, ...) {
  cat(
    "Simulator:",
    if (!is.null(x$name)) x$name else "unnamed",
    "\n"
  )
  cat("Model type:", x$type, "\n")
  invisible(x)
}


#' Create a generic training-data object
#'
#' Use this when the package cannot simulate the user's epidemiological model.
#' The user supplies one parameter row and one observed time series per
#' simulation. `observations` can represent cases, hospitalizations, deaths,
#' prevalence, or any other numeric series used by the calibrator.
#'
#' @param parameters Data frame with one row per simulation and named columns.
#' @param observations Numeric matrix/data.frame with one row per simulation.
#' @param type Optional epidemiological model label.
#' @param source Optional description of where the training data came from.
#' @return An object of class `training_data`.
#' @export
new_training_data <- function(
    parameters,
    observations,
    type = "generic",
    source = "user"
) {
  if (!is.data.frame(parameters)) {
    parameters <- as.data.frame(parameters)
  }
  if (!nrow(parameters) || !ncol(parameters)) {
    stop("parameters must contain at least one row and one column.")
  }
  if (is.null(names(parameters)) || any(names(parameters) == "")) {
    stop("parameters must have named columns.")
  }
  if (anyDuplicated(names(parameters))) {
    stop("parameters contains duplicated column names.")
  }

  observations <- as.matrix(observations)
  storage.mode(observations) <- "double"

  if (!nrow(observations) || !ncol(observations)) {
    stop("observations must contain at least one row and one column.")
  }
  if (nrow(observations) != nrow(parameters)) {
    stop(
      "parameters and observations must have the same number of rows. ",
      "Got ", nrow(parameters), " and ", nrow(observations), "."
    )
  }
  if (any(!is.finite(observations))) {
    stop("observations contains non-finite values.")
  }

  if (is.null(colnames(observations))) {
    colnames(observations) <- paste0("time_", seq_len(ncol(observations)))
  }

  structure(
    list(
      theta = parameters,
      observations = observations,
      # Backward-compatible alias for older code.
      incidence = observations
    ),
    class = "training_data",
    training_type = type,
    training_source = source
  )
}

#' @export
print.training_data <- function(x, ...) {
  cat("Training data\n")
  cat("  model type:", attr(x, "training_type"), "\n")
  cat("  source:", attr(x, "training_source"), "\n")
  cat("  simulations:", nrow(x$theta), "\n")
  cat("  parameters:", ncol(x$theta), "\n")
  cat("  observation length:", ncol(x$observations), "\n")
  invisible(x)
}

#' Read externally generated training data
#'
#' @param parameter_file CSV file containing one parameter row per simulation.
#' @param observation_file CSV file containing one observed time series per row.
#' @param type Optional epidemiological model label.
#' @export
read_training_data <- function(
    parameter_file,
    observation_file,
    type = "generic"
) {
  parameters <- read.csv(parameter_file, check.names = FALSE)
  observations <- as.matrix(
    read.csv(observation_file, check.names = FALSE)
  )

  new_training_data(
    parameters = parameters,
    observations = observations,
    type = type,
    source = "external files"
  )
}

#' Generate a generic simulation bank
#'
#' This function is model-agnostic. It samples parameter rows from the supplied
#' simulator and calls the simulator once for each row.
#'
#' @param simulator A `simulator` object created with `new_simulator()`.
#' @param n_sims Number of simulations.
#' @param seed Master seed used for parameter sampling and per-simulation seeds.
#' @return A list with `theta` (parameter table) and `observations` (curve matrix).
#' @export
simulate_training_data <- function(
    simulator,
    n_sims,
    seed = 122
) {
  if (!inherits(simulator, "simulator")) {
    stop(
      "The object 'simulator' must be of class 'simulator'. It is of class: ",
      paste(class(simulator), collapse = ", ")
    )
  }

  n_sims <- as.integer(n_sims)
  if (length(n_sims) != 1L || is.na(n_sims) || n_sims < 1L) {
    stop("n_sims must be a positive integer.")
  }

  set.seed(seed)
  theta <- simulator$sample_parameters(n_sims = n_sims)

  if (!is.data.frame(theta)) {
    stop("simulator$sample_parameters() must return a data.frame.")
  }
  if (nrow(theta) != n_sims) {
    stop(
      "simulator$sample_parameters() returned ", nrow(theta),
      " rows, but n_sims = ", n_sims, "."
    )
  }
  if (!ncol(theta)) {
    stop("The sampled parameter table has no columns.")
  }
  if (anyDuplicated(names(theta))) {
    stop("The sampled parameter table contains duplicated column names.")
  }

  simulation_seeds <- sample.int(.Machine$integer.max, n_sims)
  curves <- vector("list", n_sims)

  for (i in seq_len(n_sims)) {
    parameters <- as.list(theta[i, , drop = FALSE])

    curve <- simulator$run(
      parameters = parameters,
      seed = simulation_seeds[i]
    )

    if (!is.numeric(curve)) {
      stop("Simulation ", i, " did not return a numeric curve.")
    }
    if (!length(curve)) {
      stop("Simulation ", i, " returned an empty curve.")
    }
    if (any(!is.finite(curve))) {
      stop("Simulation ", i, " returned non-finite values.")
    }

    curves[[i]] <- as.numeric(curve)
    names(curves[[i]]) <- names(curve)
  }

  curve_lengths <- vapply(curves, length, integer(1))
  if (length(unique(curve_lengths)) != 1L) {
    stop(
      "All simulated curves must have the same length. Observed lengths: ",
      paste(sort(unique(curve_lengths)), collapse = ", ")
    )
  }

  incidence <- do.call(rbind, curves)

  curve_names <- names(curves[[1]])
  if (!is.null(curve_names) && length(curve_names) == ncol(incidence)) {
    colnames(incidence) <- curve_names
  } else {
    colnames(incidence) <- paste0("time_", seq_len(ncol(incidence)))
  }

  rownames(incidence) <- NULL

  ans <- new_training_data(
    parameters = theta,
    observations = incidence,
    type = simulator$type,
    source = if (!is.null(simulator$name)) simulator$name else "simulator"
  )
  attr(ans, "simulator_type") <- simulator$type
  attr(ans, "simulator_name") <- simulator$name
  ans
}

#' Simulate one curve from a simulator
#'
#' @param simulator A `simulator` object.
#' @param parameters Named list or one-row data frame of parameter values.
#' @param seed Simulation seed.
#' @export
simulate_from_parameters <- function(
    simulator,
    parameters,
    seed = 1
) {
  if (!inherits(simulator, "simulator")) {
    stop("simulator must be of class 'simulator'.")
  }

  if (is.data.frame(parameters)) {
    if (nrow(parameters) != 1L) {
      stop("parameters must contain exactly one row.")
    }
    parameters <- as.list(parameters[1, , drop = FALSE])
  }

  if (!is.list(parameters) || is.null(names(parameters))) {
    stop("parameters must be a named list or one-row data.frame.")
  }

  curve <- simulator$run(parameters = parameters, seed = seed)
  if (!is.numeric(curve) || !length(curve) || any(!is.finite(curve))) {
    stop("The simulator did not return a valid numeric curve.")
  }
  curve
}

# epiworldR examples ----------------------------------------------------------
# These are adapters for the two models currently used in this project.  They
# are examples built on top of the generic simulator API; the generic API does
# not depend on SIR or SEIR.

get_incidence <- function(model, ndays, keep_day_zero = FALSE) {
  x <- as.numeric(epiworldR::plot_incidence(model, plot = FALSE)[, "Infected"])
  if (!keep_day_zero) x <- x[-1]
  expected <- if (keep_day_zero) ndays + 1L else ndays
  if (length(x) < expected) x <- c(x, rep(0, expected - length(x)))
  x[seq_len(expected)]
}

log_uniform <- function(n, range) {
  exp(stats::runif(n, log(range[1]), log(range[2])))
}

# Evaluate `expr` with the random seed `seed`, then restore the caller's
# random number state.
with_seed <- function(seed, expr) {
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) old <- get(".Random.seed", envir = globalenv())
  on.exit(
    if (had_seed) assign(".Random.seed", old, envir = globalenv())
    else rm(".Random.seed", envir = globalenv())
  )
  set.seed(seed)
  expr
}

# Reported cases from true daily cases: each day's expected count is the
# reported share times the true count times a weekday factor (7 random factors
# with mean 1, so the weekly pattern starts on a random weekday), and the
# reported count is negative binomial around it (Poisson when overdispersion
# is 0): variance = mean + overdispersion * mean^2.
observe_cases <- function(x, report_rate, overdispersion = 0, weekday_sd = 0) {
  w <- exp(weekday_sd * stats::rnorm(7))
  w <- w / mean(w)
  mu <- report_rate * x * w[(seq_along(x) - 1L) %% 7L + 1L]
  if (overdispersion > 0) {
    stats::rnbinom(length(mu), size = 1 / overdispersion, mu = mu)
  } else {
    stats::rpois(length(mu), mu)
  }
}

merge_priors <- function(defaults, priors) {
  if (is.null(priors)) return(defaults)
  unknown <- setdiff(names(priors), names(defaults))
  if (length(unknown)) {
    stop("Unknown prior(s): ", paste(unknown, collapse = ", "))
  }
  defaults[names(priors)] <- priors
  defaults
}

sample_sir_theta <- function(n_sims, p) {
  out <- data.frame()

  while (nrow(out) < n_sims) {
    n_need <- n_sims - nrow(out)
    n_draw <- max(2L * n_need, 10L)

    z <- data.frame(
      n = sample(p$n[1]:p$n[2], n_draw, replace = TRUE),
      recov = runif(n_draw, p$recov[1], p$recov[2]),
      preval = runif(n_draw, p$preval[1], p$preval[2]),
      crate = runif(n_draw, p$crate[1], p$crate[2]),
      R0 = runif(n_draw, p$R0[1], p$R0[2])
    )

    z$ptran <- z$R0 * z$recov / z$crate
    z <- z[z$ptran <= 1, , drop = FALSE]
    out <- rbind(out, z)
  }

  out <- out[seq_len(n_sims), , drop = FALSE]
  rownames(out) <- NULL
  out$initial_infected <- pmax(1L, round(out$preval * out$n))
  if (!is.null(p$report_rate)) out$report_rate <- log_uniform(n_sims, p$report_rate)
  if (!is.null(p$overdispersion)) out$overdispersion <- log_uniform(n_sims, p$overdispersion)
  if (!is.null(p$weekday_sd)) out$weekday_sd <- runif(n_sims, p$weekday_sd[1], p$weekday_sd[2])
  out
}

#' Create the current epiworldR SIR simulator
#'
#' `parameters$preval` is the share of the population initially infected. To
#' start mid-epidemic, `parameters` may also hold `start_recovered`, the share
#' initially recovered. With `parameters$report_rate`, the curve holds reported
#' cases, as in [seir_simulator()].
#'
#' @param ndays Number of simulated days. The current SIR adapter preserves the
#'   historical day-zero convention and returns `ndays + 1` values.
#' @param priors Optional named list overriding the default prior ranges
#'   (`c(min, max)` each): `n`, `recov`, `preval`, `crate`, `R0`. Optional
#'   extras, as in [seir_simulator()]: `report_rate` and `overdispersion`,
#'   drawn log-uniformly, and `weekday_sd`, drawn uniformly.
#' @param keep_day_zero If TRUE (the historical convention), the curve starts
#'   with day 0, the number initially infected, followed by new infections on
#'   days 1 to `ndays` (`ndays + 1` values). If FALSE, the curve holds only new
#'   infections on days 1 to `ndays`, like the SEIR adapter.
#' @export
sir_simulator <- function(
    ndays = 60,
    priors = NULL,
    keep_day_zero = TRUE
) {
  p <- merge_priors(
    list(
      n = c(5000, 10000),
      recov = c(0.071, 0.25),
      preval = c(0.007, 0.02),
      crate = c(1, 5),
      R0 = c(1.1, 5),
      report_rate = NULL,
      overdispersion = NULL,
      weekday_sd = NULL
    ),
    priors
  )

  sample_parameters <- function(n_sims) {
    sample_sir_theta(n_sims, p)
  }

  run <- function(parameters, seed) {
    m <- epiworldR::ModelSIRCONN(
      name = "sim",
      n = parameters$n,
      prevalence = parameters$preval,
      contact_rate = parameters$crate,
      transmission_rate = parameters$ptran,
      recovery_rate = parameters$recov
    )
    if (!is.null(parameters$start_recovered)) {
      # epiworldR sets the recovered share of everyone not infected.
      epiworldR::initial_states(m, parameters$start_recovered / (1 - parameters$preval))
    }
    epiworldR::verbose_off(m)
    epiworldR::run(m, ndays = ndays, seed = seed)

    ans <- get_incidence(m, ndays, keep_day_zero = keep_day_zero)
    if (!is.null(parameters$report_rate)) {
      ans <- with_seed(seed, observe_cases(
        ans,
        report_rate    = parameters$report_rate,
        overdispersion = if (is.null(parameters$overdispersion)) 0 else parameters$overdispersion,
        weekday_sd     = if (is.null(parameters$weekday_sd)) 0 else parameters$weekday_sd
      ))
    }
    names(ans) <- paste0("day_", if (keep_day_zero) 0:ndays else seq_len(ndays))
    ans
  }

  new_simulator(
    sample_parameters = sample_parameters,
    run = run,
    type = "SIR",
    name = "epiworldR ModelSIRCONN"
  )
}

#' Create the current epiworldR SEIR simulator
#'
#' `parameters$preval` is the share of the population initially exposed. To
#' start mid-epidemic, `parameters` may also hold `start_infected` and
#' `start_recovered`: the shares initially infectious and recovered.
#'
#' With `parameters$report_rate`, the curve holds reported cases instead of all
#' new cases: each day's expected count is `report_rate` times the true count
#' times a weekday factor (seven random factors with mean 1 and log standard
#' deviation `weekday_sd`), and the count is negative binomial around it with
#' variance `mean + overdispersion * mean^2` (Poisson when `overdispersion` is
#' 0 or missing).
#'
#' @param ndays Number of returned incidence days.
#' @param priors Optional named list overriding the default prior ranges
#'   (`c(min, max)` each): `n`, `recov`, `crate`, `incub`, `R0` and
#'   `initial_infected` (the number initially exposed). Optional extras:
#'   `preval`, the share initially exposed, drawn log-uniformly (replaces
#'   `initial_infected`); `report_rate` and `overdispersion`, drawn
#'   log-uniformly, and `weekday_sd`, drawn uniformly, which switch on the
#'   reporting model above.
#' @export
seir_simulator <- function(
    ndays = 365,
    priors = NULL
) {
  p <- merge_priors(
    list(
      n = c(5000, 10000),
      recov = c(0.071, 0.25),
      crate = c(5, 15),
      incub = c(3, 21),
      R0 = c(1, 5),
      initial_infected = c(100, 2000),
      preval = NULL,
      report_rate = NULL,
      overdispersion = NULL,
      weekday_sd = NULL
    ),
    priors
  )

  sample_parameters <- function(n_sims) {
    theta <- data.frame(
      n = sample(p$n[1]:p$n[2], n_sims, replace = TRUE),
      recov = runif(n_sims, p$recov[1], p$recov[2]),
      crate = runif(n_sims, p$crate[1], p$crate[2]),
      incub = runif(n_sims, p$incub[1], p$incub[2]),
      R0 = runif(n_sims, p$R0[1], p$R0[2])
    )
    theta$initial_infected <- if (is.null(p$preval)) {
      sample(p$initial_infected[1]:p$initial_infected[2], n_sims, replace = TRUE)
    } else {
      pmax(1, round(log_uniform(n_sims, p$preval) * theta$n))
    }

    theta$preval <- theta$initial_infected / theta$n
    theta$ptran <- theta$R0 * theta$recov / theta$crate
    if (!is.null(p$report_rate)) theta$report_rate <- log_uniform(n_sims, p$report_rate)
    if (!is.null(p$overdispersion)) theta$overdispersion <- log_uniform(n_sims, p$overdispersion)
    if (!is.null(p$weekday_sd)) {
      theta$weekday_sd <- runif(n_sims, p$weekday_sd[1], p$weekday_sd[2])
    }
    theta
  }

  run <- function(parameters, seed) {
    m <- epiworldR::ModelSEIRCONN(
      name = "sim",
      n = parameters$n,
      prevalence = parameters$preval,
      contact_rate = parameters$crate,
      incubation_days = parameters$incub,
      transmission_rate = parameters$ptran,
      recovery_rate = parameters$recov
    )
    if (!is.null(parameters$start_infected) || !is.null(parameters$start_recovered)) {
      # epiworldR sets the starting state through the share carrying the virus
      # (exposed + infectious), the infectious share of the carriers, and the
      # recovered share of everyone else.
      exposed   <- parameters$preval
      infected  <- if (is.null(parameters$start_infected)) 0 else parameters$start_infected
      recovered <- if (is.null(parameters$start_recovered)) 0 else parameters$start_recovered
      carriers  <- exposed + infected
      m <- epiworldR::ModelSEIRCONN(
        name = "sim",
        n = parameters$n,
        prevalence = carriers,
        contact_rate = parameters$crate,
        incubation_days = parameters$incub,
        transmission_rate = parameters$ptran,
        recovery_rate = parameters$recov
      )
      epiworldR::initial_states(m, c(
        if (carriers > 0) infected / carriers else 0,
        recovered / (1 - carriers)
      ))
    }
    epiworldR::verbose_off(m)
    epiworldR::run(m, ndays = ndays + 1L, seed = seed)

    ans <- get_incidence(m, ndays, keep_day_zero = FALSE)
    if (!is.null(parameters$report_rate)) {
      ans <- with_seed(seed, observe_cases(
        ans,
        report_rate    = parameters$report_rate,
        overdispersion = if (is.null(parameters$overdispersion)) 0 else parameters$overdispersion,
        weekday_sd     = if (is.null(parameters$weekday_sd)) 0 else parameters$weekday_sd
      ))
    }
    names(ans) <- paste0("day_", seq_len(ndays))
    ans
  }

  new_simulator(
    sample_parameters = sample_parameters,
    run = run,
    type = "SEIR",
    name = "epiworldR ModelSEIRCONN"
  )
}


# Built-in training-bank generators -------------------------------------------
#
# These are convenience functions for the two models currently supported by
# epiworldR in this project. They are NOT required by the generic interface.

#' Generate the built-in SIR training bank
#'
#' @param n_sims Number of simulated parameter sets.
#' @param ndays Number of simulated days. Historical SIR output contains
#'   `ndays + 1` values including day zero.
#' @param seed Master random seed.
#' @param priors Optional prior overrides passed to `sir_simulator()`.
#' @param save If TRUE, save the bank as CSV files in `output_dir`.
#' @param name File/model label.
#' @param output_dir Directory the CSV files are written to when `save = TRUE`.
#' @param keep_day_zero Passed to `sir_simulator()`.
#' @export
generate_sir_training_data <- function(
    n_sims = 5000,
    ndays = 60,
    seed = 122,
    priors = NULL,
    save = FALSE,
    name = "sir",
    output_dir = tempdir(),
    keep_day_zero = TRUE
) {
  bank <- simulate_training_data(
    simulator = sir_simulator(ndays = ndays, priors = priors, keep_day_zero = keep_day_zero),
    n_sims = n_sims,
    seed = seed
  )

  if (isTRUE(save)) {
    message("Saving training bank to ", output_dir)
    save_training_data(bank, name = name, output_dir = output_dir)
  }

  bank
}

#' Generate the built-in SEIR training bank
#'
#' @param n_sims Number of simulated parameter sets.
#' @param ndays Number of returned incidence days.
#' @param seed Master random seed.
#' @param priors Optional prior overrides passed to `seir_simulator()`.
#' @param save If TRUE, save the bank as CSV files in `output_dir`.
#' @param name File/model label.
#' @param output_dir Directory the CSV files are written to when `save = TRUE`.
#' @export
generate_seir_training_data <- function(
    n_sims = 20000,
    ndays = 365,
    seed = 122,
    priors = NULL,
    save = FALSE,
    name = "seir",
    output_dir = tempdir()
) {
  bank <- simulate_training_data(
    simulator = seir_simulator(ndays = ndays, priors = priors),
    n_sims = n_sims,
    seed = seed
  )

  if (isTRUE(save)) {
    message("Saving training bank to ", output_dir)
    save_training_data(bank, name = name, output_dir = output_dir)
  }

  bank
}
