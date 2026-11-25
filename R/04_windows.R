# =============================================================================
# 04_windows.R
# Calibrating a window of daily new infections with the SEIR or SIR model
#
# This file contains:
#   - window_calibrator(): the trained any-length BiLSTM (shipped with the
#     package, or your own), with an R0 range and the state at the window start
#   - window_baselines(): ABC-MCMC, ABC-SMC, differential evolution and
#     Nelder-Mead, set up with the same unknowns and ranges as the BiLSTM
#   - predict_cases(): bands of daily new infections from a fit, by running the
#     epiworldR model or by the renewal equation
#   - coverage(), interval_score(), score_band(): how well a band matches data
#   - compare_methods(): all of the above on a list of windows
#   - simulate_window_data(), train_window_calibrator(),
#     tune_window_calibrator(): make your own any-length BiLSTM
#
# A window is a run of daily new infectious people (E -> I for SEIR, S -> I for
# SIR) of any length (15 to 365 days for the shipped models), taken from any
# part of an epidemic. Known values: population n, recovery rate recov and, for
# SEIR, the incubation period incub (days).
# =============================================================================

.STATE_OFFSET <- 1e-4

# Model description --------------------------------------------------------------

.window_info <- function(model) {
  model <- match.arg(toupper(model), c("SEIR", "SIR"))
  seir <- model == "SEIR"
  list(model = model, seir = seir,
       known = if (seir) c("n", "recov", "incub") else c("n", "recov"),
       states = if (seir) c("exposed_share", "infected_share", "recovered_share") else c("infected_share", "recovered_share"),
       crate_range = if (seir) c(5, 15) else c(1, 5))
}

# The ranges the shipped models were trained on.
.window_priors <- function(model) {
  if (.window_info(model)$seir) {
    list(n = c(5000, 10000), recov = c(0.071, 0.25), crate = c(5, 15), incub = c(1, 21), R0 = c(0.8, 5),
         preval = c(0.001, 0.4))
  } else {
    list(n = c(5000, 10000), recov = c(0.071, 0.25), crate = c(1, 5), R0 = c(0.8, 5), preval = c(0.001, 0.4))
  }
}

.cap_shares <- function(sh) sh * min(1, 0.95 / max(sum(sh), 1e-12))

# The known values of the model, and the contact rate when it is known from other data (a contact survey): then
# ptran = R0 x recov / crate is reported instead of being split arbitrarily.
.check_known <- function(info, dots) {
  missing <- setdiff(info$known, names(dots))
  if (length(missing)) stop("Missing known value(s): ", paste(missing, collapse = ", "))
  out <- lapply(dots[info$known], as.numeric)
  if (!is.null(dots$crate)) {
    cr <- as.numeric(dots$crate)
    if (length(cr) != 1 || !is.finite(cr) || cr <= 0) stop("crate must be a single positive number (contacts per day).")
    out$crate <- cr
  }
  out
}

# Days before the first run of k days without new infections (the whole curve
# if there is none).
.informative_days <- function(curve, k = 7) {
  r <- rle(as.numeric(curve) == 0)
  hit <- which(r$values & r$lengths >= k)[1]
  if (is.na(hit)) length(curve) else sum(r$lengths[seq_len(hit)]) - r$lengths[hit]
}

# Run the epiworldR model for `days` days from a starting state: the daily new
# infectious people (E -> I for SEIR, S -> I for SIR, from the transition counts)
# and the number infectious at the start of each day. `p` holds n, recov, (incub,)
# ptran, crate and the shares exposed (SEIR), infected and recovered.
.run_window_model <- function(info, p, days, seed) {
  if (info$seir) {
    E <- p$exposed_share; I <- p$infected_share; R <- p$recovered_share
    carriers <- E + I
    m <- epiworldR::ModelSEIRCONN(name = "sim", n = p$n, prevalence = carriers, contact_rate = p$crate,
                                  incubation_days = p$incub, transmission_rate = p$ptran, recovery_rate = p$recov)
    if (I > 0 || R > 0) epiworldR::initial_states(m, c(if (carriers > 0) I / carriers else 0, R / (1 - carriers)))
  } else {
    I <- p$infected_share; R <- p$recovered_share
    m <- epiworldR::ModelSIRCONN(name = "sim", n = p$n, prevalence = I, contact_rate = p$crate,
                                 transmission_rate = p$ptran, recovery_rate = p$recov)
    if (R > 0) epiworldR::initial_states(m, R / (1 - I))
  }
  epiworldR::verbose_off(m)
  epiworldR::run(m, ndays = days + 1L, seed = seed)
  tr <- epiworldR::get_hist_transition_matrix(m)
  tr <- tr[tr$state_from == (if (info$seir) "Exposed" else "Susceptible") & tr$state_to == "Infected", ]
  new <- tr$counts[match(seq_len(days), tr$date)]
  new[is.na(new)] <- 0
  h <- epiworldR::get_hist_total(m)
  inf <- h[h$state == "Infected", ]
  list(new = as.numeric(new), infected = as.numeric(inf$counts[match(seq_len(days) - 1, inf$date)]))
}

# A fit of one window -----------------------------------------------------------

.window_fit <- function(method, info, observed, known, R0, ptran, crate, state, draws, R0_range = c(NA, NA),
                        ptran_range = c(NA, NA), crate_range = c(NA, NA), fixed = character(0),
                        seconds = NA, model_runs = NA) {
  structure(
    list(
      estimate = c(R0 = R0, ptran = ptran, crate = crate, unlist(state[info$states])),
      method = method, model = info$model, observed = as.numeric(observed), known = known,
      intervals = matrix(c(R0_range, ptran_range, crate_range), nrow = 3, byrow = TRUE,
                         dimnames = list(c("R0", "ptran", "crate"), c("lower_95", "upper_95"))),
      fixed = fixed, draws = draws,
      diagnostics = list(seconds = seconds, model_runs = model_runs)
    ),
    class = c("window_fit", "calibration_result")
  )
}

#' @export
print.window_fit <- function(x, ...) {
  cat(x$method, "fit of a", length(x$observed), "day", x$model, "window\n")
  given <- !is.null(x$known$crate)
  for (p in c("R0", "ptran", "crate")) {
    r <- x$intervals[p, ]
    cat(sprintf("%-6s %.4g%s%s\n", p, x$estimate[[p]],
                if (all(is.finite(r))) sprintf("  (95%% range %.4g to %.4g)", r[1], r[2]) else "",
                if (p == "crate" && given) "  (given)" else if (p %in% x$fixed) "  (fixed, not estimated)" else ""))
  }
  cat(if (given) "ptran = R0 x recov / crate, with the contact rate given.\n" else
    "Only R0 = ptran x crate / recov can be learned from a curve; ptran and crate are not separately identifiable.\n")
  cat("State on the first day of the window:\n")
  print(signif(x$estimate[-(1:3)], 4))
  cat(sprintf("Reff on the first day of the window: %.3g (R0 x the share susceptible)\n", .reff(x)))
  invisible(x)
}


# The any-length BiLSTM ---------------------------------------------------------

#' The any-length BiLSTM calibrator for windows of daily new infections
#'
#' Estimates R0, ptran, crate and the state on the first day of the window (the
#' shares exposed, infectious and recovered) from a window of daily new
#' infectious people of any length (15 to 365 days for the shipped models) from
#' any part of an epidemic. The package ships one SEIR and one SIR model, each
#' the average of 5 networks trained on 1,000,000 noise-free simulated
#' epidemics of `ModelSEIRCONN` / `ModelSIRCONN`. Only R0 = ptran x crate /
#' recov can be learned from a curve: ptran and crate are reported, but not
#' separately reliable. If the contact rate is known from other data (a contact
#' survey such as POLYMOD), give it with the known values, for example
#' `calibrate(cases, cal, n = 10000, recov = 1/7, incub = 5, crate = 8)`: the
#' fit then reports ptran = R0 x recov / crate, with the 95% range that follows
#' from that of R0, and its draws use that contact rate.
#'
#' The 95% R0 range comes from 100,000 simulated calibration windows: the
#' errors of the `k` windows whose estimates, known values and length are
#' closest to the new window. The same windows give `n_draws` parameter sets
#' (R0 and starting state) used by [predict_cases()].
#'
#' Use with [calibrate()]: `calibrate(cases, window_calibrator("SEIR"), n =
#' 5000, recov = 1/7, incub = 5)`.
#'
#' @param model `"SEIR"` or `"SIR"`.
#' @param model_dir Directory of a trained model (made by
#'   [train_window_calibrator()]). `NULL` uses the model shipped with the
#'   package.
#' @param device `"auto"`, `"cpu"` or `"cuda"`.
#' @param k Number of similar calibration windows for the R0 range.
#' @param n_draws Number of parameter draws kept for [predict_cases()].
#' @return A `calibrator`; [calibrate()] returns a `window_fit`.
#' @export
window_calibrator <- function(model = c("SEIR", "SIR"), model_dir = NULL, device = "auto", k = 500L, n_draws = 500L) {
  info <- .window_info(model[1])
  if (is.null(model_dir)) model_dir <- package_file("models", paste0(tolower(info$model), "_window"))
  model_dir <- normalizePath(model_dir, mustWork = TRUE)
  dirs <- list.dirs(model_dir, recursive = FALSE)
  dirs <- dirs[file.exists(file.path(dirs, "model.pt"))]
  if (!length(dirs)) stop("No trained networks (model.pt) in ", model_dir)
  calib <- readRDS(file.path(model_dir, "calibration.rds"))
  lg <- function(x) log(x + .STATE_OFFSET)

  pre_process <- function(daily_cases, ...) {
    list(observed = as.numeric(daily_cases), known = .check_known(info, list(...)))
  }
  run <- function(x, ...) {
    t0 <- proc.time()[["elapsed"]]
    e <- .bilstm_ensemble(dirs, list(x$observed), as.data.frame(x$known), info, device)
    f <- (.calibration_features(e, as.data.frame(x$known), length(x$observed), info) - calib$center) / calib$scale
    d <- colSums((t(calib$features) - as.numeric(f))^2)
    j <- which(d <= sort(d, partial = k)[k])[seq_len(k)]
    res <- calib$residuals[j, , drop = FALSE]
    kn <- x$known[info$known]; crate_k <- x$known$crate
    draws <- lapply(sample(nrow(res), n_draws, replace = TRUE), function(r) {
      R0 <- max(0.05, e$R0 + res$R0[r])
      sh <- .cap_shares(vapply(info$states, function(s) max(exp(lg(e[[s]]) + res[[s]][r]) - .STATE_OFFSET, 0), 0))
      if (is.null(crate_k)) c(kn, list(ptran = e$ptran, crate = R0 * kn$recov / e$ptran), as.list(sh)) else
        c(kn, list(ptran = R0 * kn$recov / crate_k, crate = crate_k), as.list(sh))
    })
    R0_range <- e$R0 + stats::quantile(res$R0, c(0.025, 0.975), names = FALSE)
    if (is.null(crate_k)) {
      .window_fit("DeepIMC", info, x$observed, x$known, R0 = e$R0, ptran = e$ptran, crate = e$R0 * kn$recov / e$ptran,
                  state = as.list(e[info$states]), draws = draws, R0_range = R0_range,
                  seconds = proc.time()[["elapsed"]] - t0, model_runs = 0L)
    } else {
      # The contact rate is given: ptran follows from R0, with the range of R0.
      .window_fit("DeepIMC", info, x$observed, x$known, R0 = e$R0, ptran = e$R0 * kn$recov / crate_k, crate = crate_k,
                  state = as.list(e[info$states]), draws = draws, R0_range = R0_range,
                  ptran_range = R0_range * kn$recov / crate_k, fixed = "crate",
                  seconds = proc.time()[["elapsed"]] - t0, model_runs = 0L)
    }
  }
  list(pre_process = pre_process, run = run, post_process = function(x, ...) x, type = info$model,
       uses_python = TRUE) |>
    structure(class = "calibrator")
}

# Average of the networks' estimates for a list of windows (shares back on the
# original scale).
.bilstm_ensemble <- function(dirs, curves, known, info, device = "auto") {
  known <- as.data.frame(known)[, info$known, drop = FALSE]
  ests <- lapply(dirs, function(d) bilstm_predict_many(d, unname(curves), known, device = device))
  avg <- Reduce(`+`, ests) / length(ests)
  for (s in info$states) avg[[s]] <- pmax(avg[[s]] - .STATE_OFFSET, 0)
  avg
}

.calibration_features <- function(e, known, lengths, info) {
  lg <- function(x) log(x + .STATE_OFFSET)
  st <- matrix(sapply(info$states, function(s) lg(e[[s]])), nrow = nrow(e))
  out <- cbind(e$R0, st, known$recov, log(lengths))
  if (info$seir) out <- cbind(out, known$incub)
  out
}

# Features and errors of the calibration windows, for the R0 range: windows of
# simulated epidemics with known truth (`theta`, `cases` (a list of windows),
# `state` (the shares at the window start), `length`).
.build_calibration <- function(dirs, windows, info, device = "auto") {
  lg <- function(x) log(x + .STATE_OFFSET)
  e <- .bilstm_ensemble(dirs, windows$cases, windows$theta, info, device)
  Fc <- .calibration_features(e, windows$theta, windows$length, info)
  center <- colMeans(Fc); scale <- apply(Fc, 2, stats::sd)
  list(features = scale(Fc, center, scale), center = center, scale = scale,
       residuals = data.frame(R0 = windows$theta$R0 - e$R0,
                              sapply(info$states, function(s) lg(windows$state[[s]]) - lg(e[[s]]), simplify = FALSE)),
       model = info$model)
}


# The standard baselines ----------------------------------------------------------

# Unknowns of the baselines: R0 (unknowns = "R0", the contact rate fixed at the
# middle of its range) or log10 ptran and crate (unknowns = "ptran_crate"), and
# the log10 shares at the window start.
.ptran_range <- function(info) {
  pr <- .window_priors(info$model)
  c(pr$R0[1] * pr$recov[1] / pr$crate[2], min(1, pr$R0[2] * pr$recov[2] / pr$crate[1]))
}
.baseline_space <- function(info, unknowns = "R0") {
  sh <- paste0("log10_", info$states)
  sh_lower <- rep(log10(.STATE_OFFSET), length(sh))
  sh_upper <- ifelse(info$states == "recovered_share", log10(0.95), log10(0.4))
  if (unknowns == "R0") return(list(names = c("R0", sh), lower = c(0.8, sh_lower), upper = c(5, sh_upper)))
  list(names = c("log10_ptran", "crate", sh), lower = c(log10(.ptran_range(info)[1]), info$crate_range[1], sh_lower),
       upper = c(log10(.ptran_range(info)[2]), info$crate_range[2], sh_upper))
}
.baseline_pars <- function(info, theta, known, unknowns = "R0") {
  if (unknowns == "R0") {
    crate <- if (is.null(known$crate)) mean(info$crate_range) else known$crate
    ptran <- theta[[1]] * known$recov / crate; k <- 1
  } else {
    ptran <- 10^theta[[1]]; crate <- theta[[2]]; k <- 2
  }
  sh <- .cap_shares(10^theta[-seq_len(k)]); names(sh) <- info$states
  c(as.list(known[info$known]), list(ptran = ptran, crate = crate), as.list(sh))
}
.baseline_fit <- function(name, info, observed, known, theta, seconds, runs, draws = NULL, unknowns = "R0") {
  sp <- .baseline_space(info, unknowns)
  names(theta) <- sp$names
  p <- .baseline_pars(info, theta, known, unknowns)
  dp <- if (is.null(draws)) list(p) else lapply(seq_len(nrow(draws)), function(i) .baseline_pars(info, draws[i, ], known, unknowns))
  range_of <- function(f) if (is.null(draws)) c(NA, NA) else stats::quantile(vapply(dp, f, 0), c(0.025, 0.975), names = FALSE)
  free <- unknowns == "ptran_crate"
  .window_fit(name, info, observed, known, R0 = p$ptran * p$crate / known$recov, ptran = p$ptran, crate = p$crate,
              state = p[info$states], draws = dp,
              R0_range = range_of(function(q) q$ptran * q$crate / known$recov),
              ptran_range = if (free) range_of(function(q) q$ptran) else c(NA, NA),
              crate_range = if (free) range_of(function(q) q$crate) else c(NA, NA),
              fixed = if (free) character(0) else "crate", seconds = seconds, model_runs = runs)
}
.sse <- function(y, info, theta, known, seed, unknowns = "R0") {
  sum((y - .run_window_model(info, .baseline_pars(info, theta, known, unknowns), length(y), seed)$new)^2)
}

.fit_de <- function(y, info, known, settings, unknowns = "R0") {
  if (!requireNamespace("DEoptim", quietly = TRUE)) stop("DE needs the package 'DEoptim': install.packages('DEoptim')")
  sp <- .baseline_space(info, unknowns); runs <- 0L
  fn <- function(theta) { runs <<- runs + 1L; .sse(y, info, theta, known, 1L, unknowns) }
  t0 <- proc.time()[["elapsed"]]
  fit <- DEoptim::DEoptim(fn, sp$lower, sp$upper,
                          control = do.call(DEoptim::DEoptim.control, utils::modifyList(list(trace = FALSE), settings)))
  .baseline_fit("DE", info, y, known, as.numeric(fit$optim$bestmem), proc.time()[["elapsed"]] - t0, runs,
                unknowns = unknowns)
}

.fit_nm <- function(y, info, known, settings, unknowns = "R0") {
  sp <- .baseline_space(info, unknowns); runs <- 0L
  fn <- function(theta) {
    if (any(theta < sp$lower | theta > sp$upper)) return(Inf)
    runs <<- runs + 1L; .sse(y, info, theta, known, 1L, unknowns)
  }
  t0 <- proc.time()[["elapsed"]]
  fit <- stats::optim((sp$lower + sp$upper) / 2, fn, method = "Nelder-Mead", control = settings)
  .baseline_fit("NelderMead", info, y, known, fit$par, proc.time()[["elapsed"]] - t0, runs, unknowns = unknowns)
}

.fit_abc <- function(method, y, info, known, settings, unknowns = "R0", name = method) {
  sp <- .baseline_space(info, unknowns)
  sim <- new_simulator(
    sample_parameters = function(n_sims) stop("Not used."),
    run = function(parameters, seed) {
      kn <- parameters[info$known]; if (!is.null(known$crate)) kn$crate <- known$crate
      .run_window_model(info, .baseline_pars(info, unlist(parameters[sp$names]), kn, unknowns), length(y), seed)$new
    }, type = info$model)
  bounds <- stats::setNames(lapply(seq_along(sp$names), function(j) c(sp$lower[j], sp$upper[j])), sp$names)
  cal <- do.call(make_calibrator, c(list(method, sim, bounds, info$known), settings))
  t0 <- proc.time()[["elapsed"]]
  res <- do.call(calibrate, c(list(y, cal), known))
  post <- res$posterior
  draws <- if (is.matrix(post)) post else
    post$particles[sample(nrow(post$particles), 500, replace = TRUE, prob = post$weights), , drop = FALSE]
  .baseline_fit(name, info, y, known, as.numeric(calibration_estimate(res)[sp$names]), proc.time()[["elapsed"]] - t0,
                res$diagnostics$n_model_runs, as.matrix(draws[, sp$names, drop = FALSE]), unknowns)
}

#' Standard calibration methods for windows of daily new infections
#'
#' ABC-MCMC, ABC-SMC, differential evolution (DE) and Nelder-Mead, set up the
#' way someone with basic knowledge of these methods would use them, and with
#' the same unknowns and ranges as [window_calibrator()]: R0 between 0.8 and 5
#' and the state on the first day of the window (the shares exposed (SEIR),
#' infectious and recovered, searched on the log10 scale between 0.0001 and 0.4,
#' 0.95 for recovered). By default (`unknowns = "R0"`) the contact rate is fixed
#' at the middle of its training range and ptran = R0 x recov / crate, because
#' only R0 = ptran x crate / recov shapes a curve. With `unknowns =
#' "ptran_crate"` the methods search ptran (log10 scale, within the range of the
#' training data: 0.0038 to 0.25 for SEIR, 0.011 to 1 for SIR) and crate (5 to 15
#' for SEIR, 1 to 5 for SIR) instead, and R0 follows; the ABC methods then give
#' 95% ranges for ptran and crate, which show that only their product is
#' learned. A contact rate known from other data (a contact survey) can be
#' given with the known values (`crate = `); it replaces the middle of the
#' range, and ptran = R0 x recov / crate.
#'
#' Default settings (change them with `settings`):
#' \describe{
#'   \item{ABC-MCMC}{[make_calibrator()] defaults: 3,000 random-walk steps (1,500
#'     burn-in), proposal sd 8% of each range, Gaussian kernel with tolerance
#'     0.30 x the root mean square of the counts, root mean squared error.}
#'   \item{ABC-SMC}{[make_calibrator()] defaults (200 particles, 6 generations,
#'     each tolerance the median distance of the previous generation), with its
#'     limits raised to 30,000 model runs and 20,000 attempts per generation so
#'     that it finishes.}
#'   \item{DE}{`DEoptim::DEoptim()` with its defaults: population 10 x the number
#'     of unknowns, 200 generations, F = 0.8, CR = 0.5; it always runs all 200
#'     generations. Fit: sum of squared differences between the counts and one
#'     model run with a fixed seed.}
#'   \item{NelderMead}{`stats::optim(method = "Nelder-Mead")` with its defaults,
#'     started at the middle of the ranges (values outside are rejected); it
#'     stops when its trial points give fits within 1.5e-8 of each other, or
#'     after 500 runs. Same fit as DE.}
#'   \item{ABC-MCMC-summary, ABC-SMC-summary}{The same ABC methods, comparing
#'     summary statistics of the curves ([epidemic_summary()]: mean, peak,
#'     minimum, peak day, first and last week, the days by which 25%, 50% and
#'     75% of the cases happened, growth at the start and the end) instead of
#'     the whole curve, each statistic scaled by its spread over simulations
#'     from the prior. ABC-MCMC-summary first runs 200 prior simulations for
#'     the scales and takes as its tolerance the 5% quantile of their
#'     distances; ABC-SMC-summary takes the scales from its first generation.
#'     Not in the default `methods`.}
#' }
#'
#' @param model `"SEIR"` or `"SIR"`.
#' @param methods Which methods: any of `"ABC-MCMC"`, `"ABC-SMC"`, `"DE"`,
#'   `"NelderMead"`, `"ABC-MCMC-summary"` and `"ABC-SMC-summary"`.
#' @param settings Optional named list of settings per method, for example
#'   `list(DE = list(itermax = 50), NelderMead = list(maxit = 1000))`: passed to
#'   `DEoptim::DEoptim.control()`, `stats::optim(control = )`, or
#'   [make_calibrator()] (ABC).
#' @param unknowns `"R0"` (search R0, contact rate fixed) or `"ptran_crate"`
#'   (search ptran and crate).
#' @return A named list of `calibrator` objects; [calibrate()] returns a
#'   `window_fit`.
#' @examples
#' \dontrun{
#' baselines <- window_baselines("SEIR")
#' fit <- calibrate(cases, baselines$`ABC-SMC`, n = 10000, recov = 1 / 7, incub = 5)
#' # ptran and crate searched directly:
#' smc <- window_baselines("SEIR", methods = "ABC-SMC", unknowns = "ptran_crate")
#' fit2 <- calibrate(cases, smc$`ABC-SMC`, n = 10000, recov = 1 / 7, incub = 5)
#' fit2$intervals
#' }
#' @export
window_baselines <- function(model = c("SEIR", "SIR"), methods = c("ABC-MCMC", "ABC-SMC", "DE", "NelderMead"),
                             settings = list(), unknowns = c("R0", "ptran_crate")) {
  info <- .window_info(model[1])
  methods <- match.arg(methods, c("ABC-MCMC", "ABC-SMC", "DE", "NelderMead", "ABC-MCMC-summary", "ABC-SMC-summary"),
                       several.ok = TRUE)
  unknowns <- match.arg(unknowns)
  defaults <- list(`ABC-MCMC` = list(), `ABC-SMC` = list(sim_budget = 30000L, max_attempt_mult = 100L),
                   DE = list(), NelderMead = list(), `ABC-MCMC-summary` = list(distance = "summary"),
                   `ABC-SMC-summary` = list(sim_budget = 30000L, max_attempt_mult = 100L, distance = "summary"))
  lapply(stats::setNames(methods, methods), function(m) {
    s <- utils::modifyList(defaults[[m]], if (is.null(settings[[m]])) list() else settings[[m]])
    list(
      pre_process = function(daily_cases, ...) list(observed = as.numeric(daily_cases), known = .check_known(info, list(...))),
      run = function(x, ...) {
        if (unknowns == "ptran_crate" && !is.null(x$known$crate)) stop("A known crate needs unknowns = \"R0\".")
        switch(m, DE = .fit_de(x$observed, info, x$known, s, unknowns),
                                    NelderMead = .fit_nm(x$observed, info, x$known, s, unknowns),
                                    .fit_abc(sub("-summary$", "", m), x$observed, info, x$known, s, unknowns, name = m)) },
      post_process = function(x, ...) x,
      type = info$model
    ) |> structure(class = "calibrator")
  })
}


# The renewal equation ------------------------------------------------------------

# Expected daily new infectious people from a starting state, by the renewal
# equation C_t = R0 (S_{t-1}/n) sum_k g_k C_{t-k}, with g the model's generation
# interval. The model's waiting times are geometric, so the sum is carried day by
# day with the numbers exposed, infectious and susceptible (the same numbers).
.renewal_curve <- function(info, p, days) {
  n <- p$n; beta <- p$ptran * p$crate
  E <- if (info$seir) p$exposed_share * n else 0
  I <- p$infected_share * n
  S <- n - E - I - p$recovered_share * n
  new <- infected <- numeric(days)
  for (t in seq_len(days)) {
    infected[t] <- I
    new_inf <- S * (1 - exp(-beta * I / n))
    onset <- if (info$seir) E / p$incub else new_inf
    if (info$seir) E <- E + new_inf - onset
    I <- I + onset - p$recov * I
    S <- S - new_inf
    new[t] <- onset
  }
  list(new = new, infected = infected)
}

# With R0 fixed, the shares at the window start that bring the renewal curve
# closest to the observed counts (least squares, three starts).
.renewal_initial_state <- function(info, y, R0, known) {
  sp <- .baseline_space(info)
  lo <- sp$lower[-1]; hi <- sp$upper[-1]
  sse <- function(z) sum((y - .renewal_curve(info, .baseline_pars(info, c(R0, z), known), length(y))$new)^2)
  starts <- list((lo + hi) / 2, lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo))
  fits <- lapply(starts, function(s0) stats::optim(s0, sse, method = "L-BFGS-B", lower = lo, upper = hi))
  .baseline_pars(info, c(R0, fits[[which.min(vapply(fits, `[[`, 0, "value"))]]$par), known)
}


# Predictions and scores ----------------------------------------------------------

#' Predict daily cases from a window fit
#'
#' Turns a [window_fit] (from [calibrate()] with [window_calibrator()] or
#' [window_baselines()]) into 95% bands of the daily new infectious people and
#' of the number currently infectious, for the window and `ahead` days after it.
#'
#' * `method = "simulation"`: `n` runs of `ModelSEIRCONN` / `ModelSIRCONN` from
#'   the estimated starting state, with parameters drawn from the method's
#'   uncertainty (the BiLSTM's R0 range, the ABC posterior; one parameter set
#'   for DE and Nelder-Mead).
#' * `method = "renewal"`: the renewal equation C_t = R0 (S_{t-1}/n) sum_k g_k
#'   C_{t-k} (g: the model's generation interval), one curve per draw, with a
#'   Poisson count around it on each day. It gives the model's average curve and
#'   is about 200 times faster.
#'
#' `initial_state = "renewal"` works out the starting state (the initial
#' cases) by the renewal equation instead of using the fitted one: with R0 fixed
#' at 30 values from the fit's R0 range, the shares that bring the renewal curve
#' closest to the observed window.
#'
#' @param fit A `window_fit`.
#' @param ahead Days to predict after the window.
#' @param method `"simulation"` or `"renewal"`.
#' @param n Number of runs or draws.
#' @param initial_state `"fitted"` or `"renewal"`.
#' @param seed Random seed.
#' @return A data frame with `day`, whether the day is in the window, the
#'   observed count (in the window), and the median, lower and upper 95% limits
#'   of the daily new infectious (`new_*`) and the number infectious
#'   (`infected_*`).
#' @export
predict_cases <- function(fit, ahead = 0L, method = c("simulation", "renewal"), n = 500L,
                          initial_state = c("fitted", "renewal"), seed = 1L) {
  if (!inherits(fit, "window_fit")) stop("fit must be a window_fit (from calibrate() with a window calibrator).")
  method <- match.arg(method); initial_state <- match.arg(initial_state)
  info <- .window_info(fit$model)
  L <- length(fit$observed); days <- L + as.integer(ahead)
  set.seed(seed)
  draws <- fit$draws
  if (initial_state == "renewal") {
    R0s <- vapply(draws, function(p) p$ptran * p$crate / p$recov, 0)
    R0s <- if (length(draws) > 1) stats::quantile(R0s, seq(0.02, 0.98, length.out = 30), names = FALSE) else R0s
    draws <- lapply(R0s, function(r) .renewal_initial_state(info, fit$observed, r, fit$known))
  }
  pick <- draws[sample(length(draws), n, replace = TRUE)]
  runs <- if (method == "simulation") {
    lapply(seq_len(n), function(i) .run_window_model(info, pick[[i]], days, seed = i))
  } else {
    lapply(pick, function(p) { r <- .renewal_curve(info, p, days); r$new <- stats::rpois(days, pmax(r$new, 0)); r })
  }
  new <- vapply(runs, `[[`, numeric(days), "new")
  inf <- vapply(runs, `[[`, numeric(days), "infected")
  q <- function(x, p) apply(x, 1, stats::quantile, p, names = FALSE)
  data.frame(day = seq_len(days), in_window = seq_len(days) <= L, observed = c(fit$observed, rep(NA, days - L)),
             new_median = apply(new, 1, stats::median), new_lower = q(new, 0.025), new_upper = q(new, 0.975),
             infected_median = apply(inf, 1, stats::median), infected_lower = q(inf, 0.025),
             infected_upper = q(inf, 0.975))
}

#' Coverage and interval score of a band
#'
#' `coverage()`: the share of days on which the value is inside the band (%).
#' `interval_score()`: the mean over days of the band width plus `2 / alpha`
#' times how far the value falls outside the band (Gneiting and Raftery, 2007):
#' \deqn{IS = \frac{1}{T}\sum_t (u_t - l_t) + \frac{2}{\alpha}(l_t - y_t)1(y_t < l_t) +
#' \frac{2}{\alpha}(y_t - u_t)1(y_t > u_t).}
#' With a 95% band (`alpha = 0.05`) a miss costs 40 times its distance. Lower
#' is better; a band cannot score well just by being wide.
#'
#' @param y Values (observed or true counts).
#' @param lower,upper The band.
#' @param alpha 1 minus the level of the band.
#' @return A number.
#' @export
interval_score <- function(y, lower, upper, alpha = 0.05) {
  mean((upper - lower) + 2 / alpha * pmax(lower - y, 0) + 2 / alpha * pmax(y - upper, 0), na.rm = TRUE)
}

#' @rdname interval_score
#' @export
coverage <- function(y, lower, upper) 100 * mean(y >= lower & y <= upper, na.rm = TRUE)

#' Score a band from predict_cases()
#'
#' @param band A data frame from [predict_cases()].
#' @param y The values to compare with: by default the observed window.
#' @param days Which days of the band (by default the days of `y`).
#' @param what `"new"` (daily new infectious) or `"infected"` (number
#'   infectious).
#' @return Coverage (%), mean band width, error of the median (mean absolute
#'   difference) and interval score.
#' @export
score_band <- function(band, y = band$observed[band$in_window], days = seq_along(y), what = c("new", "infected")) {
  what <- match.arg(what)
  b <- band[days, paste0(what, c("_median", "_lower", "_upper"))]
  names(b) <- c("median", "lower", "upper")
  c(coverage = coverage(y, b$lower, b$upper), width = mean(b$upper - b$lower), MAE = mean(abs(b$median - y)),
    interval_score = interval_score(y, b$lower, b$upper))
}


# Comparison ----------------------------------------------------------------------

#' Compare calibration methods on windows of daily new infections
#'
#' Runs every calibrator on every window, then scores each fit with
#' [predict_cases()]: coverage and interval score of the window and of the days
#' after it, and, when the truth is given, the errors of R0, ptran, crate and
#' the starting state.
#'
#' @param windows A list of windows. Each is a list with `cases` (the daily new
#'   infections of the window), the known values (`n`, `recov`, and `incub` for
#'   SEIR), and optionally `future` (the counts after the window), the truth
#'   (`R0`, `ptran`, `crate`, and `state`: the shares at the window start, named
#'   like the estimates) and labels (single character values such as `wave` or
#'   `position`, copied to the result). [real_data_windows()] makes such a list
#'   from the data shipped with the package.
#' @param model `"SEIR"` or `"SIR"`.
#' @param calibrators Named list of calibrators; by default DeepIMC (the BiLSTM) and the
#'   four standard baselines.
#' @param method,n Passed to [predict_cases()].
#' @param cores Number of (window, method) pairs run in parallel (forked
#'   processes; 1 on Windows). Calibrators that use Python (the BiLSTM) are
#'   fitted in the main
#'   process first, because torch can hang in a forked process; their
#'   predictions are still made in parallel.
#' @return A data frame with one row per window and method: the labels, the
#'   estimates of R0, ptran and crate with their 95% ranges (when the method
#'   gives one), the true values and absolute errors (when given), the
#'   starting-state error (percentage points), coverage and interval score of
#'   the window and of the days after it, seconds and model runs. See
#'   [plot_comparison()] and [plot_estimates()].
#' @export
compare_methods <- function(windows, model = c("SEIR", "SIR"), calibrators = NULL, method = "simulation", n = 500L,
                            cores = 1L) {
  info <- .window_info(model[1])
  if (is.null(calibrators)) calibrators <- c(list(DeepIMC = window_calibrator(info$model)), window_baselines(info$model))
  if (is.null(names(windows))) names(windows) <- sprintf("w%03d", seq_along(windows))
  fit_one <- function(w, m) tryCatch(do.call(calibrate, c(list(w$cases, calibrators[[m]]), w[info$known])),
                                     error = function(e) e)
  python <- names(calibrators)[vapply(calibrators, function(cal) isTRUE(cal$uses_python), logical(1))]
  python_fits <- lapply(windows, function(w) lapply(stats::setNames(python, python), function(m) fit_one(w, m)))
  pars <- c("R0", "ptran", "crate")
  score_one <- function(w, fit) {
    future <- if (is.null(w$future)) numeric(0) else as.numeric(w$future)
    band <- predict_cases(fit, ahead = length(future), method = method, n = n)
    s_win <- score_band(band, as.numeric(w$cases))
    s_fut <- if (length(future)) score_band(band, future, length(w$cases) + seq_along(future)) else
      c(coverage = NA, interval_score = NA)
    est <- fit$estimate
    out <- list()
    for (p in pars) {
      out[[p]] <- est[[p]]
      out[[paste0(p, "_low")]] <- fit$intervals[p, 1]; out[[paste0(p, "_high")]] <- fit$intervals[p, 2]
      out[[paste0(p, "_true")]] <- if (is.null(w[[p]])) NA else w[[p]]
      out[[paste0(p, "_error")]] <- abs(est[[p]] - out[[paste0(p, "_true")]])
    }
    c(out, list(state_error = if (is.null(w$state)) NA else 100 * mean(abs(est[names(w$state)] - unlist(w$state))),
                coverage = s_win[["coverage"]], interval_score = s_win[["interval_score"]],
                coverage_ahead = s_fut[["coverage"]], interval_score_ahead = s_fut[["interval_score"]],
                seconds = fit$diagnostics$seconds, model_runs = fit$diagnostics$model_runs))
  }
  empty <- unlist(lapply(pars, function(p) paste0(p, c("", "_low", "_high", "_true", "_error"))))
  empty <- stats::setNames(as.list(rep(NA_real_, length(empty) + 7)),
                           c(empty, "state_error", "coverage", "interval_score", "coverage_ahead",
                             "interval_score_ahead", "seconds", "model_runs"))
  # One task per window and method, in parallel.
  tasks <- expand.grid(method = names(calibrators), window = names(windows), stringsAsFactors = FALSE)
  one <- function(i) {
    id <- tasks$window[i]; m <- tasks$method[i]
    w <- windows[[id]]
    labels <- w[vapply(w, function(v) is.character(v) && length(v) == 1, logical(1))]
    fit <- if (m %in% python) python_fits[[id]][[m]] else fit_one(w, m)
    res <- if (inherits(fit, "error")) fit else tryCatch(score_one(w, fit), error = function(e) e)
    failed <- inherits(res, "error")
    data.frame(c(list(window = id), labels, list(method = m, failed = failed,
                                                 error = if (failed) conditionMessage(res) else NA_character_,
                                                 days = length(w$cases)),
                 if (failed) empty else res), check.names = FALSE, stringsAsFactors = FALSE)
  }
  rows <- if (cores > 1 && .Platform$OS.type == "unix") {
    parallel::mclapply(seq_len(nrow(tasks)), one, mc.cores = cores, mc.preschedule = FALSE)
  } else {
    lapply(seq_len(nrow(tasks)), one)
  }
  bad <- !vapply(rows, is.data.frame, logical(1))
  if (any(bad)) stop("compare_methods() failed on window ", tasks$window[which(bad)[1]], ", method ",
                     tasks$method[which(bad)[1]], ": ", as.character(rows[[which(bad)[1]]]))
  cols <- unique(unlist(lapply(rows, names)))
  out <- do.call(rbind, lapply(rows, function(r) { for (c in setdiff(cols, names(r))) r[[c]] <- NA; r[cols] }))
  rownames(out) <- NULL
  out
}


# Making your own any-length BiLSTM -----------------------------------------------

# Simulate n epidemics of n_days with the daily new infectious and the shares in
# each state at the start of each day (in memory).
.simulate_window_epidemics <- function(info, n_keep, n_days, priors, min_cases, seed, cores) {
  states <- info$states
  epi <- c(exposed_share = "Exposed", infected_share = "Infected", recovered_share = "Recovered")[states]
  base <- if (info$seir) seir_simulator(ndays = n_days, priors = priors) else
    sir_simulator(ndays = n_days, priors = priors[names(priors) != "incub"], keep_day_zero = FALSE)
  sampler <- if (info$seir) base$sample_parameters else function(n_sims) {
    th <- base$sample_parameters(n_sims)
    th$initial_infected <- pmax(1, round(log_uniform(n_sims, priors$preval) * th$n))
    th$preval <- th$initial_infected / th$n
    th
  }
  run_one <- function(parameters, seed) {
    p <- parameters
    p$exposed_share <- if (info$seir) p$preval else 0
    p$infected_share <- if (info$seir) 0 else p$preval
    p$recovered_share <- 0
    if (info$seir) {
      m <- epiworldR::ModelSEIRCONN(name = "sim", n = p$n, prevalence = p$preval, contact_rate = p$crate,
                                    incubation_days = p$incub, transmission_rate = p$ptran, recovery_rate = p$recov)
      epiworldR::verbose_off(m); epiworldR::run(m, ndays = n_days + 1L, seed = seed)
    } else {
      m <- epiworldR::ModelSIRCONN(name = "sim", n = p$n, prevalence = p$preval, contact_rate = p$crate,
                                   transmission_rate = p$ptran, recovery_rate = p$recov)
      epiworldR::verbose_off(m); epiworldR::run(m, ndays = n_days, seed = seed)
    }
    h <- epiworldR::get_hist_total(m); h <- h[order(h$date), ]
    at_start <- function(st) { x <- h$counts[h$state == st] / p$n; c(x, rep(x[length(x)], n_days))[seq_len(n_days)] }
    c(get_incidence(m, n_days, keep_day_zero = FALSE), as.vector(vapply(epi, at_start, numeric(n_days))))
  }
  n_draw <- ceiling(1.15 * n_keep)
  sizes <- diff(round(seq(0, n_draw, length.out = max(cores, 1) + 1)))
  chunks <- parallel::mclapply(seq_along(sizes), function(k) {
    simulate_training_data(new_simulator(sampler, run_one, type = info$model), n_sims = sizes[k], seed = seed + k - 1)
  }, mc.cores = if (.Platform$OS.type == "unix") max(cores, 1) else 1)
  bad <- vapply(chunks, inherits, logical(1), what = "try-error")
  if (any(bad)) stop("Simulation failed: ", as.character(chunks[[which(bad)[1]]]))
  theta <- do.call(rbind, lapply(chunks, `[[`, "theta"))
  part <- function(j) do.call(rbind, lapply(chunks, function(x) x$observations[, j * n_days + seq_len(n_days), drop = FALSE]))
  cases <- part(0)
  st <- stats::setNames(lapply(seq_along(states), part), states)
  keep <- which(rowSums(cases) >= min_cases)
  if (length(keep) < n_keep) stop("Only ", length(keep), " epidemics with at least ", min_cases, " cases.")
  keep <- keep[seq_len(n_keep)]
  theta <- theta[keep, ]; rownames(theta) <- NULL
  list(theta = theta, cases = cases[keep, , drop = FALSE], states = lapply(st, function(s) s[keep, , drop = FALSE]),
       dropped_share = mean(rowSums(cases) < min_cases))
}

.write_table <- function(x, file) {
  if (requireNamespace("data.table", quietly = TRUE)) data.table::fwrite(as.data.frame(x), file) else
    utils::write.csv(x, file, row.names = FALSE)
}

#' Simulate training data for an any-length BiLSTM
#'
#' Simulates noise-free epidemics of `ModelSEIRCONN` / `ModelSIRCONN` (every new
#' infectious person counted) with parameters drawn from `priors`, and writes
#' the files [train_window_calibrator()] and [tune_window_calibrator()] read:
#' the parameters, the daily new infectious, and the log share in each state at
#' the start of each day. Each epidemic starts with a share `preval` exposed
#' (SEIR) or infectious (SIR), drawn on the log scale. Epidemics with fewer than
#' `min_cases` cases are dropped.
#'
#' The shipped models used 1,000,000 epidemics of 365 days and the default
#' priors (about 2 hours on 32 cores for SEIR).
#'
#' @param model `"SEIR"` or `"SIR"`.
#' @param n_sims Number of epidemics kept.
#' @param output_dir Where to write the files.
#' @param n_days Days simulated.
#' @param priors Named list of ranges (`c(min, max)`): `n`, `recov`, `crate`,
#'   `incub` (SEIR), `R0`, `preval`. Missing entries use the shipped models'
#'   ranges.
#' @param min_cases Minimum number of cases.
#' @param seed Random seed.
#' @param cores Number of cores (forked processes).
#' @return Invisibly, a `window_data` list with the model, the file paths and
#'   the settings.
#' @export
simulate_window_data <- function(model = c("SEIR", "SIR"), n_sims, output_dir, n_days = 365L, priors = NULL,
                                 min_cases = 20L, seed = 122L, cores = 1L) {
  info <- .window_info(model[1])
  priors <- utils::modifyList(.window_priors(info$model), if (is.null(priors)) list() else priors)
  b <- .simulate_window_epidemics(info, n_sims, n_days, priors, min_cases, seed, cores)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir)
  prefix <- tolower(info$model)
  files <- c(theta = file.path(output_dir, paste0(prefix, "_theta.csv")),
             cases = file.path(output_dir, paste0(prefix, "_observations.csv")),
             stats::setNames(file.path(output_dir, paste0(prefix, "_log_", info$states, ".csv")), paste0("log_", info$states)))
  .write_table(b$theta, files[["theta"]])
  .write_table(b$cases, files[["cases"]])
  for (s in info$states) .write_table(round(log(b$states[[s]] + .STATE_OFFSET), 5), files[[paste0("log_", s)]])
  out <- structure(list(model = info$model, files = files, n_sims = n_sims, n_days = n_days, priors = priors,
                        dropped_share = b$dropped_share), class = "window_data")
  saveRDS(out, file.path(output_dir, "window_data.rds"))
  invisible(out)
}

# Settings of the shipped models (from the hyperparameter search).
.tuned_settings <- function(model) {
  if (.window_info(model)$seir) {
    list(learning_rate = 7.16e-4, batch_size = 32L, hidden_size = 128L, num_layers = 3L, dropout = 0.13)
  } else {
    list(learning_rate = 7.94e-4, batch_size = 32L, hidden_size = 64L, num_layers = 2L, dropout = 0.28)
  }
}

#' Train an any-length BiLSTM for windows of daily new infections
#'
#' Trains one network per seed on data from [simulate_window_data()]: inputs are
#' random windows of 15 to 365 days inside the informative part of each
#' epidemic (before the first week without new infections), log(1 + count),
#' and the known values; outputs are ptran, R0 and the log shares in each state
#' at the window start. Then simulates `n_calib` more epidemics, cuts one random
#' window from each, and stores the networks' errors on them for the R0 range of
#' [window_calibrator()]. The result can be used with `window_calibrator(model,
#' model_dir = output_dir)`.
#'
#' The shipped models used 5 seeds, 50 epochs, the tuned settings and a cosine
#' learning-rate schedule; each network took about 4 hours on a recent GPU for
#' SEIR.
#'
#' @param data A `window_data` object from [simulate_window_data()] (or the path
#'   of its folder).
#' @param output_dir Where to save the networks and the calibration windows.
#' @param seeds One network per seed.
#' @param hyperparameters List with `learning_rate`, `batch_size`,
#'   `hidden_size`, `num_layers`, `dropout` (for example the result of
#'   [tune_window_calibrator()]), or the path of its JSON file. `NULL` uses the
#'   shipped models' settings.
#' @param epochs,lr_schedule,device,resume Passed to
#'   [train_bilstm_calibrator()]. With `resume = TRUE` a stopped run continues
#'   from its last epoch.
#' @param window_min,window_max Range of window lengths (by default 15 days to
#'   the simulated length, at most 365).
#' @param n_calib Number of calibration windows for the R0 range.
#' @param cores Cores for simulating the calibration windows.
#' @return Invisibly, `output_dir`.
#' @export
train_window_calibrator <- function(data, output_dir, seeds = c(122L, 1L, 2L, 3L, 4L), hyperparameters = NULL,
                                    epochs = 50L, lr_schedule = "cosine", window_min = 15L, window_max = NULL,
                                    n_calib = 100000L, device = "auto", resume = TRUE, cores = 1L) {
  if (is.character(data)) data <- readRDS(file.path(data, "window_data.rds"))
  info <- .window_info(data$model)
  if (is.null(window_max)) window_max <- min(365L, as.integer(data$n_days))
  hp <- if (is.null(hyperparameters)) .tuned_settings(info$model) else if (is.character(hyperparameters))
    jsonlite::read_json(hyperparameters) else hyperparameters
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir)
  state_targets <- paste0("log_", info$states)
  dirs <- file.path(output_dir, paste0("model_seed_", seeds))
  for (i in seq_along(seeds)) {
    if (file.exists(file.path(dirs[i], "model.pt"))) next
    train_bilstm_calibrator(
      name = tolower(info$model), known_names = info$known, target_names = c("ptran", "R0"), type = info$model,
      parameter_file = data$files[["theta"]], observation_file = data$files[["cases"]], output_dir = dirs[i],
      epochs = epochs, batch_size = as.integer(hp$batch_size), learning_rate = hp$learning_rate, seed = seeds[i],
      hidden_size = as.integer(hp$hidden_size), num_layers = as.integer(hp$num_layers), dropout = hp$dropout,
      physics_weight = 0, incidence_max = 10000, incidence_transform = "log1p",
      output_activation = c("sigmoid", "softplus", rep("sigmoid", length(info$states))),
      window_min = window_min, window_max = window_max, window_stop_zeros = 7L,
      state_files = data$files[state_targets], grad_clip = 1, lr_schedule = lr_schedule, resume = resume,
      device = device
    )
  }
  # Calibration windows for the R0 range.
  b <- .simulate_window_epidemics(info, n_calib, data$n_days, data$priors, 20L, 7L, cores)
  set.seed(7L)
  active <- apply(b$cases, 1, .informative_days)
  use <- which(active >= window_min)
  L <- vapply(use, function(i) { hi <- min(window_max, active[i]); if (hi == window_min) window_min else sample(window_min:hi, 1) },
              numeric(1))
  s <- vapply(seq_along(use), function(j) sample.int(active[use[j]] - L[j] + 1, 1), numeric(1))
  windows <- list(theta = b$theta[use, ], length = L,
                  cases = lapply(seq_along(use), function(j) b$cases[use[j], s[j]:(s[j] + L[j] - 1)]),
                  state = data.frame(lapply(b$states, function(m) m[cbind(use, s)])))
  saveRDS(.build_calibration(dirs, windows, info, device), file.path(output_dir, "calibration.rds"))
  jsonlite::write_json(list(model = info$model, seeds = seeds, epochs = epochs, hyperparameters = hp,
                            lr_schedule = lr_schedule, window_min = window_min, window_max = window_max,
                            n_sims = data$n_sims, n_calib = n_calib, priors = data$priors),
                       file.path(output_dir, "window_model.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(output_dir)
}

#' Tune the settings of an any-length BiLSTM
#'
#' A hyperparameter search with Optuna (Python): each trial trains a network for
#' `epochs` epochs on `subset` of the training epidemics with settings drawn
#' from the search space (learning rate 1e-4 to `lr_max`, batch size 32 to 256,
#' hidden size 64 to 256, 1 to 3 layers, dropout 0 to 0.5) and is scored on the
#' validation error, half R0 and half the starting state; weak trials are
#' stopped early. Several R sessions (for example several GPU jobs) can share one
#' search by using the same `output_dir`.
#'
#' The shipped models' settings come from 40 trials of 20 epochs on 100,000
#' epidemics, with windows of 30 to 365 days; the shipped networks were then
#' trained with these settings on windows of 15 to 365 days.
#'
#' @param data A `window_data` object from [simulate_window_data()] (or the path
#'   of its folder).
#' @param output_dir Where the search is stored (`journal.log`, `trials.csv`,
#'   `best_params.json`).
#' @param n_trials Number of trials.
#' @param epochs Epochs per trial.
#' @param subset Number of epidemics used (the first ones).
#' @param lr_max Largest learning rate tried.
#' @param timeout Optional time limit in seconds.
#' @param device `"auto"`, `"cpu"` or `"cuda"`.
#' @param window_min Shortest window length in days (the longest is the
#'   simulated length, at most 365).
#' @return The best settings (a list), also saved as `best_params.json`.
#' @export
tune_window_calibrator <- function(data, output_dir, n_trials = 40L, epochs = 20L, subset = 100000L, lr_max = 1e-3,
                                   timeout = NULL, device = "auto", window_min = 15L) {
  if (is.character(data)) data <- readRDS(file.path(data, "window_data.rds"))
  info <- .window_info(data$model)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir)
  files <- data$files
  if (subset < data$n_sims) {
    sub_dir <- file.path(output_dir, "subset"); dir.create(sub_dir, showWarnings = FALSE)
    files <- vapply(names(files), function(k) {
      f <- file.path(sub_dir, basename(files[[k]]))
      if (!file.exists(f)) writeLines(readLines(files[[k]], n = subset + 1L), f)
      f
    }, "")
  }
  state_targets <- paste0("log_", info$states)
  python <- reticulate::py_config()$python
  args <- c(
    package_file("python", "tune_bilstm.py"),
    "--theta", files[["theta"]], "--curves", files[["cases"]],
    "--known", paste(info$known, collapse = ","), "--targets", "ptran,R0",
    "--state-files", paste0(state_targets, "=", files[state_targets], collapse = ","),
    "--incidence-transform", "log1p", "--physics-weight", 0,
    "--study-dir", output_dir, "--study-name", paste0(tolower(info$model), "_window"),
    "--score-targets", paste(c(rep("R0", length(info$states)), state_targets), collapse = ","),
    "--epochs", as.integer(epochs), "--n-trials", as.integer(n_trials), "--lr-max", lr_max,
    "--window-min", as.integer(window_min), "--window-max", min(365L, as.integer(data$n_days)), "--window-stop-zeros", 7L, "--grad-clip", 1,
    "--incidence-max", 10000,
    "--output-activation", paste(c("sigmoid", "softplus", rep("sigmoid", length(info$states))), collapse = ","),
    "--device", device,
    if (!is.null(timeout)) c("--timeout", timeout)
  )
  status <- system2(python, args = args)
  if (!identical(status, 0L)) stop("The hyperparameter search failed with status ", status)
  best <- jsonlite::read_json(file.path(output_dir, "best_params.json"))
  best[c("learning_rate", "batch_size", "hidden_size", "num_layers", "dropout")]
}
