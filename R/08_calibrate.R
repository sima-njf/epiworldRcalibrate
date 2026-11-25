# =============================================================================
# 08_calibrate.R
# One function for every calibration method, for the SIR and SEIR models
#
# This file contains:
#   - calibrate_window(): DeepIMC, ABC-SMC, ABC-MCMC, differential evolution
#     or Nelder-Mead on one window of daily counts (one method or several)
#   - print(), summary() and plot() methods for the fits
#   - known_values(): the known values of a disease, as in the paper
# =============================================================================

.METHODS <- c("DeepIMC", "ABC-SMC", "ABC-MCMC", "DE", "NelderMead")

# The method's name in the package, whatever the spelling ("ABC_SMC", "abc-smc", "NELDERMEAD", "Nelder-Mead", ...);
# NA when it is not a method.
.method_key <- function(method) {
  map <- c(DEEPIMC = "DeepIMC", BILSTM = "DeepIMC", ABCSMC = "ABC-SMC", ABCMCMC = "ABC-MCMC", DE = "DE",
           DIFFERENTIALEVOLUTION = "DE", DEOPTIM = "DE", NELDERMEAD = "NelderMead", NM = "NelderMead")
  unname(map[toupper(gsub("[^A-Za-z]", "", as.character(method)))])
}
.method_name <- function(method) {
  if (!is.character(method) || !length(method)) stop("method must be a character vector.", call. = FALSE)
  out <- .method_key(method)
  if (anyNA(out)) stop("Unknown method ", paste0("\"", method[is.na(out)], "\"", collapse = ", "),
                       ". Use \"DeepIMC\", \"ABC_SMC\", \"ABC_MCMC\", \"DE\" or \"NELDERMEAD\".", call. = FALSE)
  out
}

# The DeepIMC calibrators are kept once loaded (reading the networks and the calibration windows takes a few seconds).
.calibrator_cache <- new.env(parent = emptyenv())

# The calibrator of a method. The ABC methods compare summary statistics of the curves (abc_distance = "summary", as
# in the paper) or the whole curves (abc_distance = "curve").
.method_calibrator <- function(method, model, settings = list(), abc_distance = "summary") {
  m <- .method_name(method)[1]; model <- .window_info(model)$model
  if (m == "DeepIMC") {
    key <- paste0("DeepIMC_", model)
    if (is.null(.calibrator_cache[[key]])) .calibrator_cache[[key]] <- window_calibrator(model)
    return(.calibrator_cache[[key]])
  }
  inner <- if (m %in% c("ABC-SMC", "ABC-MCMC") && abc_distance == "summary") paste0(m, "-summary") else m
  window_baselines(model, inner, settings = stats::setNames(list(as.list(settings)), inner))[[1]]
}

# The settings of method m: `settings` named by method (list(DE = list(itermax = 50), ...)), or the settings of the
# only method.
.settings_for <- function(settings, m, several) {
  if (!length(settings)) return(list())
  keys <- if (is.null(names(settings))) NA else .method_key(names(settings))
  if (all(!is.na(keys))) { i <- which(keys == m); return(if (length(i)) as.list(settings[[i[1]]]) else list()) }
  if (several) stop("With several methods, give settings by method, for example list(DE = list(itermax = 50)).",
                    call. = FALSE)
  as.list(settings)
}

#' Calibrate a window of daily counts with DeepIMC or a standard method
#'
#' One function for the five calibration methods of the package, for the SEIR
#' and the SIR model. The window is a run of daily new infections (or cases,
#' or admissions) from any part of an epidemic; the method estimates R0 and the
#' state on the first day of the window (the shares exposed (SEIR), infectious
#' and recovered), with a 95% range for R0 where the method gives one.
#'
#' \describe{
#'   \item{`"DeepIMC"`}{The trained any-length BiLSTM shipped with the package
#'     ([window_calibrator()]): one pass of five networks, no model runs, well
#'     under a second. Windows of 15 to 365 days. The 95% R0 range comes from
#'     the errors of the 500 most similar of 100,000 simulated calibration
#'     windows.}
#'   \item{`"ABC_SMC"`}{ABC sequential Monte Carlo (Toni et al. 2009; Beaumont
#'     et al. 2009): 200 particles, 6 generations, each tolerance the median
#'     distance of the previous generation, at most 30,000 model runs.}
#'   \item{`"ABC_MCMC"`}{ABC Markov chain Monte Carlo (Marjoram et al. 2003):
#'     3,000 random-walk steps (the last 1,500 kept), proposal s.d. 8% of each
#'     range, Gaussian kernel with its tolerance the 5% quantile of the
#'     distances of 200 simulations from the prior.}
#'   \item{`"DE"`}{Differential evolution, `DEoptim::DEoptim()` with its
#'     defaults (population 10 x the number of unknowns, 200 generations,
#'     F = 0.8, CR = 0.5), minimizing the sum of squared differences between
#'     the counts and one model run with a fixed seed.}
#'   \item{`"NELDERMEAD"`}{Nelder-Mead, `stats::optim()` with its defaults,
#'     started at the middle of the ranges; same fit as DE.}
#' }
#' The spelling of `method` does not matter: `"ABC_SMC"`, `"ABC-SMC"` and
#' `"abc smc"` are the same method, as are `"NELDERMEAD"`, `"NelderMead"` and
#' `"Nelder-Mead"`.
#'
#' All methods search R0 between 0.8 and 5 and the starting shares on the
#' log10 scale between 0.0001 and 0.4 (0.95 for the recovered share). A curve
#' alone shows only R0 = ptran x crate / recov, so the standard methods fix the
#' contact rate at the middle of its training range (10 for SEIR, 3 for SIR)
#' unless it is given (`crate =`, for example from a contact survey); DeepIMC
#' reports its own split of R0 into ptran and crate, which is not reliable on
#' its own.
#'
#' @param cases Daily counts of the window.
#' @param model `"SEIR"` or `"SIR"`.
#' @param method One method or several: `"DeepIMC"` (default), `"ABC_SMC"`,
#'   `"ABC_MCMC"`, `"DE"`, `"NELDERMEAD"`.
#' @param ... The known values: `n` (the population, or the effective
#'   population the counts come from), `recov` (1 / the infectious period in
#'   days) and, for SEIR, `incub` (the latent period in days); optionally
#'   `crate`, a contact rate known from other data.
#' @param known The known values as a list instead (for example from
#'   [known_values()]); values in `...` replace those in `known`.
#' @param settings Settings of the method, passed to `DEoptim::DEoptim.control()`
#'   (DE), `stats::optim(control = )` (Nelder-Mead) or [make_calibrator()]
#'   (ABC, for example `list(n_particles = 500)`); with several methods, a list
#'   by method: `list(DE = list(itermax = 50), ABC_SMC = list(n_particles = 100))`.
#' @param abc_distance How the ABC methods compare the counts with a model run:
#'   `"summary"` (eleven summary statistics of the curve, [epidemic_summary()],
#'   each scaled by its spread over simulations from the prior; the default,
#'   as in the paper) or `"curve"` (the root mean squared difference of the
#'   whole curves).
#' @param seed Random seed, set before each method.
#' @return For one method, a `window_fit`: `estimate` (R0, ptran, crate and the
#'   starting shares), `intervals` (95% ranges), `draws` (parameter sets for
#'   [predict_cases()]), the observed counts, the known values and
#'   `diagnostics` (seconds and model runs). For several methods, a
#'   `window_fits` list of them; `summary()` gives a table and `plot()` the 95%
#'   bands of each method.
#' @seealso [calibrate_curve()] to split, smooth and calibrate a whole curve;
#'   [predict_cases()] for the bands; [known_values()].
#' @examples
#' \dontrun{
#' data(flu_baltimore_1918)
#' y <- flu_baltimore_1918$cases[1:40]
#' fit <- calibrate_window(y, model = "SEIR", method = "DeepIMC", n = 10000, recov = 0.25, incub = 1.4)
#' fit
#' plot(fit)
#'
#' # The five methods on the same window (about 20 minutes on one core)
#' fits <- calibrate_window(y, model = "SEIR", method = c("DeepIMC", "ABC_SMC", "ABC_MCMC", "DE", "NELDERMEAD"),
#'                          known = known_values("influenza", "SEIR"))
#' summary(fits)
#' plot(fits)
#' }
#' @export
calibrate_window <- function(cases, model = c("SEIR", "SIR"), method = "DeepIMC", ..., known = NULL, settings = list(),
                             abc_distance = c("summary", "curve"), seed = 1L) {
  model <- .window_info(model[1])$model
  abc_distance <- match.arg(abc_distance)
  kn <- utils::modifyList(if (is.null(known)) list() else as.list(known), list(...))
  y <- as.numeric(cases)
  if (!length(y) || anyNA(y) || any(y < 0)) stop("cases must be non-negative counts without NA.", call. = FALSE)
  ms <- unique(.method_name(method))
  fits <- lapply(stats::setNames(ms, ms), function(m) {
    if (m == "DeepIMC" && (length(y) < 15 || length(y) > 365))
      stop("DeepIMC needs a window of 15 to 365 days; this one has ", length(y), ".", call. = FALSE)
    cal <- .method_calibrator(m, model, .settings_for(settings, m, length(ms) > 1), abc_distance)
    set.seed(seed)
    f <- do.call(calibrate, c(list(y, cal), kn))
    f$method <- m
    if (m %in% c("ABC-SMC", "ABC-MCMC")) f$abc_distance <- abc_distance
    f
  })
  if (length(fits) == 1) fits[[1]] else structure(fits, class = "window_fits")
}

# The effective reproduction number on the first day of a fit's window: R0 x the share still susceptible.
.reff <- function(f) {
  st <- f$estimate[setdiff(names(f$estimate), c("R0", "ptran", "crate"))]
  f$estimate[["R0"]] * (1 - sum(st))
}

.fit_table <- function(fits) {
  do.call(rbind, lapply(names(fits), function(m) {
    f <- fits[[m]]
    data.frame(method = m, R0 = f$estimate[["R0"]], R0_lower = f$intervals["R0", 1], R0_upper = f$intervals["R0", 2],
               Reff = .reff(f), seconds = f$diagnostics$seconds, model_runs = f$diagnostics$model_runs)
  }))
}

#' @export
summary.window_fits <- function(object, ...) .fit_table(object)

#' @export
print.window_fits <- function(x, ...) {
  cat(length(x), "fits of a", length(x[[1]]$observed), "day", x[[1]]$model, "window\n")
  tab <- .fit_table(x); num <- vapply(tab, is.numeric, TRUE); tab[num] <- lapply(tab[num], signif, 3)
  print(tab, row.names = FALSE)
  cat("R0 with its 95% range; Reff = R0 x the share susceptible on the first day of the window.\n")
  invisible(x)
}

#' Plot the fit of a window
#'
#' The 95% band and median of [predict_cases()] over the window (and `ahead`
#' days after it) with the observed counts: one panel per method for a
#' `window_fits` list.
#'
#' @param x A `window_fit` or `window_fits` from [calibrate_window()].
#' @param ahead Days after the window to run the model on.
#' @param future Counts observed after the window (optional).
#' @param n Number of model runs.
#' @param ... Passed to [plot_bands()] (for example `ncol`).
#' @return Invisibly, the coverage and interval score of each band.
#' @export
plot.window_fit <- function(x, ahead = 0L, future = NULL, n = 200L, ...) {
  plot_bands(stats::setNames(list(predict_cases(x, ahead = ahead, n = n)), x$method), future = future, ...)
}

#' @rdname plot.window_fit
#' @export
plot.window_fits <- function(x, ahead = 0L, future = NULL, n = 200L, ...) {
  plot_bands(lapply(x, predict_cases, ahead = ahead, n = n), future = future, ...)
}

#' The known values of a disease
#'
#' The values the models need besides the counts, as in the paper, from the
#' incubation period and serial interval in [disease_parameters]: for SEIR,
#' `incub` = the incubation period and `recov` = 1 / (serial interval minus
#' incubation period, at least 4 days); for SIR, `recov` = 1 / the serial
#' interval (between 4 and 14 days).
#'
#' @param disease One of `disease_parameters$disease`: `"COVID-19"`,
#'   `"influenza"`, `"SARS"`, `"Ebola"`, `"MERS"`, `"smallpox"`, `"measles"`.
#' @param model `"SEIR"` or `"SIR"`.
#' @param n The (effective) population the counts come from. The shipped
#'   networks were trained on populations of 5,000 to 10,000; the paper uses
#'   10,000 for every curve.
#' @return A list with `n`, `recov` and, for SEIR, `incub`.
#' @examples
#' known_values("COVID-19", "SEIR")
#' known_values("influenza", "SIR")
#' @export
known_values <- function(disease, model = c("SEIR", "SIR"), n = 10000) {
  model <- .window_info(model[1])$model
  tab <- .dataset("disease_parameters")
  d <- tab[tolower(tab$disease) == tolower(disease[1]), ]
  if (!nrow(d)) stop("Unknown disease \"", disease[1], "\". Use one of: ", paste(tab$disease, collapse = ", "), ".",
                     call. = FALSE)
  if (model == "SEIR") list(n = n, recov = 1 / max(4, d$serial_interval - d$incubation), incub = d$incubation) else
    list(n = n, recov = 1 / min(14, max(4, d$serial_interval)))
}
