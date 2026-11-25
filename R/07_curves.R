# Whole epidemic curves: split at the breakpoints, smooth each part, calibrate each part, and run the epidemic ------
#
# A model with constant parameters cannot follow a curve whose course changes (an intervention, a change in
# behaviour, a new variant). calibrate_curve() splits the curve where its growth rate changes (segmented regression,
# split_curve()), smooths each part on its own if asked (smooth_cases()), and calibrates each part in its own window
# with DeepIMC or one of the comparison methods. run_epidemic() runs the calibrated model again over each part (the
# epidemic the estimates imply), and can run each part's fit on over the next part, as if the change had not
# happened. Nothing here forecasts beyond the observed days.

# The bandwidth of a Gaussian kernel smoother chosen by leave-one-out cross-validation (Wand and Jones 1995, ch. 5):
# of the fractions of the series `fracs`, the one whose kernel best predicts each day from the other days. The
# kernel is that of stats::ksmooth(): a normal density with its quartiles at +/- bandwidth / 4.
.cv_bandwidth <- function(y, fracs = seq(0.02, 1, by = 0.01)) {
  n <- length(y); d <- outer(seq_len(n), seq_len(n), "-")
  cv <- vapply(fracs, function(fr) {
    s <- 0.25 * fr * n / stats::qnorm(0.75)
    w <- exp(-0.5 * (d / s)^2); diag(w) <- 0
    den <- rowSums(w); if (any(den < 1e-12)) return(Inf)
    mean((y - drop(w %*% y) / den)^2)
  }, 0)
  fracs[which.min(cv)]
}

# The Gaussian kernel of stats::ksmooth(kernel = "normal", bandwidth = frac x days) (standard deviation 0.3706506 x
# bandwidth) without its cut-off at 4 standard deviations, from the days `use` only, evaluated on every day. It is the
# smoother of the bandwidth chosen by calibration, which smooths from every other day.
.nw <- function(y, frac, use = seq_along(y)) {
  n <- length(y); s <- 0.25 * frac * n / stats::qnorm(0.75)
  w <- exp(-0.5 * (outer(seq_len(n), use, "-") / s)^2)
  drop(w %*% y[use]) / rowSums(w)
}

# Weighted interval score (Bracher et al. 2021, eq. 1) of model runs (a days x runs matrix) against the counts y,
# averaged over the days, with the intervals of the COVID-19 Forecast Hub: half the absolute error of the median plus
# the interval scores of the K = 11 central intervals (alpha = 0.02, 0.05, 0.1, ..., 0.9), each weighted by alpha / 2,
# divided by K + 1/2.
.WIS_ALPHA <- c(0.02, 0.05, seq(0.1, 0.9, by = 0.1))
.wis <- function(runs, y) {
  is <- function(a) { l <- apply(runs, 1, stats::quantile, a / 2, names = FALSE)
                      u <- apply(runs, 1, stats::quantile, 1 - a / 2, names = FALSE)
                      (u - l) + 2 / a * pmax(l - y, 0) + 2 / a * pmax(y - u, 0) }
  w <- 0.5 * abs(y - apply(runs, 1, stats::median)) + Reduce(`+`, lapply(.WIS_ALPHA, function(a) a / 2 * is(a)))
  mean(w / (length(.WIS_ALPHA) + 0.5))
}

# The daily new infectious of n runs of the model over a fit's window, from its parameter draws (days x runs).
.window_runs <- function(fit, n, seed) {
  info <- .window_info(fit$model); days <- length(fit$observed)
  set.seed(seed)
  pick <- if (length(fit$draws) == n) fit$draws else fit$draws[sample(length(fit$draws), n, replace = TRUE)]
  vapply(seq_len(n), function(i) .run_window_model(info, pick[[i]], days, seed = i)$new, numeric(days))
}

# The bandwidths tried when the bandwidth is chosen by calibration (fractions of the part).
.CALIBRATION_FRACS <- c(0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)

# The bandwidth chosen by calibration, as in the paper: for each fraction, the counts are smoothed from every other day
# only, calibrated by DeepIMC (`deep`, a window_calibrator()) and run `runs` times, and the runs are scored by the WIS
# against the raw counts of the days left out; then the two sets of days swap and the two scores are averaged. The
# fraction with the lowest held-out WIS is chosen, so that the counts that score a bandwidth never smooth or calibrate
# with it. The DeepIMC fits are made first, in this process; the model runs can be spread over `cores` processes.
.calibration_bandwidth <- function(y, deep, runs = 200L, cores = 1L, seed = 1L, ...) {
  n <- length(y)
  if (n < 15L || n > 365L) stop("choosing the bandwidth by calibration needs 15 to 365 days (DeepIMC); this part has ", n, ".")
  jobs <- expand.grid(fold = 1:2, frac = .CALIBRATION_FRACS)
  fits <- lapply(seq_len(nrow(jobs)), function(k) {
    sm <- .nw(y, jobs$frac[k], seq(jobs$fold[k], n, by = 2L))
    if (any(!is.finite(sm))) return(NULL)
    tryCatch(calibrate(sm, deep, ...), error = function(e) NULL)
  })
  score <- function(k) {
    if (is.null(fits[[k]])) return(NA_real_)
    held <- seq(3L - jobs$fold[k], n, by = 2L)
    r <- tryCatch(.window_runs(fits[[k]], runs, seed + k), error = function(e) NULL)
    if (is.null(r)) NA_real_ else .wis(r[held, , drop = FALSE], y[held])
  }
  ho <- if (cores > 1L && .Platform$OS.type != "windows") {
    res <- parallel::mclapply(seq_len(nrow(jobs)), score, mc.cores = cores, mc.preschedule = FALSE)
    vapply(res, function(z) if (is.numeric(z) && length(z) == 1) z else NA_real_, 0)
  } else vapply(seq_len(nrow(jobs)), score, 0)
  scores <- data.frame(frac = .CALIBRATION_FRACS,
                       heldout_WIS = vapply(.CALIBRATION_FRACS, function(fr) mean(ho[jobs$frac == fr]), 0))
  if (all(is.na(scores$heldout_WIS))) stop("no bandwidth could be scored by calibration.")
  list(frac = scores$frac[which.min(scores$heldout_WIS)], scores = scores)
}

# A part smoothed with the bandwidth chosen by calibration: the smoothed counts, with the fraction in the attribute
# "bandwidth" and the held-out WIS of every fraction in the attribute "scores".
.smooth_by_calibration <- function(y, deep, runs = 200L, cores = 1L, seed = 1L, ...) {
  b <- .calibration_bandwidth(y, deep, runs = runs, cores = cores, seed = seed, ...)
  structure(.nw(y, b$frac), bandwidth = b$frac, scores = b$scores)
}

# The centred 7-day moving average (fewer days at the ends of the series).
.ma7 <- function(y) { n <- length(y); vapply(seq_len(n), function(t) mean(y[max(1L, t - 3L):min(n, t + 3L)]), 0) }

#' Smooth daily counts
#'
#' By default (`smoothing = "ma7"`) the centred 7-day moving average, the usual
#' smoothing of daily counts before an epidemic model is fitted (it removes a
#' weekly reporting cycle); fewer days are averaged at the ends of the series.
#' Otherwise a Gaussian kernel smoother (`stats::ksmooth`, normal kernel)
#' whose bandwidth is a fraction of the length of the series. The fraction can
#' be given, or chosen from the data in one of two ways.
#'
#' * `smoothing = "cv"`: from the counts themselves, by leave-one-out
#'   cross-validation (Wand and Jones, 1995, Kernel Smoothing): of the fractions
#'   from 2% to 100% of the series, the one whose kernel best predicts each
#'   day from the other days. It needs no model runs.
#' * `smoothing = "calibration"`: by calibrating DeepIMC on held-out days, as in
#'   the paper. For each of ten fractions from 2% to 100%, the counts are
#'   smoothed from every other day only, calibrated by DeepIMC
#'   ([window_calibrator()]) and run `runs` times with the agent-based model,
#'   and the runs are scored by the weighted interval score against the raw
#'   counts of the days left out; then the two sets of days swap. The fraction
#'   with the lowest mean held-out score is chosen and the counts are smoothed
#'   with it from all days. It needs the `model` and its known values (`n`,
#'   `recov` and, for SEIR, `incub`), 15 to 365 days, and 20 DeepIMC fits and
#'   `20 * runs` model runs (about two minutes on one core with the default 200
#'   runs).
#'
#' @param cases Daily counts.
#' @param smoothing `"ma7"` (centred 7-day moving average, the default and the
#'   main smoothing of the paper), `"cv"` (Gaussian kernel, bandwidth chosen by
#'   cross-validation), `"calibration"` (bandwidth chosen by calibrating DeepIMC
#'   on held-out days), `"none"`, or a number between 0 and 1: the bandwidth of
#'   the Gaussian kernel as a fraction of the series (0.2 and 0.5 are used in
#'   the paper).
#' @param model `"SEIR"` or `"SIR"` (for `smoothing = "calibration"`).
#' @param ... The known values for `smoothing = "calibration"`: `n`, `recov`
#'   and, for SEIR, `incub`.
#' @param known The known values as a list instead (for example from
#'   [known_values()]).
#' @param runs Model runs per fraction and set of days (`"calibration"`).
#' @param cores Processes for the model runs (`"calibration"`; forked, so 1 on
#'   Windows).
#' @param seed Random seed (`"calibration"`).
#' @return The smoothed counts, with the kernel's fraction in the attribute
#'   `"bandwidth"` (0 for `"none"`, `NA` for `"ma7"`); with `"calibration"`, the
#'   held-out score of every fraction tried in the attribute `"scores"`.
#' @examples
#' y <- c(1, 3, 2, 6, 9, 7, 14, 18, 15, 25, 30, 22, 35, 41, 30, 44, 50, 38)
#' smooth_cases(y)
#' smooth_cases(y, "cv")
#' smooth_cases(y, 0.2)
#' \dontrun{
#' smooth_cases(y, "calibration", model = "SEIR", known = known_values("COVID-19", "SEIR"))
#' }
#' @export
smooth_cases <- function(cases, smoothing = "ma7", model = c("SEIR", "SIR"), ..., known = NULL, runs = 200L, cores = 1L,
                         seed = 1L) {
  y <- as.numeric(cases)
  if (identical(smoothing, "none") || identical(smoothing, FALSE) || identical(smoothing, 0)) return(structure(y, bandwidth = 0))
  if (identical(smoothing, "ma7") || isTRUE(smoothing)) return(structure(.ma7(y), bandwidth = NA_real_))
  if (identical(smoothing, "calibration")) {
    kn <- utils::modifyList(if (is.null(known)) list() else as.list(known), list(...))
    return(do.call(.smooth_by_calibration, c(list(y, .method_calibrator("DeepIMC", model[1]), runs = runs, cores = cores,
                                                  seed = seed), kn)))
  }
  frac <- if (identical(smoothing, "cv")) .cv_bandwidth(y) else suppressWarnings(as.numeric(smoothing))
  if (length(frac) != 1 || !is.finite(frac) || frac <= 0 || frac > 1)
    stop("smoothing must be \"ma7\", \"cv\", \"calibration\", \"none\" or a fraction between 0 and 1.")
  x <- seq_along(y)
  structure(stats::ksmooth(x, y, kernel = "normal", bandwidth = frac * length(y), x.points = x)$y, bandwidth = frac)
}

#' Split an epidemic curve where its growth rate changes
#'
#' Segmented (breakpoint) regression (Muggeo, 2003; R package `segmented`), as
#' Kuechenhoff et al. (2021) used it for the first COVID-19 wave in Germany: a
#' negative binomial regression of the daily counts on time whose slope, the
#' growth rate of log counts per day, changes at K breakpoints, with
#' day-of-week effects if `weekday = TRUE` (for counts by date of report).
#' K = 0 to `max_breakpoints` is chosen by BIC among the fits whose parts all
#' have at least `min_days` days. For each K the breakpoints are estimated by
#' `segmented::segmented()` (20 bootstrap restarts), alternating with the
#' negative binomial size.
#'
#' @param cases Daily counts.
#' @param dates Optional dates of the counts (class `Date`); by default the
#'   days are numbered 1, 2, ...
#' @param weekday Add day-of-week effects (needs `dates`).
#' @param max_breakpoints The largest number of breakpoints tried.
#' @param min_days The shortest part allowed (15, the shortest window of the
#'   shipped networks).
#' @return A `curve_split`: `breakpoints` (the first day of each new part, its
#'   date, and the growth rate per day before and after it), `parts` (first and
#'   last day, dates, days and cases of each part), the `bic` of each K, and
#'   `fitted` (the expected counts of the chosen regression). `plot()` shows
#'   the counts, the regression and the breakpoints.
#' @examples
#' data(sars_hongkong_2003)
#' s <- split_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date)
#' s
#' plot(s)
#' @export
split_curve <- function(cases, dates = NULL, weekday = FALSE, max_breakpoints = 5L, min_days = 15L) {
  for (p in c("segmented", "MASS"))
    if (!requireNamespace(p, quietly = TRUE)) stop("split_curve() needs the package ", p, ": install.packages(\"", p, "\")")
  y <- as.numeric(cases); n <- length(y)
  if (is.null(dates)) dates <- seq_len(n)
  if (length(dates) != n) stop("dates must have the same length as cases.")
  if (weekday && !inherits(dates, "Date")) stop("weekday = TRUE needs dates of class Date.")
  d <- data.frame(y = y, t = seq_len(n))
  f <- if (weekday) { d$wd <- factor(weekdays(dates)); y ~ t + wd } else y ~ t
  nb0 <- suppressWarnings(MASS::glm.nb(f, data = d))
  fits <- list(list(K = 0L, bic = stats::BIC(nb0), psi = numeric(0), slope = stats::coef(nb0)[["t"]],
                    fitted = as.numeric(stats::fitted(nb0))))
  if (n >= 2 * min_days) for (K in seq_len(max_breakpoints)) {
    theta <- nb0$theta; psi <- NULL; nbK <- NULL
    for (it in 1:10) {
      g <- stats::glm(f, family = MASS::negative.binomial(theta), data = d)
      ctl <- segmented::seg.control(n.boot = 20, seed = 1)
      s <- tryCatch(suppressWarnings(if (is.null(psi)) segmented::segmented(g, seg.Z = ~t, npsi = K, control = ctl) else
                                       segmented::segmented(g, seg.Z = ~t, psi = psi, control = ctl)), error = function(e) NULL)
      if (is.null(s$psi) || nrow(s$psi) != K) { nbK <- NULL; break }
      psi <- sort(unname(s$psi[, "Est."]))
      U <- outer(d$t, psi, function(t, p) pmax(t - p, 0)); colnames(U) <- paste0("U", seq_len(K))
      nbK <- tryCatch(suppressWarnings(MASS::glm.nb(stats::update(f, stats::as.formula(paste(". ~ . +", paste(colnames(U), collapse = " + ")))),
                                   data = cbind(d, U))), error = function(e) NULL)
      if (is.null(nbK)) break
      done <- abs(nbK$theta - theta) < 1e-3 * theta; theta <- nbK$theta
      if (done) break
    }
    if (is.null(nbK) || any(diff(c(1L, floor(psi) + 1L, n + 1L)) < min_days)) next
    ll <- stats::logLik(nbK); b <- stats::coef(nbK)[paste0("U", seq_len(K))]
    fits[[length(fits) + 1]] <- list(K = K, bic = -2 * as.numeric(ll) + (attr(ll, "df") + K) * log(n), psi = psi,
                                     slope = stats::coef(nbK)[["t"]] + c(0, cumsum(unname(b))),
                                     fitted = as.numeric(stats::fitted(nbK)))
  }
  bic <- vapply(fits, function(z) z$bic, 0); best <- fits[[which.min(bic)]]
  pos <- floor(best$psi) + 1L
  starts <- c(1L, pos); ends <- c(pos - 1L, n)
  structure(list(
    breakpoints = data.frame(day = pos, date = dates[pos], growth_before = utils::head(best$slope, -1),
                             growth_after = best$slope[-1])[seq_len(best$K), , drop = FALSE],
    parts = data.frame(part = seq_along(starts), start = starts, end = ends, from = dates[starts], to = dates[ends],
                       days = ends - starts + 1L, cases = vapply(seq_along(starts), function(i) sum(y[starts[i]:ends[i]]), 0)),
    bic = stats::setNames(bic, vapply(fits, function(z) z$K, 0L)), cases = y, dates = dates, fitted = best$fitted,
    weekday = weekday),
    class = "curve_split")
}

#' @export
print.curve_split <- function(x, ...) {
  K <- nrow(x$breakpoints)
  cat("Segmented regression:", K, if (K == 1) "breakpoint" else "breakpoints", "(chosen by BIC),", nrow(x$parts), "parts\n")
  if (K) { b <- x$breakpoints; b$growth_before <- round(b$growth_before, 3); b$growth_after <- round(b$growth_after, 3); print(b, row.names = FALSE) }
  print(x$parts, row.names = FALSE)
  invisible(x)
}

# The calibrator of a method: DeepIMC (the shipped window networks) or a comparison method, as in the paper (the ABC
# methods on summary statistics unless abc_distance = "curve"), or a calibrator object.
.curve_calibrator <- function(method, model, settings = list(), abc_distance = "summary") {
  if (inherits(method, "calibrator")) return(method)
  .method_calibrator(method, model, settings, abc_distance)
}

#' Calibrate a whole epidemic curve, with or without smoothing and splitting
#'
#' A model with constant parameters cannot follow a curve whose course changes
#' (an intervention, a change in behaviour, a new variant). With `split =
#' TRUE` the curve is split where its growth rate changes ([split_curve()]);
#' each part is smoothed on its own ([smooth_cases()]) and calibrated in its own
#' window by DeepIMC or one of the standard methods ([calibrate_window()]).
#' Splitting first keeps the smoothing from carrying the counts of one part into
#' the next. With `split = FALSE` the whole curve is one window; with `smoothing
#' = FALSE` (or `"none"`) the raw counts are calibrated.
#'
#' The counts are read as the daily new infections (or cases, or admissions)
#' of an effective population of `n` people; give the known values of the
#' disease: `n`, `recov` (1 / infectious period) and, for SEIR, `incub` (the
#' latent period in days), or `known = known_values("COVID-19", "SEIR")`. Each
#' part needs 15 to 365 days for DeepIMC, and at most `n` cases (the model
#' cannot infect more people than it has); other parts are skipped with a
#' warning. The paper treats every curve as the epidemic of an effective
#' population of `n = 10000`, the range the networks were trained on.
#'
#' @param cases Daily counts.
#' @param dates Optional dates of the counts (class `Date`).
#' @param model `"SEIR"` or `"SIR"`.
#' @param method `"DeepIMC"` (default), `"ABC_SMC"`, `"ABC_MCMC"`, `"DE"`,
#'   `"NELDERMEAD"` (any spelling, see [calibrate_window()]), or a calibrator.
#' @param split `TRUE` (split by segmented regression), `FALSE` (the whole
#'   curve in one window), or the days (positions or dates) at which new parts
#'   start.
#' @param smoothing `TRUE` or `"ma7"` (centred 7-day moving average, the default
#'   and the main smoothing of the paper), `FALSE` or `"none"` (the raw counts),
#'   `"cv"` (Gaussian kernel, bandwidth chosen by cross-validation of the
#'   counts), `"calibration"` (bandwidth chosen by calibrating DeepIMC on
#'   held-out days; see [smooth_cases()]), or a fraction of each part for the
#'   Gaussian kernel (for example 0.2). With `"calibration"` the bandwidth of
#'   each part is always chosen with DeepIMC, whatever the `method`, and the
#'   part smoothed with it is calibrated by `method`; it takes about two minutes
#'   per part on one core.
#' @param ... The known values: `n`, `recov` and, for SEIR, `incub`; and
#'   optionally `crate`, the contact rate per day if it is known from other
#'   data (a contact survey): ptran = R0 x recov / crate is then reported for
#'   each part (the curve alone cannot separate ptran from crate).
#' @param known The known values as a list instead (for example from
#'   [known_values()]); values in `...` replace those in `known`.
#' @param weekday,max_breakpoints,min_days Passed to [split_curve()]. Use
#'   `weekday = TRUE` for counts by date of report, which have a weekly cycle.
#' @param settings,abc_distance Passed to [calibrate_window()].
#' @param smoothing_runs Model runs per bandwidth and set of days when
#'   `smoothing = "calibration"`.
#' @param cores Processes for those model runs (forked, so 1 on Windows).
#' @param seed Random seed, set before each part.
#' @return A `curve_fit`: `parts` (each part's days, cases, bandwidth, R0 with
#'   its 95% range, effective reproduction number on its first day, ptran,
#'   crate and seconds), `fits` (the `window_fit` of each part), `split` (the
#'   [split_curve()] result, if any), the counts, the `smoothed` counts that
#'   were calibrated and, with `smoothing = "calibration"`, `bandwidth_scores`
#'   (the held-out score of every bandwidth tried for each part). `plot()`
#'   shows the curve the fits imply ([recover_curve()]).
#' @examples
#' \dontrun{
#' data(sars_hongkong_2003)
#' kv <- known_values("SARS", "SEIR")
#' # Split by segmented regression, each part smoothed by the 7-day average
#' fit <- calibrate_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date, model = "SEIR", known = kv)
#' fit
#' plot(fit)
#' # One window, raw counts, differential evolution
#' fit_de <- calibrate_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date, model = "SEIR", method = "DE",
#'                           split = FALSE, smoothing = FALSE, known = kv)
#' }
#' @export
calibrate_curve <- function(cases, dates = NULL, model = c("SEIR", "SIR"), method = "DeepIMC", split = TRUE,
                            smoothing = "ma7", ..., known = NULL, weekday = FALSE, max_breakpoints = 5L, min_days = 15L,
                            settings = list(), abc_distance = c("summary", "curve"), smoothing_runs = 200L, cores = 1L,
                            seed = 1L) {
  model <- .window_info(model[1])$model
  abc_distance <- match.arg(abc_distance)
  kn <- utils::modifyList(if (is.null(known)) list() else as.list(known), list(...))
  if (isTRUE(smoothing)) smoothing <- "ma7" else if (isFALSE(smoothing)) smoothing <- "none"
  y <- as.numeric(cases); n <- length(y)
  if (!n || anyNA(y) || any(y < 0)) stop("cases must be non-negative counts without NA.", call. = FALSE)
  if (is.null(dates)) dates <- seq_len(n)
  if (length(dates) != n) stop("dates must have the same length as cases.", call. = FALSE)
  sp <- NULL
  starts <- if (isTRUE(split)) {
    sp <- split_curve(y, dates, weekday = weekday, max_breakpoints = max_breakpoints, min_days = min_days)
    sp$parts$start
  } else if (isFALSE(split)) 1L else sort(unique(c(1L, if (inherits(split, "Date")) match(split, dates) else as.integer(split))))
  if (anyNA(starts) || any(starts < 1 | starts > n)) stop("split days must be inside the curve.", call. = FALSE)
  ends <- c(starts[-1] - 1L, n)
  mname <- if (is.character(method)) .method_name(method)[1] else "calibrator"
  cal <- .curve_calibrator(if (is.character(method)) mname else method, model,
                           .settings_for(settings, mname, FALSE), abc_distance)
  by_cal <- identical(smoothing, "calibration")
  deep <- if (by_cal) .method_calibrator("DeepIMC", model) else NULL
  fits <- list(); rows <- list(); scores <- list(); smoothed <- rep(NA_real_, n)
  for (p in seq_along(starts)) {
    i <- starts[p]:ends[p]
    if (mname == "DeepIMC" && (length(i) < 15 || length(i) > 365)) {
      warning("part ", p, " has ", length(i), " days; DeepIMC needs 15 to 365. It is not calibrated."); next
    }
    if (is.numeric(kn$n) && sum(y[i]) > kn$n) {
      warning("part ", p, " has ", sum(y[i]), " cases, more than the population n = ", kn$n, ". It is not calibrated.")
      next
    }
    sm <- if (by_cal && length(i) >= 15 && length(i) <= 365) {
      do.call(.smooth_by_calibration, c(list(y[i], deep, runs = smoothing_runs, cores = cores, seed = p), kn))
    } else {
      if (by_cal) warning("part ", p, " has ", length(i), " days; choosing the bandwidth by calibration needs 15 to 365 ",
                          "(DeepIMC). It is smoothed with the cross-validated bandwidth.")
      smooth_cases(y[i], if (by_cal) "cv" else smoothing)
    }
    smoothed[i] <- as.numeric(sm)
    if (!is.null(attr(sm, "scores"))) scores[[length(scores) + 1]] <- data.frame(part = p, attr(sm, "scores"))
    set.seed(seed + p - 1L)
    f <- do.call(calibrate, c(list(as.numeric(sm), cal), kn))
    if (mname != "calibrator") f$method <- mname
    fits[[p]] <- f
    rows[[p]] <- data.frame(part = p, start = starts[p], end = ends[p], from = dates[starts[p]], to = dates[ends[p]],
                            days = length(i), cases = sum(y[i]), bandwidth = attr(sm, "bandwidth"),
                            R0 = f$estimate[["R0"]], R0_lower = f$intervals["R0", 1], R0_upper = f$intervals["R0", 2],
                            Reff = .reff(f), ptran = f$estimate[["ptran"]],
                            crate = f$estimate[["crate"]], seconds = f$diagnostics$seconds)
  }
  if (!length(rows)) stop("no part could be calibrated.", call. = FALSE)
  structure(list(parts = do.call(rbind, rows), fits = fits, split = sp, cases = y, dates = dates, smoothed = smoothed,
                 model = model, method = mname, smoothing = smoothing, known = kn,
                 bandwidth_scores = if (length(scores)) do.call(rbind, scores) else NULL), class = "curve_fit")
}

.smoothing_label <- function(s) {
  if (identical(s, "ma7")) "7-day moving average" else if (identical(s, "none")) "none (raw counts)" else
    if (is.character(s)) s else paste0("Gaussian kernel, ", 100 * s, "% of each part")
}

#' @export
print.curve_fit <- function(x, ...) {
  cat(x$method, "calibration of a", length(x$cases), "day", x$model, "curve in", nrow(x$parts),
      if (nrow(x$parts) == 1) "part" else "parts", if (!is.null(x$split)) "(split by segmented regression)" else "",
      "; smoothing:", .smoothing_label(x$smoothing), "\n")
  p <- x$parts; num <- vapply(p, is.numeric, TRUE); p[num] <- lapply(p[num], signif, 3)
  print(p[, c("part", "from", "to", "days", "cases", "R0", "R0_lower", "R0_upper", "Reff", "seconds")], row.names = FALSE)
  cat("R0 = ptran x crate / recov; Reff = R0 x the share susceptible on the first day of the part.\n")
  invisible(x)
}

#' Recover the epidemic curve from a calibration, and show it
#'
#' Runs the agent-based model again over each part of a [calibrate_curve()]
#' fit, `n` times from the estimated state on the first day of the part with
#' parameters drawn from the method's uncertainty: the epidemic the estimates
#' imply (its median and 95% band), to compare with the observed counts. With
#' `run_on = TRUE` each part's fit is also run on over the next part, as if
#' the change at the breakpoint had not happened (the epidemic without the
#' change, for example without an intervention; see also
#' [intervention_effect()]). No days after the curve are predicted.
#' `recover_curve()` and `run_epidemic()` are the same function.
#'
#' @param fit A `curve_fit` from [calibrate_curve()].
#' @param n Number of runs per part.
#' @param run_on Also run each part's fit on over the next part.
#' @param seed Random seed.
#' @return A `curve_run`: a data frame with, for each part and day, the date,
#'   the observed count, and the median and 95% band of the daily new
#'   infectious (`what = "fit"`, and `what = "run on"` over the next part);
#'   plot it with `plot()`, which can also mark known measures.
#' @examples
#' \dontrun{
#' fit <- calibrate_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date, model = "SEIR",
#'                        known = known_values("SARS", "SEIR"))
#' run <- recover_curve(fit, run_on = TRUE)
#' plot(run, interventions = subset(interventions, outbreak == "Hong Kong SARS 2003"))
#' }
#' @export
recover_curve <- function(fit, n = 500L, run_on = FALSE, seed = 1L) {
  if (!inherits(fit, "curve_fit")) stop("fit must be a curve_fit (from calibrate_curve()).", call. = FALSE)
  p <- fit$parts; out <- list()
  for (k in seq_len(nrow(p))) {
    f <- fit$fits[[p$part[k]]]
    nxt <- if (run_on && k < nrow(p) && p$start[k + 1] == p$end[k] + 1L) p$days[k + 1] else 0L
    b <- predict_cases(f, ahead = nxt, n = n, seed = seed + k)
    i <- p$start[k] + seq_len(nrow(b)) - 1L
    out[[k]] <- data.frame(part = p$part[k], what = ifelse(b$in_window, "fit", "run on"), day = i, date = fit$dates[i],
                           observed = fit$cases[i], median = b$new_median, lower = b$new_lower, upper = b$new_upper)
  }
  structure(do.call(rbind, out), class = c("curve_run", "data.frame"), fit = fit)
}

#' @rdname recover_curve
#' @export
run_epidemic <- recover_curve

# Known measures as a data frame with date and measure (from a data frame such as `interventions`, or dates).
.as_measures <- function(interventions, outbreak = NULL, kind = NULL) {
  if (is.null(interventions)) return(NULL)
  if (inherits(interventions, "Date") || is.numeric(interventions))
    return(data.frame(date = interventions, measure = format(interventions), kind = "restriction"))
  m <- as.data.frame(interventions)
  if (!"date" %in% names(m)) stop("interventions must have a column date (or be dates).", call. = FALSE)
  if (!is.null(outbreak) && "outbreak" %in% names(m)) m <- m[m$outbreak %in% outbreak, , drop = FALSE]
  if (!"measure" %in% names(m)) m$measure <- format(m$date)
  if (!"kind" %in% names(m)) m$kind <- "restriction"
  if (!is.null(kind)) m <- m[m$kind %in% kind, , drop = FALSE]
  m[order(m$date), , drop = FALSE]
}

# The numbered measures under the plot, two per line.
.list_measures <- function(m) {
  if (is.null(m) || !nrow(m)) return(invisible(NULL))
  txt <- paste0(seq_len(nrow(m)), ": ", format(m$date), " ", m$measure)
  txt <- ifelse(nchar(txt) > 70, paste0(substr(txt, 1, 67), "..."), txt)
  lines <- vapply(split(txt, ceiling(seq_along(txt) / 2)), paste, "", collapse = "    ")
  for (j in seq_along(lines)) graphics::mtext(lines[j], side = 1, line = 2.2 + 0.75 * j, adj = 0, cex = 0.6, col = .COL$other)
  invisible(m)
}

.COL_MEASURE <- c(restriction = "#c2410c", relaxation = "#15803d", other = "#7c3aed")
# Vertical lines at known measures inside the x range, numbered at the top; returns the measures drawn.
.draw_measures <- function(m, xlim, top) {
  if (is.null(m) || !nrow(m)) return(invisible(NULL))
  m <- m[as.numeric(m$date) >= xlim[1] & as.numeric(m$date) <= xlim[2], , drop = FALSE]
  if (!nrow(m)) return(invisible(NULL))
  col <- ifelse(m$kind %in% names(.COL_MEASURE), .COL_MEASURE[m$kind], .COL_MEASURE[["other"]])
  graphics::abline(v = m$date, col = col, lty = 3, lwd = 1.6)
  graphics::text(m$date, top, seq_len(nrow(m)), col = col, cex = 0.75, font = 2, pos = 4, offset = 0.2)
  invisible(m)
}

#' @param x A `curve_run`.
#' @param interventions Known measures to mark: a data frame with `date` and
#'   `measure` (and `kind`: restrictions in orange, relaxations in green), such
#'   as a subset of [interventions], or dates. They are numbered at the top of
#'   the plot and listed under it.
#' @param smoothed Also draw the smoothed counts that were calibrated.
#' @param main,xlab,ylab Title and axis labels.
#' @param ... Passed to [plot()].
#' @rdname recover_curve
#' @export
plot.curve_run <- function(x, interventions = NULL, smoothed = TRUE, main = NULL, xlab = "", ylab = "Daily count", ...) {
  fit <- attr(x, "fit"); d <- as.data.frame(x)
  top <- max(c(fit$cases, d$upper), na.rm = TRUE) * 1.08
  graphics::plot(fit$dates, fit$cases, type = "n", ylim = c(0, top), xlab = xlab, ylab = ylab, bty = "n", las = 1,
                 main = if (is.null(main)) paste(fit$method, fit$model, "fit of each part") else main, ...)
  .grid(x = FALSE)
  band <- function(z, col) {
    graphics::polygon(c(z$date, rev(z$date)), c(z$lower, rev(z$upper)), col = grDevices::adjustcolor(col, 0.25), border = NA)
    graphics::lines(z$date, z$median, col = col, lwd = 2)
  }
  for (k in unique(d$part)) {
    band(d[d$part == k & d$what == "fit", ], .COL$bilstm)
    if (any(d$part == k & d$what == "run on")) band(d[d$part == k & d$what == "run on", ], "#c0306a")
  }
  graphics::points(fit$dates, fit$cases, pch = 16, cex = 0.45, col = .COL$ink)
  sm <- isTRUE(smoothed) && !is.null(fit$smoothed) && !identical(fit$smoothing, "none")
  if (sm) graphics::lines(fit$dates, fit$smoothed, col = .COL$other, lwd = 1)
  multi <- nrow(fit$parts) > 1
  if (multi) graphics::abline(v = fit$dates[fit$parts$start[-1]], lty = 2, col = .COL$ink)
  m <- .draw_measures(.as_measures(interventions), range(as.numeric(fit$dates)), top)
  ron <- any(d$what == "run on"); km <- !is.null(m)
  graphics::legend("topright", bty = "n", cex = 0.75, inset = c(0, 0.04),
                   lwd = c(2, if (ron) 2, NA, if (sm) 1, if (multi) 1, if (km) 1.6),
                   lty = c(1, if (ron) 1, NA, if (sm) 1, if (multi) 2, if (km) 3),
                   pch = c(NA, if (ron) NA, 16, if (sm) NA, if (multi) NA, if (km) NA),
                   col = c(.COL$bilstm, if (ron) "#c0306a", .COL$ink, if (sm) .COL$other, if (multi) .COL$ink,
                           if (km) .COL_MEASURE[["restriction"]]),
                   legend = c("each part calibrated (median, 95% band)", if (ron) "run on without the change",
                              "observed counts", if (sm) "smoothed counts", if (multi) "breakpoints",
                              if (km) "known measures (numbered)"))
  .list_measures(m)
  invisible(x)
}

#' @export
plot.curve_fit <- function(x, interventions = NULL, n = 500L, run_on = FALSE, ...)
  plot(recover_curve(x, n = n, run_on = run_on), interventions = interventions, ...)

#' @param x A `curve_split`.
#' @param interventions Known measures to mark (see [recover_curve()]).
#' @param main,xlab,ylab Title and axis labels.
#' @param ... Passed to [plot()].
#' @rdname split_curve
#' @export
plot.curve_split <- function(x, interventions = NULL, main = "Segmented regression", xlab = "", ylab = "Daily count", ...) {
  top <- max(c(x$cases, x$fitted), na.rm = TRUE) * 1.1
  graphics::plot(x$dates, x$cases, type = "n", ylim = c(0, top), xlab = xlab, ylab = ylab, bty = "n", las = 1,
                 main = main, ...)
  .grid(x = FALSE)
  graphics::points(x$dates, x$cases, pch = 16, cex = 0.45, col = .COL$ink)
  graphics::lines(x$dates, .ma7(x$cases), col = .COL$other, lwd = 1)
  graphics::lines(x$dates, x$fitted, col = .COL$bilstm, lwd = 2)
  K <- nrow(x$breakpoints)
  if (K) graphics::abline(v = x$dates[x$breakpoints$day], lty = 2, col = .COL$ink)
  slope <- c(if (K) x$breakpoints$growth_before else NULL, if (K) utils::tail(x$breakpoints$growth_after, 1) else NULL)
  if (!K) slope <- log(max(x$fitted[length(x$fitted)], 1e-9) / max(x$fitted[1], 1e-9)) / max(1, length(x$fitted) - 1)
  mid <- (x$parts$start + x$parts$end) / 2
  graphics::text(x$dates[round(mid)], top * 0.97, sprintf("%+.3f/day", slope), cex = 0.7, col = .COL$bilstm)
  m <- .draw_measures(.as_measures(interventions), range(as.numeric(x$dates)), top * 0.9)
  graphics::legend("topright", bty = "n", cex = 0.75, inset = c(0, 0.06),
                   lwd = c(NA, 1, 2, if (K) 1, if (!is.null(m)) 1.6), lty = c(NA, 1, 1, if (K) 2, if (!is.null(m)) 3),
                   pch = c(16, NA, NA, if (K) NA, if (!is.null(m)) NA),
                   col = c(.COL$ink, .COL$other, .COL$bilstm, if (K) .COL$ink, if (!is.null(m)) .COL_MEASURE[["restriction"]]),
                   legend = c("daily counts", "7-day average", "regression (growth rate per day above)",
                              if (K) "breakpoints", if (!is.null(m)) "known measures (numbered)"))
  .list_measures(m)
  invisible(x)
}
