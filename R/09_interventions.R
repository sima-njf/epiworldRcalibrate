# =============================================================================
# 09_interventions.R
# Interventions: where the curve changed after a known measure, and the
# epidemic without the measure
#
# This file contains:
#   - find_intervention(): the breakpoint that follows a known restriction
#     within the incubation period and at which the growth rate falls
#   - intervention_effect(): the fit to the part before the cut run on over the
#     part after it, as if transmission had not changed; the cases expected
#     without the measure and the cases averted
#
# As in the paper: the breakpoints are found from the counts alone (segmented
# regression, split_curve()) and only then compared with the dates of the known
# measures (Kuechenhoff et al. 2021; Boudou et al. 2021).
# =============================================================================

#' Find the breakpoint of a curve that follows a known measure
#'
#' The breakpoints are estimated from the counts alone ([split_curve()]) and
#' then compared with the dates of the known restrictions. A breakpoint is an
#' *intervention cut* when it comes 0 to `max_lag` days after a known
#' restriction (by default 14 days, the longest incubation period of COVID-19,
#' as in Boudou et al. 2021) and the growth rate falls there; of several, the
#' one where the growth rate falls most is chosen, matched to the latest
#' restriction before it.
#'
#' @param x A `curve_split` ([split_curve()]) or a `curve_fit`
#'   ([calibrate_curve()] with `split = TRUE`).
#' @param interventions Known measures: a data frame with `date`, `measure` and
#'   `kind` (only `"restriction"` rows are used), such as [interventions], or
#'   dates.
#' @param outbreak Optional: keep only the rows of `interventions` with this
#'   `outbreak` (for example `"New Zealand COVID-19"`).
#' @param max_lag Longest delay, in days, from a measure to the breakpoint.
#' @return An `intervention_match` data frame, one row per breakpoint: its
#'   day and date, the growth rate per day before and after it and the change,
#'   the latest restriction 0 to `max_lag` days before it (date, measure and
#'   delay), whether it qualifies (a restriction before it and a fall in
#'   growth), and which one is `chosen`.
#' @examples
#' data(covid_waves_2020); data(interventions)
#' nz <- subset(covid_waves_2020, region == "New Zealand")
#' s <- split_curve(nz$cases, nz$date, weekday = TRUE)
#' find_intervention(s, interventions, outbreak = "New Zealand COVID-19")
#' @export
find_intervention <- function(x, interventions, outbreak = NULL, max_lag = 14L) {
  sp <- if (inherits(x, "curve_fit")) x$split else x
  if (!inherits(sp, "curve_split"))
    stop("x must be a curve_split, or a curve_fit made with split = TRUE.", call. = FALSE)
  m <- .as_measures(interventions, outbreak, kind = "restriction")
  b <- sp$breakpoints
  out <- data.frame(breakpoint_day = b$day, breakpoint = b$date, growth_before = b$growth_before,
                    growth_after = b$growth_after, change = b$growth_after - b$growth_before)
  out$measure_date <- if (inherits(sp$dates, "Date")) as.Date(rep(NA, nrow(b))) else rep(NA_real_, nrow(b))
  out$measure <- rep(NA_character_, nrow(b)); out$lag_days <- rep(NA_real_, nrow(b))
  if (!is.null(m) && nrow(m) && nrow(b)) {
    if (inherits(sp$dates, "Date") != inherits(m$date, "Date"))
      stop("the measures and the curve must both have dates (class Date), or both days.", call. = FALSE)
    for (i in seq_len(nrow(b))) {
      lag <- as.numeric(b$date[i]) - as.numeric(m$date); j <- which(lag >= 0 & lag <= max_lag)
      if (!length(j)) next
      j <- j[which.min(lag[j])]
      out$measure_date[i] <- m$date[j]; out$measure[i] <- m$measure[j]; out$lag_days[i] <- lag[j]
    }
  }
  out$qualifies <- !is.na(out$measure) & out$change < 0
  out$chosen <- out$qualifies & out$change == suppressWarnings(min(out$change[out$qualifies]))
  structure(out, class = c("intervention_match", "data.frame"), max_lag = max_lag)
}

#' @export
print.intervention_match <- function(x, ...) {
  d <- as.data.frame(x); num <- vapply(d, is.numeric, TRUE) & !names(d) %in% c("breakpoint_day", "lag_days")
  d[num] <- lapply(d[num], signif, 3)
  if (!nrow(d)) cat("The curve has no breakpoint.\n") else print(d, row.names = FALSE)
  if (any(x$chosen)) {
    i <- which(x$chosen)
    cat("Intervention cut:", format(x$breakpoint[i]), "-", x$lag_days[i], "days after", format(x$measure_date[i]),
        "(", x$measure[i], "); growth", sprintf("%+.3f", x$growth_before[i]), "->", sprintf("%+.3f", x$growth_after[i]),
        "per day.\n")
  } else if (nrow(d)) {
    cat("No breakpoint comes 0 to", attr(x, "max_lag"), "days after a known restriction with a fall in growth.\n")
  }
  invisible(x)
}

# The daily new infectious of n runs of a fit's model for `days` days from the state on the first day of its window,
# with its parameter draws (days x runs).
.fit_runs <- function(fit, days, n, seed) {
  info <- .window_info(fit$model)
  set.seed(seed)
  pick <- if (length(fit$draws) == n) fit$draws else fit$draws[sample(length(fit$draws), n, replace = TRUE)]
  vapply(seq_len(n), function(i) .run_window_model(info, pick[[i]], days, seed = i)$new, numeric(days))
}

#' The epidemic without a measure: cases expected and averted
#'
#' The fits to the two parts next to a cut give the reproduction number before
#' and after it. To see what changed at the cut, the fit to the part before the
#' cut is run on over the part after it, as if transmission had not changed:
#' `n` runs of the agent-based model from the state estimated on the first day
#' of the part before the cut, each with one of the method's parameter draws.
#' Compared with the counts observed after the cut, the runs give the cases
#' expected without the measure and the cases averted, with 95% intervals.
#'
#' This is a scenario of the model rather than a forecast: it assumes that
#' nothing but the measure changed at the cut. When the counts after the cut
#' approach the effective population `n`, the fit after the cut describes an
#' epidemic running out of susceptible people rather than one changed by the
#' measure; when the counts before the cut are few, the runs may die out.
#'
#' @param fit A `curve_fit` from [calibrate_curve()], split so that a part
#'   starts at the cut.
#' @param cut The first day after the cut: a date or a day of the curve. By
#'   default the intervention cut found by [find_intervention()] from
#'   `interventions`.
#' @param interventions,outbreak,max_lag Passed to [find_intervention()] when
#'   `cut` is not given; `interventions` also labels the measure.
#' @param n Number of model runs.
#' @param early_days The cases averted are also given for the first
#'   `early_days` days after the cut.
#' @param seed Random seed.
#' @return An `intervention_effect`: `summary` (a one-row data frame: the cut,
#'   the measure, the days before and after, R0 and the effective reproduction
#'   number (Reff = R0 x the share susceptible on the first day of the part)
#'   before and after, the cases observed after the cut, the cases expected
#'   without the measure (median and 95% interval), and the cases averted and
#'   their share, also over the first `early_days` days), `bands` (by day: the
#'   observed counts, the band of each part's own fit, and the band without the
#'   measure), and `match` (from [find_intervention()], if used). `plot()` shows
#'   it.
#' @examples
#' \dontrun{
#' nz <- subset(covid_waves_2020, region == "New Zealand")
#' fit <- calibrate_curve(nz$cases, nz$date, model = "SEIR", weekday = TRUE,
#'                        known = known_values("COVID-19", "SEIR"))
#' eff <- intervention_effect(fit, interventions = interventions, outbreak = "New Zealand COVID-19")
#' eff
#' plot(eff)
#' }
#' @export
intervention_effect <- function(fit, cut = NULL, interventions = NULL, outbreak = NULL, max_lag = 14L, n = 500L,
                                early_days = 28L, seed = 1L) {
  if (!inherits(fit, "curve_fit")) stop("fit must be a curve_fit (from calibrate_curve()).", call. = FALSE)
  mt <- NULL; measure_date <- NA; measure <- NA_character_
  if (is.null(cut)) {
    if (is.null(interventions)) stop("Give the cut (the first day after it), or the known measures.", call. = FALSE)
    mt <- find_intervention(fit, interventions, outbreak, max_lag)
    if (!any(mt$chosen)) stop("No breakpoint comes 0 to ", max_lag, " days after a known restriction with a fall in ",
                              "growth; give the cut yourself.", call. = FALSE)
    i <- which(mt$chosen); cut <- mt$breakpoint[i]; measure_date <- mt$measure_date[i]; measure <- mt$measure[i]
  } else if (!is.null(interventions)) {
    m <- .as_measures(interventions, outbreak, kind = "restriction")
    lag <- as.numeric(cut) - as.numeric(m$date); j <- which(lag >= 0 & lag <= max_lag)
    if (length(j)) { j <- j[which.min(lag[j])]; measure_date <- m$date[j]; measure <- m$measure[j] }
  }
  day <- if (inherits(cut, "Date")) match(cut, fit$dates) else as.integer(cut)
  p <- fit$parts; k <- which(p$start == day)
  if (is.na(day) || length(k) != 1 || k == 1)
    stop("The cut must be the first day of a calibrated part other than the first; the parts start on ",
         paste(format(fit$dates[p$start[-1]]), collapse = ", "), ".", call. = FALSE)
  if (p$end[k - 1] != day - 1L) stop("The part just before the cut was not calibrated.", call. = FALSE)
  before <- fit$fits[[p$part[k - 1]]]; after <- fit$fits[[p$part[k]]]
  L1 <- p$days[k - 1]; L2 <- p$days[k]
  without <- .fit_runs(before, L1 + L2, n, seed)
  own_after <- .fit_runs(after, L2, n, seed + 1L)
  obs <- fit$cases[p$start[k - 1]:p$end[k]]; obs_after <- obs[L1 + seq_len(L2)]
  early <- min(early_days, L2)
  q <- function(z) stats::quantile(z, c(0.5, 0.025, 0.975), names = FALSE)
  pct <- function(o, w) if (!is.finite(w) || w < 1) NA_real_ else 100 * (1 - o / w)
  tot <- q(colSums(without[L1 + seq_len(L2), , drop = FALSE]))
  tot_e <- q(colSums(without[L1 + seq_len(early), , drop = FALSE]))
  summary <- data.frame(cut = fit$dates[day], measure_date = measure_date, measure = measure, days_before = L1,
                        days_after = L2, R0_before = before$estimate[["R0"]], R0_after = after$estimate[["R0"]],
                        Reff_before = .reff(before), Reff_after = .reff(after), observed_after = sum(obs_after),
                        expected_without = tot[1], expected_lower = tot[2], expected_upper = tot[3],
                        averted = tot[1] - sum(obs_after), averted_pct = pct(sum(obs_after), tot[1]),
                        averted_pct_lower = pct(sum(obs_after), tot[2]), averted_pct_upper = pct(sum(obs_after), tot[3]),
                        early_days = early, observed_early = sum(obs_after[seq_len(early)]), expected_early = tot_e[1],
                        averted_early_pct = pct(sum(obs_after[seq_len(early)]), tot_e[1]))
  qd <- function(r, pr) apply(r, 1, stats::quantile, pr, names = FALSE)
  fit_runs <- rbind(without[seq_len(L1), , drop = FALSE], own_after)
  idx <- p$start[k - 1]:p$end[k]
  bands <- data.frame(day = idx, date = fit$dates[idx], observed = obs, side = rep(c("before", "after"), c(L1, L2)),
                      fit_median = qd(fit_runs, 0.5), fit_lower = qd(fit_runs, 0.025), fit_upper = qd(fit_runs, 0.975),
                      without_median = c(rep(NA, L1), qd(without[L1 + seq_len(L2), , drop = FALSE], 0.5)),
                      without_lower = c(rep(NA, L1), qd(without[L1 + seq_len(L2), , drop = FALSE], 0.025)),
                      without_upper = c(rep(NA, L1), qd(without[L1 + seq_len(L2), , drop = FALSE], 0.975)))
  structure(list(summary = summary, bands = bands, match = mt, method = fit$method, model = fit$model,
                 population = fit$known$n), class = "intervention_effect")
}

#' @export
print.intervention_effect <- function(x, ...) {
  s <- x$summary; f <- function(v) formatC(round(v), format = "d", big.mark = ",")
  cat(x$method, x$model, "- the epidemic without the change at", format(s$cut), "\n")
  if (!is.na(s$measure)) cat("Measure:", format(s$measure_date), s$measure, "\n")
  cat(sprintf("R0   before %.2f  after %.2f\nReff before %.2f  after %.2f  (R0 x the share susceptible)\n",
              s$R0_before, s$R0_after, s$Reff_before, s$Reff_after))
  cat("Cases in the", s$days_after, "days after the cut: observed", f(s$observed_after), "; expected without the change",
      f(s$expected_without), "(95% interval", f(s$expected_lower), "to", paste0(f(s$expected_upper), ")"), "\n")
  cat(sprintf("Averted: %s cases, %.0f%% (%.0f%% to %.0f%%); over the first %d days %.0f%%\n", f(s$averted),
              s$averted_pct, s$averted_pct_lower, s$averted_pct_upper, s$early_days, s$averted_early_pct))
  if (!is.null(x$population) && s$observed_after > 0.5 * x$population)
    cat("Note: the counts after the cut exceed half of the effective population (n =", f(x$population),
        "); the fall may come from fewer susceptible people rather than from the measure.\n")
  cat("A scenario of the model, not a forecast: it assumes that nothing but the measure changed at the cut.\n")
  invisible(x)
}

#' Plot the epidemic without a measure
#'
#' The observed counts around the cut, the band of each part's own fit (blue),
#' the band of the part before the cut run on over the part after it (pink:
#' the epidemic without the change), the cut (dashed) and the measure (dotted).
#'
#' @param x An `intervention_effect`.
#' @param main,xlab,ylab Title and axis labels.
#' @param ... Passed to [plot()].
#' @return `x`, invisibly.
#' @export
plot.intervention_effect <- function(x, main = NULL, xlab = "", ylab = "Daily count", ...) {
  b <- x$bands; s <- x$summary
  top <- max(c(b$observed, b$fit_upper, b$without_upper), na.rm = TRUE) * 1.08
  graphics::plot(b$date, b$observed, type = "n", ylim = c(0, top), xlab = xlab, ylab = ylab, bty = "n", las = 1,
                 main = if (is.null(main)) paste(x$method, x$model, "- the epidemic without the change") else main, ...)
  .grid(x = FALSE)
  band <- function(z, lo, md, hi, col) {
    graphics::polygon(c(z$date, rev(z$date)), c(z[[lo]], rev(z[[hi]])), col = grDevices::adjustcolor(col, 0.25), border = NA)
    graphics::lines(z$date, z[[md]], col = col, lwd = 2)
  }
  for (sd in c("before", "after")) band(b[b$side == sd, ], "fit_lower", "fit_median", "fit_upper", .COL$bilstm)
  band(b[b$side == "after", ], "without_lower", "without_median", "without_upper", "#c0306a")
  graphics::points(b$date, b$observed, pch = 16, cex = 0.45, col = .COL$ink)
  graphics::abline(v = s$cut, lty = 2, col = .COL$ink)
  has_m <- !is.na(s$measure)
  if (has_m) graphics::abline(v = s$measure_date, lty = 3, lwd = 1.6, col = .COL_MEASURE[["restriction"]])
  graphics::legend("topleft", bty = "n", cex = 0.75, lwd = c(2, 2, NA, 1, if (has_m) 1.6), lty = c(1, 1, NA, 2, if (has_m) 3),
                   pch = c(NA, NA, 16, NA, if (has_m) NA),
                   col = c(.COL$bilstm, "#c0306a", .COL$ink, .COL$ink, if (has_m) .COL_MEASURE[["restriction"]]),
                   legend = c("each part's fit (median, 95% band)", "part before the cut run on: without the change",
                              "observed counts", "cut", if (has_m) "measure"))
  f <- function(v) formatC(round(v), format = "d", big.mark = ",")
  graphics::mtext(sprintf("After the cut: %s cases observed, %s expected without the change (95%% %s to %s): %.0f%% averted",
                          f(s$observed_after), f(s$expected_without), f(s$expected_lower), f(s$expected_upper),
                          s$averted_pct), side = 3, line = 0.2, adj = 0, cex = 0.72, col = .COL$other)
  if (has_m) graphics::mtext(paste("Measure:", format(s$measure_date), s$measure), side = 1, line = 2.6, adj = 0,
                             cex = 0.62, col = .COL$other)
  invisible(x)
}
