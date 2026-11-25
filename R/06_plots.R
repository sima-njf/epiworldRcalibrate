# =============================================================================
# 06_plots.R
# Plots to compare calibration methods (base graphics)
#
# This file contains:
#   - plot_bands(): the 95% bands of predict_cases() for several methods
#   - plot_comparison(): R0 error, coverage, interval score and time by method,
#     from compare_methods()
#   - plot_estimates(): R0 (or ptran, crate) with 95% ranges by method and
#     window, against the truth
#   - plot_ptran_crate(): the draws of fits in the ptran-crate plane, with the
#     curve of constant R0 (only the product of ptran and crate is learned)
# =============================================================================

.COL <- list(bilstm = "#2a78d6", other = "#52514e", truth = "#eb6834", ink = "#0b0b0b", grid = "#e4e3df")
.method_col <- function(m) ifelse(grepl("^(BiLSTM|DeepIMC)", m), .COL$bilstm, .COL$other)
.grid <- function(x = TRUE, y = TRUE) {
  u <- graphics::par("usr")
  if (y) graphics::abline(h = pretty(u[3:4]), col = .COL$grid, lwd = 1)
  if (x) graphics::abline(v = pretty(u[1:2]), col = .COL$grid, lwd = 1)
}

#' Plot the 95% bands of several methods
#'
#' One panel per method: the band and median of [predict_cases()], the observed
#' window (filled points) and, if given, the counts after it (open points), with
#' the coverage and interval score of the window (and of the days after it).
#'
#' @param bands A data frame from [predict_cases()], or a named list of them
#'   (names: the methods).
#' @param future Counts after the window (optional).
#' @param what `"new"` (daily new infectious) or `"infected"` (number
#'   infectious; then give the true values in `truth`).
#' @param truth Values to show instead of the observed counts (for example the
#'   true number infectious of a simulated epidemic), covering the window and
#'   the days after it.
#' @param ncol Number of panel columns.
#' @param xlab,ylab Axis labels.
#' @return Invisibly, a data frame of the scores shown.
#' @examples
#' \dontrun{
#' bands <- lapply(fits, predict_cases, ahead = 14)
#' plot_bands(bands, future = after_window)
#' }
#' @export
plot_bands <- function(bands, future = NULL, what = c("new", "infected"), truth = NULL, ncol = min(3, length(bands)),
                       xlab = "day of the window", ylab = NULL) {
  what <- match.arg(what)
  if (is.data.frame(bands)) bands <- list(Fit = bands)
  if (is.null(names(bands))) names(bands) <- paste("Fit", seq_along(bands))
  L <- sum(bands[[1]]$in_window)
  y <- if (!is.null(truth)) as.numeric(truth) else c(bands[[1]]$observed[seq_len(L)], as.numeric(future))
  if (is.null(ylab)) ylab <- if (what == "new") "daily new infections" else "number infectious"
  cols <- paste0(what, c("_median", "_lower", "_upper"))
  ylim <- range(c(0, y, unlist(lapply(bands, function(b) b[, cols]))), na.rm = TRUE)
  nrow <- ceiling(length(bands) / ncol)
  op <- graphics::par(mfrow = c(nrow, ncol), mar = c(4, 4, 3.2, 0.8), mgp = c(2.4, 0.7, 0), las = 1)
  on.exit(graphics::par(op))
  scores <- list()
  for (m in names(bands)) {
    b <- bands[[m]]
    days <- seq_len(min(nrow(b), length(y)))
    graphics::plot(NA, xlim = c(1, nrow(b)), ylim = ylim, xlab = xlab, ylab = ylab, bty = "n")
    .grid()
    graphics::polygon(c(b$day, rev(b$day)), c(b[[cols[3]]], rev(b[[cols[2]]])),
                      col = grDevices::adjustcolor(.method_col(m), 0.25), border = NA)
    graphics::lines(b$day, b[[cols[1]]], col = .method_col(m), lwd = 2)
    if (nrow(b) > L) graphics::abline(v = L + 0.5, lty = 2, col = .COL$other)
    inw <- days <= L
    graphics::points(days[inw], y[days][inw], pch = 16, cex = 0.55, col = .COL$ink)
    if (any(!inw)) graphics::points(days[!inw], y[days][!inw], pch = 1, cex = 0.7, col = .COL$ink)
    s_w <- score_band(b, y[seq_len(L)], seq_len(L), what)
    s_a <- if (any(!inw)) score_band(b, y[days][!inw], days[!inw], what) else c(coverage = NA, interval_score = NA)
    scores[[m]] <- data.frame(method = m, coverage = s_w[["coverage"]], interval_score = s_w[["interval_score"]],
                              coverage_ahead = s_a[["coverage"]], interval_score_ahead = s_a[["interval_score"]])
    graphics::title(m, adj = 0, line = 1.9, cex.main = 1.05, font.main = 2)
    graphics::mtext(sprintf("window: coverage %.0f%%, IS %.3g%s", s_w[["coverage"]], s_w[["interval_score"]],
                            if (any(!inw)) sprintf(" | after: coverage %.0f%%", s_a[["coverage"]]) else ""),
                    side = 3, adj = 0, line = 0.6, cex = 0.72, col = .COL$other)
  }
  invisible(do.call(rbind, scores))
}

#' Compare methods: errors, coverage, interval score and time
#'
#' One panel per measure, one row per method: a small point for each window and
#' a large point for the mean (blue: DeepIMC). Coverage has a line at 95%;
#' time is on a log scale.
#'
#' @param results A data frame from [compare_methods()].
#' @param measures Columns of `results` to show; those that are all `NA` are
#'   left out.
#' @param ncol Number of panel columns.
#' @return Invisibly, the means by method.
#' @export
plot_comparison <- function(results, measures = c("R0_error", "state_error", "coverage", "interval_score",
                                                  "coverage_ahead", "seconds"), ncol = 3) {
  labels <- c(R0_error = "R0 absolute error", ptran_error = "ptran absolute error",
              crate_error = "crate absolute error", state_error = "Starting-state error (points)",
              coverage = "Coverage of the window (%)", interval_score = "Interval score, window",
              coverage_ahead = "Coverage after the window (%)", interval_score_ahead = "Interval score, after the window",
              seconds = "Seconds per window (log scale)", model_runs = "Model runs per window")
  results <- results[!results$failed, , drop = FALSE]
  measures <- measures[measures %in% names(results)]
  measures <- measures[vapply(measures, function(v) any(is.finite(results[[v]])), logical(1))]
  methods <- unique(results$method)
  k <- length(methods)
  nrow <- ceiling(length(measures) / ncol)
  op <- graphics::par(mfrow = c(nrow, min(ncol, length(measures))), mar = c(3.6, 7.5, 2.4, 1), mgp = c(2.2, 0.6, 0),
                      las = 1)
  on.exit(graphics::par(op))
  means <- data.frame(method = methods)
  for (v in measures) {
    x <- results[[v]]
    logx <- v %in% c("seconds", "model_runs")
    if (logx) x <- pmax(x, 1e-3)
    mu <- vapply(methods, function(m) mean(x[results$method == m], na.rm = TRUE), 0)
    means[[v]] <- mu
    xlim <- range(c(x, if (grepl("^coverage", v)) c(95, 100)), na.rm = TRUE)
    graphics::plot(NA, xlim = xlim, ylim = c(k + 0.5, 0.5), log = if (logx) "x" else "", yaxt = "n",
                   xaxt = if (logx) "n" else "s", xlab = "", ylab = "", bty = "n")
    if (logx) {
      at <- 10^(floor(log10(xlim[1])):ceiling(log10(xlim[2])))
      graphics::axis(1, at = at, labels = formatC(at, format = "fg"))
    }
    graphics::abline(h = seq_len(k), col = .COL$grid)
    if (grepl("^coverage", v)) graphics::abline(v = 95, lty = 2, col = .COL$truth, lwd = 1.5)
    yy <- match(results$method, methods) + stats::runif(nrow(results), -0.15, 0.15)
    graphics::points(x, yy, pch = 16, cex = 0.5, col = grDevices::adjustcolor(.method_col(results$method), 0.45))
    graphics::points(mu, seq_len(k), pch = 21, cex = 1.7, bg = .method_col(methods), col = "white", lwd = 1.5)
    graphics::axis(2, at = seq_len(k), labels = methods, tick = FALSE)
    graphics::title(if (v %in% names(labels)) labels[[v]] else v, adj = 0, cex.main = 0.95, font.main = 2)
  }
  invisible(means)
}

#' Estimates with 95% ranges, by window and method
#'
#' One panel per window: each method's estimate (point) and 95% range (bar,
#' when the method gives one), with the true value as an orange line when
#' known.
#'
#' @param results A data frame from [compare_methods()].
#' @param parameter `"R0"`, `"ptran"` or `"crate"`.
#' @param windows Which windows (values of `results$window`); at most 12 are
#'   shown.
#' @param ncol Number of panel columns.
#' @return Invisibly, `NULL`.
#' @export
plot_estimates <- function(results, parameter = c("R0", "ptran", "crate"), windows = NULL, ncol = 3) {
  parameter <- match.arg(parameter)
  results <- results[!results$failed, , drop = FALSE]
  if (is.null(windows)) windows <- unique(results$window)
  windows <- utils::head(windows, 12)
  methods <- unique(results$method)
  k <- length(methods)
  est <- results[[parameter]]; lo <- results[[paste0(parameter, "_low")]]; hi <- results[[paste0(parameter, "_high")]]
  tru <- results[[paste0(parameter, "_true")]]
  xlim <- range(c(est, lo, hi, tru)[results$window %in% windows], na.rm = TRUE)
  op <- graphics::par(mfrow = c(ceiling(length(windows) / ncol), min(ncol, length(windows))),
                      mar = c(3.6, 7.5, 2.4, 1), mgp = c(2.2, 0.6, 0), las = 1)
  on.exit(graphics::par(op))
  for (w in windows) {
    r <- which(results$window == w)
    yy <- match(results$method[r], methods)
    graphics::plot(NA, xlim = xlim, ylim = c(k + 0.5, 0.5), yaxt = "n", xlab = parameter, ylab = "", bty = "n")
    graphics::abline(h = seq_len(k), col = .COL$grid)
    if (any(is.finite(tru[r]))) graphics::abline(v = tru[r][is.finite(tru[r])][1], col = .COL$truth, lwd = 2)
    ok <- is.finite(lo[r]) & is.finite(hi[r])
    graphics::segments(lo[r][ok], yy[ok], hi[r][ok], yy[ok], lwd = 3, col = .method_col(results$method[r][ok]),
                       lend = 1)
    graphics::points(est[r], yy, pch = 21, cex = 1.5, bg = .method_col(results$method[r]), col = "white", lwd = 1.5)
    graphics::axis(2, at = seq_len(k), labels = methods, tick = FALSE)
    lab <- paste(c(w, unlist(results[r[1], intersect(c("wave", "position"), names(results))]),
                   paste(results$days[r[1]], "days")), collapse = ", ")
    graphics::title(lab, adj = 0, cex.main = 0.85, font.main = 2)
  }
  invisible(NULL)
}

#' The draws of fits in the ptran-crate plane
#'
#' Shows why only R0 = ptran x crate / recov is learned: the draws of a method
#' that searches ptran and crate (for example ABC-SMC from
#' `window_baselines(unknowns = "ptran_crate")`) spread along the curve of
#' constant R0 instead of around one point.
#'
#' @param fits A `window_fit` or a named list of them.
#' @param truth Optional true values, `c(ptran = , crate = )`; its R0 curve is
#'   drawn in orange.
#' @param ncol Number of panel columns.
#' @return Invisibly, `NULL`.
#' @export
plot_ptran_crate <- function(fits, truth = NULL, ncol = min(3, length(fits))) {
  if (inherits(fits, "window_fit")) fits <- list(fits)
  if (is.null(names(fits))) names(fits) <- vapply(fits, `[[`, "", "method")
  op <- graphics::par(mfrow = c(ceiling(length(fits) / ncol), ncol), mar = c(4, 5.6, 3, 1), mgp = c(2.6, 0.7, 0),
                      las = 1)
  on.exit(graphics::par(op))
  for (m in names(fits)) {
    f <- fits[[m]]
    info <- .window_info(f$model)
    recov <- f$known$recov
    pt <- vapply(f$draws, `[[`, 0, "ptran"); cr <- vapply(f$draws, `[[`, 0, "crate")
    xlim <- info$crate_range
    ylim <- range(c(pt, f$estimate[["ptran"]], truth[["ptran"]], .ptran_range(info)), na.rm = TRUE)
    graphics::plot(NA, xlim = xlim, ylim = ylim, log = "y", xlab = "crate (contacts per day)", ylab = "", bty = "n")
    graphics::title(ylab = "ptran (transmission probability)", line = 3.9)
    .grid(y = FALSE)
    cc <- seq(xlim[1], xlim[2], length.out = 100)
    graphics::lines(cc, f$estimate[["R0"]] * recov / cc, col = .method_col(m), lwd = 2, lty = 2)
    if (!is.null(truth)) {
      R0_true <- truth[["ptran"]] * truth[["crate"]] / recov
      graphics::lines(cc, R0_true * recov / cc, col = .COL$truth, lwd = 2)
      graphics::points(truth[["crate"]], truth[["ptran"]], pch = 4, cex = 1.6, lwd = 2.5, col = .COL$truth)
    }
    graphics::points(cr, pt, pch = 16, cex = 0.5, col = grDevices::adjustcolor(.method_col(m), 0.35))
    graphics::points(f$estimate[["crate"]], f$estimate[["ptran"]], pch = 21, cex = 1.7, bg = .method_col(m),
                     col = "white", lwd = 1.5)
    graphics::title(m, adj = 0, line = 1.6, cex.main = 1.05, font.main = 2)
    graphics::mtext(sprintf("R0 %.2f; dashed: constant R0%s", f$estimate[["R0"]],
                            if (is.null(truth)) "" else "; orange: the truth"),
                    side = 3, adj = 0, line = 0.4, cex = 0.72, col = .COL$other)
  }
  invisible(NULL)
}
