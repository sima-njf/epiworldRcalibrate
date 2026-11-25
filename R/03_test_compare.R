# =============================================================================
# 03_test_compare.R
# Step 3: apply a calibrator, test it, and compare against standard baselines
#
# This file contains:
#   - ABC-MCMC
#   - ABC-SMC
#   - Nelder-Mead
#   - Differential Evolution
#   - generic comparison metrics
#   - optional SIR/SEIR convenience wrappers
#
# Standard study design:
#   primary method (BiLSTM / Transformer / other)
#       vs ABC-MCMC, ABC-SMC, Nelder-Mead, and DE
# =============================================================================

# Generic calibration entry point ---------------------------------------------

#' Calibrate observed data
#'
#' This function is deliberately model- and method-agnostic.  All details are
#' supplied by the `calibrator` object.
#'
#' @param daily_cases Observed epidemic curve or other input expected by the
#'   calibrator.
#' @param calibrator A `calibrator` object.
#' @param ... Additional arguments passed to all three calibrator steps.
#' @return The post-processed calibration result.
#' @export
calibrate <- function(daily_cases, calibrator, ...) {

  if (!inherits(calibrator, "calibrator"))
    stop(
      "The object -calibrator- should be of class 'calibrator'. ",
      "It is of class ", paste(class(calibrator), collapse = ", ")
    )

  ans <- calibrator$pre_process(daily_cases, ...)
  ans <- calibrator$run(ans, ...)
  calibrator$post_process(ans, ...)
}



# Generic benchmark calibration methods ---------------------------------------
#
# These constructors adapt the four comparison methods used in the project to
# the same `calibrator` interface as BiLSTM:
#   * ABC-MCMC
#   * ABC-SMC
#   * Nelder-Mead
#   * Differential Evolution
#
# They are deliberately model-agnostic. A simulator and parameter bounds are
# supplied by the user. The epidemic model may be SIR, SEIR, SIRH, SEIRD, or a
# custom simulator.

.validate_bounds <- function(bounds) {
  if (is.list(bounds) && !is.data.frame(bounds)) {
    if (is.null(names(bounds)) || any(names(bounds) == "")) {
      stop("A bounds list must be named.")
    }
    bad <- vapply(bounds, length, integer(1)) != 2L
    if (any(bad)) {
      stop("Each element of bounds must contain c(lower, upper).")
    }
    out <- data.frame(
      lower = vapply(bounds, function(x) as.numeric(x[[1]]), numeric(1)),
      upper = vapply(bounds, function(x) as.numeric(x[[2]]), numeric(1)),
      row.names = names(bounds),
      check.names = FALSE
    )
  } else {
    out <- as.data.frame(bounds)
    if (ncol(out) != 2L) {
      stop("bounds must have exactly two columns: lower and upper.")
    }
    names(out) <- c("lower", "upper")
    if (is.null(rownames(out)) || any(rownames(out) == "")) {
      stop("Matrix/data-frame bounds must use parameter names as row names.")
    }
  }

  out$lower <- as.numeric(out$lower)
  out$upper <- as.numeric(out$upper)

  if (any(!is.finite(out$lower)) || any(!is.finite(out$upper))) {
    stop("All bounds must be finite.")
  }
  if (any(out$lower >= out$upper)) {
    stop("Every lower bound must be smaller than its upper bound.")
  }

  out
}

.make_known <- function(dots, known_names) {
  missing <- setdiff(known_names, names(dots))
  if (length(missing)) {
    stop("Missing known parameter(s): ", paste(missing, collapse = ", "))
  }
  dots[known_names]
}

.merge_candidate <- function(known, theta) {
  if (length(intersect(names(known), names(theta)))) {
    stop(
      "A parameter cannot be both known and calibrated: ",
      paste(intersect(names(known), names(theta)), collapse = ", ")
    )
  }
  c(known, as.list(theta))
}

.metric_value <- function(observed, predicted, metric = c("rmse", "mse", "mae")) {
  metric <- match.arg(metric)
  observed <- as.numeric(observed)
  predicted <- as.numeric(predicted)

  if (length(observed) != length(predicted)) return(Inf)
  if (any(!is.finite(predicted))) return(Inf)

  err <- predicted - observed
  switch(
    metric,
    rmse = sqrt(mean(err^2)),
    mse = mean(err^2),
    mae = mean(abs(err))
  )
}

#' Summary statistics of an epidemic curve
#'
#' The statistics the ABC methods compare when `distance = "summary"`
#' (instead of the whole curve): for a run of daily counts, the mean, the peak
#' and the minimum (all as log(1 + count)), the day of the peak of the 7-day
#' average, the mean of the first and of the last week (log scale), the days
#' by which 25%, 50% and 75% of the cases of the run have happened, and the
#' growth rate (slope of log(1 + count) per day) over the first and the last
#' two weeks.
#'
#' The ABC methods divide each statistic by its spread (median absolute
#' deviation) over simulations from the prior, so that statistics in different
#' units count alike, and use the root mean square of the scaled differences
#' as the distance.
#'
#' @param x Daily counts.
#' @return A named numeric vector of 11 statistics.
#' @examples
#' epidemic_summary(c(1, 3, 8, 20, 41, 60, 52, 33, 20, 9, 4, 2, 1, 0))
#' @export
epidemic_summary <- function(x) {
  x <- pmax(as.numeric(x), 0)
  L <- length(x)
  k <- min(7L, L)
  w <- min(14L, L)
  ma <- if (L >= 7) as.numeric(stats::filter(x, rep(1 / 7, 7), sides = 2)) else rep(mean(x), L)
  ma[is.na(ma)] <- x[is.na(ma)]
  cs <- cumsum(x)
  day_q <- function(q) if (cs[L] > 0) which(cs >= q * cs[L])[1] else (L + 1) / 2
  slope <- function(v) if (length(v) < 2) 0 else stats::cov(seq_along(v), log1p(v)) / stats::var(seq_along(v))
  c(log_mean = log1p(mean(x)), log_peak = log1p(max(x)), log_min = log1p(min(x)), peak_day = which.max(ma),
    log_first_week = log1p(mean(x[seq_len(k)])), log_last_week = log1p(mean(x[(L - k + 1):L])),
    day_25 = day_q(0.25), day_50 = day_q(0.5), day_75 = day_q(0.75),
    growth_start = slope(x[seq_len(w)]), growth_end = slope(x[(L - w + 1):L]))
}

# Spread of each summary statistic over prior simulations (median absolute
# deviation, or the standard deviation when that is 0, or 1).
.summary_scale <- function(S) {
  sc <- apply(S, 2, stats::mad, na.rm = TRUE)
  sdv <- apply(S, 2, stats::sd, na.rm = TRUE)
  bad <- !is.finite(sc) | sc <= 0
  sc[bad] <- sdv[bad]
  sc[!is.finite(sc) | sc <= 0] <- 1
  sc
}

.summary_distance <- function(s_obs, s_sim, scale) {
  if (anyNA(s_sim)) return(Inf)
  sqrt(mean(((s_sim - s_obs) / scale)^2))
}

# Draw n parameter vectors uniformly within the bounds.
.runif_bounds <- function(n, bnd) {
  theta <- matrix(stats::runif(n * nrow(bnd)), nrow = n, ncol = nrow(bnd))
  theta <- sweep(theta, 2, bnd$upper - bnd$lower, `*`)
  theta <- sweep(theta, 2, bnd$lower, `+`)
  colnames(theta) <- rownames(bnd)
  theta
}

.safe_simulate <- function(simulator, parameters, seed) {
  tryCatch(
    as.numeric(simulator$run(parameters = parameters, seed = seed)),
    error = function(e) NULL
  )
}

.evaluate_theta <- function(
    theta,
    observed,
    known,
    simulator,
    seeds,
    metric
) {
  names(theta) <- names(theta)
  p <- .merge_candidate(known, theta)

  vals <- vapply(seeds, function(sd) {
    pred <- .safe_simulate(simulator, p, sd)
    if (is.null(pred)) return(Inf)
    .metric_value(observed, pred, metric)
  }, numeric(1))

  mean(vals)
}

.new_calibration_result <- function(
    estimate,
    method,
    posterior = NULL,
    intervals = NULL,
    diagnostics = list()
) {
  estimate <- as.numeric(estimate) |>
    stats::setNames(names(estimate))

  structure(
    list(
      estimate = estimate,
      method = method,
      posterior = posterior,
      intervals = intervals,
      diagnostics = diagnostics
    ),
    class = "calibration_result"
  )
}

#' @export
print.calibration_result <- function(x, ...) {
  cat("Calibration result:", x$method, "\n")
  print(x$estimate)
  if (length(x$diagnostics)) {
    cat("Diagnostics:\n")
    print(x$diagnostics)
  }
  invisible(x)
}

#' Extract a point estimate from a calibration result
#'
#' Works for both structured benchmark results and simple named numeric outputs
#' such as the BiLSTM adapter.
#' @export
calibration_estimate <- function(x) {
  if (inherits(x, "calibration_result")) return(x$estimate)
  if (is.numeric(x) && !is.null(names(x))) return(x)
  if (is.list(x) && !is.null(x$estimate)) return(unlist(x$estimate))
  stop("Could not extract a named point estimate from calibration result.")
}

.weighted_quantile <- function(x, w, probs) {
  o <- order(x)
  x <- x[o]
  w <- w[o]
  w <- w / sum(w)
  cw <- cumsum(w)

  vapply(
    probs,
    function(p) x[which(cw >= p)[1]],
    numeric(1)
  )
}

.bounds_midpoint <- function(bnd) {
  stats::setNames((bnd$lower + bnd$upper) / 2, rownames(bnd))
}

.in_bounds <- function(theta, bnd) {
  all(theta >= bnd$lower & theta <= bnd$upper)
}

#' Build a calibration method by name
#'
#' A single entry point for every search-based calibration method built into
#' this package (a trained BiLSTM is created separately, with
#' \code{pretrained_sir_calibrator()} / \code{bilstm_calibrator()}). Pick a
#' method by name and supply the search range (\code{bounds}) and which
#' values are already known (\code{known_names}); everything else is
#' forwarded to that method's own tuning arguments through \code{...} - for
#' example \code{n_samples}/\code{burnin} for \code{"ABC-MCMC"}, or
#' \code{maxiter} for \code{"DE"}.
#'
#' @param method One of \code{"ABC-MCMC"}, \code{"ABC-SMC"},
#'   \code{"NelderMead"}, \code{"DE"}.
#' @param simulator A \code{simulator} object.
#' @param bounds Search range for each parameter being calibrated.
#' @param known_names Names of parameters supplied at \code{calibrate()} time
#'   instead of being searched over.
#' @param derive Optional \code{function(estimate, known)} adding derived
#'   parameters (e.g. R0) to the result.
#' @param ... Extra tuning arguments for the chosen method.
#' @return A \code{calibrator} object.
#' @export
make_calibrator <- function(
    method,
    simulator,
    bounds,
    known_names = character(),
    derive = NULL,
    ...
) {
  switch(
    method,
    "ABC-MCMC"   = .abc_mcmc_calibrator(
      simulator = simulator, bounds = bounds, known_names = known_names,
      derive = derive, ...
    ),
    "ABC-SMC"    = .abc_smc_calibrator(
      simulator = simulator, bounds = bounds, known_names = known_names,
      derive = derive, ...
    ),
    "NelderMead" = .nelder_mead_calibrator(
      simulator = simulator, bounds = bounds, known_names = known_names,
      derive = derive, ...
    ),
    "DE"         = .de_calibrator(
      simulator = simulator, bounds = bounds, known_names = known_names,
      derive = derive, ...
    ),
    stop(
      "Unknown method \"", method, "\". Use one of ",
      "\"ABC-MCMC\", \"ABC-SMC\", \"NelderMead\", \"DE\"."
    )
  )
}

# ABC-MCMC calibrator ---------------------------------------------------------
# Uses an independent Uniform prior over the supplied box bounds, a symmetric
# Gaussian random-walk proposal, and the Gaussian ABC kernel
# exp(-d^2 / (2 * epsilon^2)). The default epsilon is 0.30 times the RMS of
# the observed series, matching the current project comparison.
# With distance = "summary" the curves are compared through epidemic_summary()
# (or the function `summary`): each statistic is scaled by its spread over
# n_pilot prior simulations, and epsilon is the pilot_quantile quantile of the
# pilot distances.
# Built by make_calibrator("ABC-MCMC", ...); not exported on its own.
.abc_mcmc_calibrator <- function(
    simulator,
    bounds,
    known_names = character(),
    n_samples = 3000L,
    burnin = 1500L,
    prop_frac = 0.08,
    eps_frac = 0.30,
    distance = "rmse",
    summary = epidemic_summary,
    n_pilot = 200L,
    pilot_quantile = 0.05,
    derive = NULL,
    type = simulator$type,
    name = "ABC-MCMC"
) {
  if (!inherits(simulator, "simulator")) stop("simulator must be a simulator.")
  bnd <- .validate_bounds(bounds)
  pars <- rownames(bnd)

  if (burnin >= n_samples) stop("burnin must be smaller than n_samples.")

  pre_process <- function(daily_cases, ...) {
    list(
      observed = as.numeric(daily_cases),
      known = .make_known(list(...), known_names)
    )
  }

  run <- function(x, ...) {
    observed <- x$observed
    known <- x$known
    prop_sd <- prop_frac * (bnd$upper - bnd$lower)
    use_summary <- identical(distance, "summary")

    n_model_runs <- 0L

    simulate_one <- function(theta) {
      n_model_runs <<- n_model_runs + 1L
      names(theta) <- pars
      .safe_simulate(simulator, .merge_candidate(known, theta), seed = sample.int(.Machine$integer.max, 1L))
    }

    if (use_summary) {
      s_obs <- summary(observed)
      S <- t(apply(.runif_bounds(n_pilot, bnd), 1, function(th) {
        pred <- simulate_one(th)
        if (is.null(pred)) rep(NA_real_, length(s_obs)) else summary(pred)
      }))
      s_scale <- .summary_scale(S)
      pilot_d <- apply(S, 1, function(s) .summary_distance(s_obs, s, s_scale))
      eps <- max(as.numeric(stats::quantile(pilot_d[is.finite(pilot_d)], pilot_quantile)), 1e-8)
    } else {
      eps <- max(eps_frac * sqrt(mean(observed^2)), 1e-8)
    }

    dist_one <- function(theta) {
      pred <- simulate_one(theta)
      if (is.null(pred)) return(Inf)
      if (use_summary) .summary_distance(s_obs, summary(pred), s_scale) else .metric_value(observed, pred, distance)
    }

    kern <- function(d) {
      if (!is.finite(d)) return(0)
      exp(-(d^2) / (2 * eps^2))
    }

    cur <- .bounds_midpoint(bnd)
    cur_d <- dist_one(cur)
    cur_k <- kern(cur_d)

    chain <- matrix(
      NA_real_,
      nrow = n_samples,
      ncol = length(pars),
      dimnames = list(NULL, pars)
    )
    dists <- numeric(n_samples)
    n_acc <- 0L

    for (it in seq_len(n_samples)) {
      prop <- cur + stats::rnorm(length(pars), sd = prop_sd)

      if (.in_bounds(prop, bnd)) {
        names(prop) <- pars
        pd <- dist_one(prop)
        pk <- kern(pd)

        accept <- if (cur_k <= 0 && pk <= 0) {
          pd < cur_d
        } else {
          stats::runif(1) <
            min(1, pk / max(cur_k, .Machine$double.xmin))
        }

        if (accept) {
          cur <- prop
          cur_d <- pd
          cur_k <- pk
          n_acc <- n_acc + 1L
        }
      }

      chain[it, ] <- cur
      dists[it] <- cur_d
    }

    posterior <- chain[(burnin + 1L):n_samples, , drop = FALSE]
    estimate <- apply(posterior, 2, stats::median)

    if (is.function(derive)) {
      estimate <- c(estimate, derive(estimate, known))
    }

    intervals <- t(apply(
      posterior,
      2,
      stats::quantile,
      probs = c(0.025, 0.975),
      names = FALSE
    ))
    colnames(intervals) <- c("lower_95", "upper_95")

    .new_calibration_result(
      estimate = estimate,
      method = name,
      posterior = posterior,
      intervals = intervals,
      diagnostics = list(
        accept_rate = n_acc / n_samples,
        epsilon = eps,
        n_unique_frac = nrow(unique(posterior)) / nrow(posterior),
        n_model_runs = n_model_runs
      )
    )
  }

  list(
    pre_process = pre_process,
    run = run,
    post_process = function(x, ...) x,
    type = type
  ) |>
    structure(class = "calibrator")
}

# Multivariate-normal helpers used by ABC-SMC. They avoid requiring mvtnorm.
.rmvn_one <- function(mu, chol_S) {
  as.numeric(mu + crossprod(chol_S, stats::rnorm(length(mu))))
}

.dmvn_one <- function(x, mu, S_inv, det_S) {
  d <- length(x)
  dx <- as.numeric(x - mu)
  q <- as.numeric(t(dx) %*% S_inv %*% dx)
  exp(-0.5 * q) / sqrt((2 * pi)^d * det_S)
}

.weighted_cov <- function(x, w) {
  mu <- colSums(x * w)
  dv <- sweep(x, 2, mu)
  t(dv) %*% (dv * w)
}

# ABC-SMC calibrator ----------------------------------------------------------
# Uses a multivariate Gaussian perturbation kernel with covariance equal to
# twice the weighted covariance of the previous particle population.
# With distance = "summary" the curves are compared through epidemic_summary()
# (or the function `summary`), each statistic scaled by its spread over the
# first generation (drawn from the prior).
# Built by make_calibrator("ABC-SMC", ...); not exported on its own.
.abc_smc_calibrator <- function(
    simulator,
    bounds,
    known_names = character(),
    n_particles = 200L,
    n_generations = 6L,
    eps_quantile = 0.5,
    max_attempt_mult = 25L,
    sim_budget = 8000L,
    distance = "rmse",
    summary = epidemic_summary,
    derive = NULL,
    type = simulator$type,
    name = "ABC-SMC"
) {
  if (!inherits(simulator, "simulator")) stop("simulator must be a simulator.")
  bnd <- .validate_bounds(bounds)
  pars <- rownames(bnd)
  d <- length(pars)

  pre_process <- function(daily_cases, ...) {
    list(
      observed = as.numeric(daily_cases),
      known = .make_known(list(...), known_names)
    )
  }

  run <- function(x, ...) {
    observed <- x$observed
    known <- x$known
    n_model_runs <- 0L
    use_summary <- identical(distance, "summary")
    if (use_summary) s_obs <- summary(observed)

    simulate_one <- function(theta) {
      n_model_runs <<- n_model_runs + 1L
      names(theta) <- pars
      .safe_simulate(simulator, .merge_candidate(known, theta), seed = sample.int(.Machine$integer.max, 1L))
    }
    dist_one <- function(theta) {
      pred <- simulate_one(theta)
      if (is.null(pred)) return(Inf)
      if (use_summary) .summary_distance(s_obs, summary(pred), s_scale) else .metric_value(observed, pred, distance)
    }

    theta <- .runif_bounds(n_particles, bnd)

    if (use_summary) {
      S <- t(apply(theta, 1, function(th) {
        pred <- simulate_one(th)
        if (is.null(pred)) rep(NA_real_, length(s_obs)) else summary(pred)
      }))
      s_scale <- .summary_scale(S)
      dist <- apply(S, 1, function(s) .summary_distance(s_obs, s, s_scale))
    } else {
      dist <- apply(theta, 1, dist_one)
    }
    w <- rep(1 / n_particles, n_particles)
    eps <- as.numeric(stats::quantile(dist, eps_quantile, names = FALSE))

    trace <- data.frame(
      generation = 1L,
      epsilon = NA_real_,
      next_epsilon = eps,
      accept_rate = 1,
      n_sims = n_model_runs,
      ess = n_particles
    )

    ok <- TRUE
    generations_done <- 1L

    if (n_generations >= 2L) {
      for (g in 2:n_generations) {
        prev_theta <- theta
        prev_w <- w

        S <- 2 * .weighted_cov(prev_theta, prev_w)
        S <- S + diag(1e-10 * pmax(diag(S), 1), d)

        chol_S <- tryCatch(
          chol(S),
          error = function(e) diag(sqrt(pmax(diag(S), 1e-12)), d)
        )
        S_inv <- tryCatch(solve(S), error = function(e) NULL)
        det_S <- det(S)

        if (is.null(S_inv) || !is.finite(det_S) || det_S <= 0) {
          ok <- FALSE
          break
        }

        new_theta <- matrix(
          NA_real_,
          nrow = n_particles,
          ncol = d,
          dimnames = list(NULL, pars)
        )
        new_dist <- numeric(n_particles)

        i <- 1L
        attempts <- 0L
        max_attempts <- max_attempt_mult * n_particles
        runs_before <- n_model_runs

        while (
          i <= n_particles &&
          attempts < max_attempts &&
          n_model_runs < sim_budget
        ) {
          attempts <- attempts + 1L
          idx <- sample.int(n_particles, 1L, prob = prev_w)
          cand <- .rmvn_one(prev_theta[idx, ], chol_S)

          if (!.in_bounds(cand, bnd)) next

          dd <- dist_one(cand)
          if (is.finite(dd) && dd <= eps) {
            new_theta[i, ] <- cand
            new_dist[i] <- dd
            i <- i + 1L
          }
        }

        if (i <= n_particles) {
          ok <- FALSE
          break
        }

        dens <- vapply(seq_len(n_particles), function(k) {
          sum(vapply(seq_len(n_particles), function(j) {
            prev_w[j] * .dmvn_one(
              new_theta[k, ],
              prev_theta[j, ],
              S_inv,
              det_S
            )
          }, numeric(1)))
        }, numeric(1))

        w <- 1 / pmax(dens, .Machine$double.xmin)
        w <- w / sum(w)
        theta <- new_theta
        dist <- new_dist

        next_eps <- as.numeric(
          stats::quantile(dist, eps_quantile, names = FALSE)
        )
        ess <- 1 / sum(w^2)

        trace <- rbind(
          trace,
          data.frame(
            generation = g,
            epsilon = eps,
            next_epsilon = next_eps,
            accept_rate = n_particles / attempts,
            n_sims = n_model_runs - runs_before,
            ess = ess
          )
        )

        eps <- next_eps
        generations_done <- g
      }
    }

    if (!ok) {
      stop(
        "ABC-SMC exhausted its particle/attempt/simulation budget. ",
        "Increase sim_budget or max_attempt_mult."
      )
    }

    estimate <- vapply(
      seq_len(ncol(theta)),
      function(j) .weighted_quantile(theta[, j], w, 0.5),
      numeric(1)
    )
    names(estimate) <- pars

    if (is.function(derive)) {
      estimate <- c(estimate, derive(estimate, known))
    }

    intervals <- t(vapply(
      seq_len(ncol(theta)),
      function(j) .weighted_quantile(
        theta[, j],
        w,
        c(0.025, 0.975)
      ),
      numeric(2)
    ))
    rownames(intervals) <- pars
    colnames(intervals) <- c("lower_95", "upper_95")

    posterior <- list(
      particles = theta,
      weights = w
    )

    .new_calibration_result(
      estimate = estimate,
      method = name,
      posterior = posterior,
      intervals = intervals,
      diagnostics = list(
        generations = generations_done,
        final_epsilon = eps,
        ess = 1 / sum(w^2),
        n_model_runs = n_model_runs,
        trace = trace
      )
    )
  }

  list(
    pre_process = pre_process,
    run = run,
    post_process = function(x, ...) x,
    type = type
  ) |>
    structure(class = "calibrator")
}

.to_unconstrained <- function(theta, bnd) {
  p <- (theta - bnd$lower) / (bnd$upper - bnd$lower)
  p <- pmin(pmax(p, 1e-10), 1 - 1e-10)
  stats::qlogis(p)
}

.from_unconstrained <- function(z, bnd) {
  out <- bnd$lower + (bnd$upper - bnd$lower) * stats::plogis(z)
  stats::setNames(out, rownames(bnd))
}

# Nelder-Mead calibrator -------------------------------------------------------
# Optimizes on an unconstrained logit scale that maps back into the supplied
# parameter bounds. The objective is averaged over common random-number seeds.
# Built by make_calibrator("NelderMead", ...); not exported on its own.
.nelder_mead_calibrator <- function(
    simulator,
    bounds,
    known_names = character(),
    n_eval_seeds = 3L,
    eval_seed = 12345L,
    maxit = 1000L,
    reltol = 1e-7,
    restarts = 2L,
    distance = "mse",
    derive = NULL,
    type = simulator$type,
    name = "NelderMead"
) {
  if (!inherits(simulator, "simulator")) stop("simulator must be a simulator.")
  bnd <- .validate_bounds(bounds)
  pars <- rownames(bnd)

  pre_process <- function(daily_cases, ...) {
    list(
      observed = as.numeric(daily_cases),
      known = .make_known(list(...), known_names)
    )
  }

  run <- function(x, ...) {
    seeds <- as.integer(eval_seed + seq_len(n_eval_seeds) - 1L)
    n_evals <- 0L

    objective <- function(theta) {
      n_evals <<- n_evals + 1L
      names(theta) <- pars
      .evaluate_theta(
        theta = theta,
        observed = x$observed,
        known = x$known,
        simulator = simulator,
        seeds = seeds,
        metric = distance
      )
    }

    theta0 <- .bounds_midpoint(bnd)
    # The midpoint is 0 on the logit scale; rounding can leave values like
    # 1e-16 there, and optim() sizes its first simplex relative to the start
    # (10% of it), which would make the simplex vanish and the search stop at
    # once. Start at exactly 0, with parscale 10 so the first simplex steps are
    # 1 on the logit scale (about a quarter of each parameter's range).
    z0 <- .to_unconstrained(theta0, bnd)
    z0[abs(z0) < 1e-8] <- 0

    fn_z <- function(z) objective(.from_unconstrained(z, bnd))

    fit <- stats::optim(
      par = z0,
      fn = fn_z,
      method = "Nelder-Mead",
      control = list(maxit = maxit, reltol = reltol, parscale = rep(10, length(z0)))
    )

    if (restarts > 0L) {
      for (k in seq_len(restarts)) {
        fit <- stats::optim(
          par = fit$par,
          fn = fn_z,
          method = "Nelder-Mead",
          control = list(maxit = maxit, reltol = reltol)
        )
      }
    }

    estimate <- .from_unconstrained(fit$par, bnd)
    if (is.function(derive)) {
      estimate <- c(estimate, derive(estimate, x$known))
    }

    .new_calibration_result(
      estimate = estimate,
      method = name,
      diagnostics = list(
        objective = fit$value,
        convergence = fit$convergence,
        n_evals = n_evals,
        n_model_runs = n_evals * n_eval_seeds
      )
    )
  }

  list(
    pre_process = pre_process,
    run = run,
    post_process = function(x, ...) x,
    type = type
  ) |>
    structure(class = "calibrator")
}

# Differential Evolution calibrator -------------------------------------------
# Uses `DEoptimR::JDEoptim()` over the same box bounds and the same objective
# definition as the Nelder-Mead adapter.
# Built by make_calibrator("DE", ...); not exported on its own.
.de_calibrator <- function(
    simulator,
    bounds,
    known_names = character(),
    n_eval_seeds = 3L,
    eval_seed = 12345L,
    NP = NULL,
    maxiter = 120L,
    tol = 1e-8,
    distance = "mse",
    derive = NULL,
    type = simulator$type,
    name = "DE"
) {
  if (!inherits(simulator, "simulator")) stop("simulator must be a simulator.")
  if (!requireNamespace("DEoptimR", quietly = TRUE)) {
    stop(
      "The DE baseline requires the optional package 'DEoptimR'. ",
      "Install it with install.packages('DEoptimR')."
    )
  }

  bnd <- .validate_bounds(bounds)
  pars <- rownames(bnd)
  if (is.null(NP)) NP <- 10L * length(pars)

  pre_process <- function(daily_cases, ...) {
    list(
      observed = as.numeric(daily_cases),
      known = .make_known(list(...), known_names)
    )
  }

  run <- function(x, ...) {
    seeds <- as.integer(eval_seed + seq_len(n_eval_seeds) - 1L)
    n_evals <- 0L

    objective <- function(theta) {
      n_evals <<- n_evals + 1L
      names(theta) <- pars
      .evaluate_theta(
        theta = theta,
        observed = x$observed,
        known = x$known,
        simulator = simulator,
        seeds = seeds,
        metric = distance
      )
    }

    fit <- DEoptimR::JDEoptim(
      lower = bnd$lower,
      upper = bnd$upper,
      fn = objective,
      NP = as.integer(NP),
      maxiter = as.integer(maxiter),
      tol = tol
    )

    estimate <- stats::setNames(as.numeric(fit$par), pars)
    if (is.function(derive)) {
      estimate <- c(estimate, derive(estimate, x$known))
    }

    .new_calibration_result(
      estimate = estimate,
      method = name,
      diagnostics = list(
        objective = fit$value,
        convergence = fit$convergence,
        n_evals = n_evals,
        n_model_runs = n_evals * n_eval_seeds
      )
    )
  }

  list(
    pre_process = pre_process,
    run = run,
    post_process = function(x, ...) x,
    type = type
  ) |>
    structure(class = "calibrator")
}

#' Create the standard comparison baselines
#'
#' Returns ABC-MCMC, ABC-SMC, Nelder-Mead, and Differential Evolution
#' calibrators. Add the user's primary method (e.g. BiLSTM or Transformer) to
#' this list before calling `compare_calibrators()`.
#'
#' @export
baseline_calibrators <- function(
    simulator,
    bounds,
    known_names = character(),
    derive = NULL,
    type = simulator$type,
    abc_mcmc_args = list(),
    abc_smc_args = list(),
    nm_args = list(),
    de_args = list()
) {
  make <- function(method, extra) {
    do.call(
      make_calibrator,
      c(
        list(
          method = method,
          simulator = simulator,
          bounds = bounds,
          known_names = known_names,
          derive = derive,
          type = type
        ),
        extra
      )
    )
  }

  list(
    `ABC-MCMC` = make("ABC-MCMC", abc_mcmc_args),
    `ABC-SMC` = make("ABC-SMC", abc_smc_args),
    NelderMead = make("NelderMead", nm_args),
    DE = make("DE", de_args)
  )
}


# Generic comparison utilities ------------------------------------------------

#' Compare calibration methods on the same observation
#'
#' Every method is run on the same observed series and known parameters.
#' Optional truth gives parameter-recovery metrics. Optional simulator gives a
#' reconstruction RMSE using the calibrated point estimate.
#'
#' @param observed Numeric observed series.
#' @param calibrators Named list of `calibrator` objects.
#' @param known Named list of known parameters passed to every calibrator.
#' @param truth Optional named vector of true parameters.
#' @param simulator Optional simulator used to reconstruct a fitted curve.
#' @param reconstruct Optional function(estimate, known, simulator, seed) that
#'   returns one reconstructed curve. Use this when the point estimate contains
#'   derived parameters or must be mapped before simulation.
#' @param n_reps Number of reconstruction replicates. The pointwise median curve
#'   is compared with the observation.
#' @param seed Reconstruction seed.
#' @return A `calibrator_comparison` object.
#' @export
compare_calibrators <- function(
    observed,
    calibrators,
    known = list(),
    truth = NULL,
    simulator = NULL,
    reconstruct = NULL,
    n_reps = 1L,
    seed = 123L
) {
  if (!is.list(calibrators) || !length(calibrators)) {
    stop("calibrators must be a non-empty named list.")
  }
  if (is.null(names(calibrators)) || any(names(calibrators) == "")) {
    stop("calibrators must be a named list.")
  }
  if (!is.list(known) || (length(known) && is.null(names(known)))) {
    stop("known must be a named list.")
  }

  observed <- as.numeric(observed)
  n_reps <- as.integer(n_reps)
  if (n_reps < 1L) stop("n_reps must be >= 1.")

  results <- list()
  estimate_rows <- list()
  summary_rows <- list()
  curves <- list()

  default_reconstruct <- function(estimate, known, simulator, seed) {
    p <- c(known, as.list(estimate))
    simulator$run(parameters = p, seed = seed)
  }

  if (is.null(reconstruct)) reconstruct <- default_reconstruct

  for (method in names(calibrators)) {
    cal <- calibrators[[method]]
    if (!inherits(cal, "calibrator")) {
      stop(method, " is not a calibrator object.")
    }

    t0 <- proc.time()
    ans <- do.call(
      calibrate,
      c(
        list(
          daily_cases = observed,
          calibrator = cal
        ),
        known
      )
    )
    tt <- proc.time() - t0
    elapsed <- unname(tt[["elapsed"]])

    est <- calibration_estimate(ans)
    results[[method]] <- ans

    common_truth <- character()
    param_mae <- NA_real_
    param_rmse <- NA_real_

    if (!is.null(truth)) {
      truth <- unlist(truth)
      common_truth <- intersect(names(est), names(truth))

      if (length(common_truth)) {
        err <- est[common_truth] - truth[common_truth]
        param_mae <- mean(abs(err))
        param_rmse <- sqrt(mean(err^2))

        estimate_rows[[method]] <- data.frame(
          method = method,
          parameter = common_truth,
          truth = as.numeric(truth[common_truth]),
          estimate = as.numeric(est[common_truth]),
          absolute_error = abs(as.numeric(err)),
          squared_error = as.numeric(err)^2,
          stringsAsFactors = FALSE
        )
      }
    }

    curve_rmse <- NA_real_
    curve_mae <- NA_real_

    if (!is.null(simulator)) {
      rep_curves <- lapply(seq_len(n_reps), function(r) {
        tryCatch(
          as.numeric(
            reconstruct(
              estimate = est,
              known = known,
              simulator = simulator,
              seed = seed + r - 1L
            )
          ),
          error = function(e) NULL
        )
      })

      ok <- vapply(
        rep_curves,
        function(x) !is.null(x) && length(x) == length(observed),
        logical(1)
      )

      if (any(ok)) {
        mat <- do.call(rbind, rep_curves[ok])
        med <- apply(mat, 2, stats::median)
        curve_rmse <- sqrt(mean((med - observed)^2))
        curve_mae <- mean(abs(med - observed))
        curves[[method]] <- med
      }
    }

    diag <- if (inherits(ans, "calibration_result")) ans$diagnostics else list()

    summary_rows[[method]] <- data.frame(
      method = method,
      wall_sec = elapsed,
      parameter_mae = param_mae,
      parameter_rmse = param_rmse,
      curve_rmse = curve_rmse,
      curve_mae = curve_mae,
      n_evals = if (!is.null(diag$n_evals)) diag$n_evals else NA_real_,
      n_model_runs = if (!is.null(diag$n_model_runs)) {
        diag$n_model_runs
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }

  estimates <- if (length(estimate_rows)) {
    do.call(rbind, estimate_rows)
  } else {
    data.frame()
  }

  summary <- do.call(rbind, summary_rows)
  rownames(summary) <- NULL

  curve_df <- NULL
  if (length(curves)) {
    curve_df <- data.frame(
      time = seq_along(observed),
      observed = observed,
      check.names = FALSE
    )
    for (method in names(curves)) {
      curve_df[[method]] <- curves[[method]]
    }
  }

  structure(
    list(
      summary = summary,
      estimates = estimates,
      curves = curve_df,
      results = results
    ),
    class = "calibrator_comparison"
  )
}

#' @export
print.calibrator_comparison <- function(x, ...) {
  cat("Calibrator comparison\n")
  print(x$summary, row.names = FALSE)
  invisible(x)
}
