test_that("the standard methods calibrate a window (small budgets)", {
  skip_on_cran()
  y <- sir_window()
  fits <- calibrate_window(y, model = "SIR", method = c("ABC_SMC", "ABC_MCMC", "DE", "NELDERMEAD"),
                           n = 5000, recov = 0.15,
                           settings = list(ABC_SMC = list(n_particles = 20L, n_generations = 2L),
                                           ABC_MCMC = list(n_samples = 60L, burnin = 30L),
                                           DE = list(itermax = 3), NELDERMEAD = list(maxit = 30)))
  expect_s3_class(fits, "window_fits")
  expect_equal(names(fits), c("ABC-SMC", "ABC-MCMC", "DE", "NelderMead"))
  for (f in fits) {
    expect_s3_class(f, "window_fit")
    expect_true(f$estimate[["R0"]] >= 0.8 && f$estimate[["R0"]] <= 5)
    expect_gt(f$diagnostics$model_runs, 0)
  }
  tab <- summary(fits)
  expect_equal(nrow(tab), 4)
  expect_true(all(tab$Reff <= tab$R0))
  b <- predict_cases(fits$DE, ahead = 5, n = 20)
  expect_equal(nrow(b), length(y) + 5)
  expect_error(calibrate_window(y, model = "SIR", method = "NELDERMEAD", n = 5000), "Missing known value")
})

test_that("calibrate_curve() works without splitting or smoothing, and with given split days", {
  skip_on_cran()
  y <- sir_window(60)
  one <- calibrate_curve(y, model = "SIR", method = "NELDERMEAD", split = FALSE, smoothing = FALSE,
                         n = 5000, recov = 0.15, settings = list(maxit = 20))
  expect_equal(nrow(one$parts), 1)
  expect_equal(one$parts$bandwidth, 0)
  two <- calibrate_curve(y, model = "SIR", method = "NelderMead", split = 31, smoothing = TRUE,
                         known = list(n = 5000, recov = 0.15), settings = list(maxit = 20))
  expect_equal(two$parts$start, c(1, 31))
  run <- recover_curve(two, n = 20, run_on = TRUE)
  expect_true(all(c("fit", "run on") %in% run$what))
  grDevices::pdf(NULL); on.exit(grDevices::dev.off())
  expect_silent(plot(run, interventions = 31))
  eff <- intervention_effect(two, cut = 31, n = 20)
  expect_s3_class(eff, "intervention_effect")
  expect_equal(eff$summary$observed_after, sum(y[31:60]))
  expect_silent(plot(eff))
})

test_that("find_intervention() picks the breakpoint after a restriction where growth falls", {
  nz <- subset(covid_waves_2020, region == "New Zealand")
  s <- split_curve(nz$cases, nz$date, weekday = TRUE)
  m <- find_intervention(s, interventions, outbreak = "New Zealand COVID-19")
  expect_s3_class(m, "intervention_match")
  expect_equal(nrow(m), nrow(s$breakpoints))
  if (any(m$chosen)) {
    expect_true(m$lag_days[m$chosen] >= 0 && m$lag_days[m$chosen] <= 14)
    expect_lt(m$change[m$chosen], 0)
  }
})
