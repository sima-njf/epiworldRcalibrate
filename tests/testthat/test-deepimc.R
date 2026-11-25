test_that("DeepIMC calibrates SEIR and SIR windows", {
  skip_without_python()
  y <- sir_window()
  f <- calibrate_window(y, model = "SIR", method = "DeepIMC", n = 5000, recov = 0.15)
  expect_s3_class(f, "window_fit")
  expect_equal(f$method, "DeepIMC")
  expect_true(all(is.finite(f$intervals["R0", ])))
  g <- calibrate_window(y, model = "SEIR", known = known_values("COVID-19", "SEIR", n = 5000))
  expect_true(is.finite(g$estimate[["R0"]]))
  expect_error(calibrate_window(y[1:10], model = "SIR", n = 5000, recov = 0.15), "15 to 365")
})

test_that("calibrate_curve() with DeepIMC splits and smooths", {
  skip_without_python()
  fit <- calibrate_curve(flu_baltimore_1918$cases, model = "SEIR", known = known_values("influenza", "SEIR"))
  expect_true(all(is.finite(fit$parts$R0)))
  expect_equal(nrow(fit$parts), nrow(fit$split$parts))
})
