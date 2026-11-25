test_that("smooth_cases() smooths as documented", {
  y <- c(1, 3, 2, 6, 9, 7, 14, 18, 15, 25, 30, 22, 35, 41, 30, 44, 50, 38)
  m7 <- smooth_cases(y)
  expect_true(is.na(attr(m7, "bandwidth")))
  expect_equal(m7[10], mean(y[7:13]))
  expect_equal(m7[1], mean(y[1:4]))
  expect_equal(as.numeric(smooth_cases(y, "none")), y)
  expect_equal(attr(smooth_cases(y, 0.2), "bandwidth"), 0.2)
  expect_error(smooth_cases(y, "wavelet"))
})

test_that("split_curve() splits into parts of at least 15 days", {
  s <- split_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date)
  expect_s3_class(s, "curve_split")
  expect_true(all(s$parts$days >= 15))
  expect_equal(sum(s$parts$days), nrow(sars_hongkong_2003))
  expect_length(s$fitted, nrow(sars_hongkong_2003))
  grDevices::pdf(NULL); on.exit(grDevices::dev.off())
  expect_silent(plot(s, interventions = subset(interventions, outbreak == "Hong Kong SARS 2003")))
})

test_that("method names are matched in any spelling", {
  expect_equal(epiworldRcalibrate:::.method_name(c("ABC_SMC", "abc-mcmc", "NELDERMEAD", "Nelder-Mead", "de", "DeepIMC")),
               c("ABC-SMC", "ABC-MCMC", "NelderMead", "NelderMead", "DE", "DeepIMC"))
  expect_error(epiworldRcalibrate:::.method_name("MLE"), "Unknown method")
})
