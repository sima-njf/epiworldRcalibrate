test_that("the datasets load with dates and counts", {
  for (nm in c("covid_admissions_utah", "sars_hongkong_2003", "ebola_sierraleone_2014", "sars_toronto_2003",
               "mers_korea_2015", "smallpox_kosovo_1972", "flu_pennsylvania_2009", "measles_hagelloch_1861",
               "ebola_kikwit_1995")) {
    e <- new.env(); utils::data(list = nm, package = "epiworldRcalibrate", envir = e); d <- e[[nm]]
    expect_s3_class(d$date, "Date")
    expect_true(all(d[[2]] >= 0, na.rm = TRUE))
  }
  expect_equal(length(unique(covid_waves_2020$region)), 27)
  expect_true(all(covid_waves_2020$outbreak[covid_waves_2020$region == "Norway"] == "Norway COVID-19"))
  expect_s3_class(interventions$date, "Date")
  expect_true(all(interventions$kind %in% c("restriction", "relaxation", "event")))
  expect_true(all(c("COVID-19", "influenza", "SARS") %in% disease_parameters$disease))
})

test_that("known_values() follows the paper", {
  kv <- known_values("COVID-19", "SEIR")
  expect_equal(kv, list(n = 10000, recov = 1 / 4, incub = 5.2))
  expect_equal(known_values("covid-19", "SIR"), list(n = 10000, recov = 1 / 7.5))
  expect_equal(known_values("Ebola", "SEIR", n = 5000)$recov, 1 / 4)
  expect_error(known_values("plague"), "Unknown disease")
})
