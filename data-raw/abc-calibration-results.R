# data-raw/process_covid_calibration.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(epiworldR)
  library(epiworldRcalibrate)
  library(tictoc)
})

# Load COVID data ----------------------------------------------------------
get_covid_data <- function(n_days) {
  # Use packaged data
  covid_data <- utah_covid_data

  # Filter to last n_days
  last_date <- max(covid_data$Date, na.rm = TRUE)
  covid_data <- subset(covid_data, Date > (last_date - n_days)) |>
    dplyr::arrange(Date)

  stopifnot("Daily.Cases" %in% names(covid_data))
  covid_data
}

n_days <- 61
covid_data <- get_covid_data(n_days)
incidence <- covid_data$Daily.Cases
incidence_vec <- incidence
stopifnot(length(incidence) == n_days)

cat("Date range:", as.character(min(covid_data$Date)), "to",
    as.character(max(covid_data$Date)), "\n")
cat("Total cases observed:", sum(incidence), "\n")
cat("Mean daily cases:", round(mean(incidence), 1), "\n")

# Configuration ------------------------------------------------------------
N <- 20000     # Population size (BiLSTM)
recov <- 1 / 7  # Recovery rate (7-day infectious period)
model_ndays <- 60  # Simulation days
model_seed <- 122

# Simulation Functions -----------------------------------------------------
simulate_epidemic_calib <- function(params, ndays = model_ndays, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  sim_model <- ModelSIRCONN(
    name              = "sim",
    n                 = as.integer(params[1]),
    prevalence        = as.numeric(params[2]),
    contact_rate      = as.numeric(params[3]),
    transmission_rate = as.numeric(params[5]),
    recovery_rate     = as.numeric(params[4])
  )

  verbose_off(sim_model)
  run(sim_model, ndays = ndays)

  counts <- get_hist_total(sim_model)
  infected_counts <- counts[counts$state == "Infected", "counts"]

  return(infected_counts)
}

simulation_fun <- function(params, lfmcmc_obj, observed_data_info) {
  sim_params <- c(
    observed_data_info$n,
    observed_data_info$preval,
    params[1],  # contact_rate
    params[2],  # recovery_rate
    params[3]   # transmission_prob
  )

  infected_counts <- simulate_epidemic_calib(sim_params, ndays = model_ndays)
  return(as.numeric(infected_counts))
}

summary_fun <- function(data, lfmcmc_obj) {
  return(as.numeric(data))
}

proposal_fun <- function(old_params, lfmcmc_obj) {
  new_crate <- old_params[1] * exp(rnorm(1, sd = 0.1))
  new_recov <- plogis(qlogis(old_params[2]) + rnorm(1, sd = 0.1))
  new_ptran <- plogis(qlogis(old_params[3]) + rnorm(1, sd = 0.1))

  return(c(new_crate, new_recov, new_ptran))
}

kernel_fun <- function(simulated_stat, observed_stat, epsilon, lfmcmc_obj) {
  diff <- sum((simulated_stat - observed_stat)^2)
  return(exp(-diff / (2 * epsilon^2)))
}

# Run ABC Calibration ------------------------------------------------------
cat("\n=== ABC Calibration ===\n")

initial_infected <- incidence_vec[1]
initial_prevalence <- initial_infected / N

observed_data_info <- list(
  n = N,
  preval = initial_prevalence
)

dummy_model <- ModelSIRCONN(
  name              = "dummy",
  n                 = as.integer(N),
  prevalence        = 0.1,
  contact_rate      = 5.0,
  transmission_rate = 0.1,
  recovery_rate     = 0.1
)

local_simulation_fun <- function(params, lfmcmc_obj) {
  return(simulation_fun(params, lfmcmc_obj, observed_data_info))
}

lfmcmc_obj <- LFMCMC(dummy_model)
lfmcmc_obj <- set_simulation_fun(lfmcmc_obj, local_simulation_fun)
lfmcmc_obj <- set_summary_fun(lfmcmc_obj, summary_fun)
lfmcmc_obj <- set_proposal_fun(lfmcmc_obj, proposal_fun)
lfmcmc_obj <- set_kernel_fun(lfmcmc_obj, kernel_fun)
lfmcmc_obj <- set_observed_data(lfmcmc_obj, incidence_vec)

init_params <- c(5, 1/7, 0.0571)
n_samples_calib <- 3000
burnin <- 1500
epsilon <- sqrt(sum(incidence_vec^2)) * 0.05

cat("Running LFMCMC with", n_samples_calib, "samples (50% burn-in)...\n")

tic("ABC calibration")
run_lfmcmc(
  lfmcmc = lfmcmc_obj,
  params_init = init_params,
  n_samples = n_samples_calib,
  epsilon = epsilon,
  seed = model_seed
)
abc_time_output <- toc()

# Extract calibration time
abc_calibration_time <- abc_time_output$toc - abc_time_output$tic

# Process Results ----------------------------------------------------------
accepted <- get_all_accepted_params(lfmcmc_obj)
post_burnin <- tail(accepted, n = (n_samples_calib - burnin))

abc_median <- apply(post_burnin, 2, median)
abc_lower <- apply(post_burnin, 2, quantile, probs = 0.025)
abc_upper <- apply(post_burnin, 2, quantile, probs = 0.975)

abc_crate <- abc_median[1]
abc_recov <- abc_median[2]
abc_ptran <- abc_median[3]
abc_R0 <- (abc_crate * abc_ptran) / abc_recov

cat("\nCalibrated Parameters (Median with 95% CI):\n")
cat("Contact Rate:", round(abc_crate, 4),
    "[", round(abc_lower[1], 4), ",", round(abc_upper[1], 4), "]\n")
cat("Recovery Rate:", round(abc_recov, 4),
    "[", round(abc_lower[2], 4), ",", round(abc_upper[2], 4), "]\n")
cat("Transmission Prob:", round(abc_ptran, 4),
    "[", round(abc_lower[3], 4), ",", round(abc_upper[3], 4), "]\n")
cat("R0:", round(abc_R0, 4), "\n")

# Save calibration results -------------------------------------------------
abc_calibration_params <- list(
  contact_rate = abc_crate,
  recovery_rate = abc_recov,
  transmission_prob = abc_ptran,
  R0 = abc_R0,
  contact_rate_ci = c(lower = abc_lower[1], upper = abc_upper[1]),
  recovery_rate_ci = c(lower = abc_lower[2], upper = abc_upper[2]),
  transmission_prob_ci = c(lower = abc_lower[3], upper = abc_upper[3]),
  calibration_time_seconds = abc_calibration_time,
  n_samples = n_samples_calib,
  burnin = burnin,
  epsilon = epsilon,
  seed = model_seed,
  posterior_samples = post_burnin
)

# Save to data folder
usethis::use_data(abc_calibration_params, overwrite = TRUE)

cat("\n✓ Saved abc_calibration_params to data/abc_calibration_params.rda\n")
cat("  Calibration time:", round(abc_calibration_time, 2), "seconds\n")
