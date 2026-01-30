#' Utah COVID-19 Data
#'
#' A dataset containing COVID-19 trends and epidemic data for Utah,
#' downloaded from the Utah Coronavirus Dashboard.
#'
#' @format A data frame with the following columns:
#' \describe{
#'   \item{Date}{Date of observation (Date class)}
#'   \item{Daily.Cases}{Number of daily COVID-19 cases (numeric)}
#'   \item{...}{Additional columns from the Utah COVID-19 trends data}
#' }
#'
#' @source Utah Coronavirus Dashboard
#'   \url{https://coronavirus-dashboard.utah.gov/}
#'
#' @details
#' This dataset is extracted from the "Trends_Epidemic" file within the
#' Utah COVID-19 data ZIP archive. The data includes the most recent
#' period of observations as specified during data preparation.
#'
#' The data is processed to:
#' \itemize{
#'   \item Convert dates to Date class
#'   \item Filter to recent observations
#'   \item Arrange chronologically by date
#' }
#'
#' @examples
#' \dontrun{
#' # Load the data
#' data(utah_covid_data)
#'
#' # View structure
#' str(utah_covid_data)
#'
#' # Plot daily cases
#' plot(utah_covid_data$Date, utah_covid_data$Daily.Cases, type = "l")
#' }
"utah_covid_data"

#' ABC Calibration Parameters for COVID-19 SIR Model
#'
#' Results from ABC calibration of a SIR-CONN epidemic model fitted to
#' Utah COVID-19 data using LFMCMC methods.
#'
#' @format A list with 13 elements:
#' \describe{
#'   \item{contact_rate}{Median calibrated contact rate}
#'   \item{recovery_rate}{Median calibrated recovery rate}
#'   \item{transmission_prob}{Median calibrated transmission probability}
#'   \item{R0}{Basic reproduction number}
#'   \item{contact_rate_ci}{95% credible interval for contact rate}
#'   \item{recovery_rate_ci}{95% credible interval for recovery rate}
#'   \item{transmission_prob_ci}{95% credible interval for transmission probability}
#'   \item{calibration_time_seconds}{Calibration time in seconds}
#'   \item{n_samples}{Total MCMC samples (2000)}
#'   \item{burnin}{Burn-in samples (1000)}
#'   \item{epsilon}{ABC tolerance parameter}
#'   \item{seed}{Random seed (122)}
#'   \item{posterior_samples}{Matrix of posterior samples (1000 × 3)}
#' }
#'
#' @details
#' Calibrated using last 61 days of Utah COVID-19 data with population N=30,000.
#' Uses exponential kernel with epsilon = 5% of observed incidence L2 norm.
#'
#' @examples
#' data(abc_calibration_params)
#' abc_calibration_params$contact_rate
#' abc_calibration_params$R0
#'
#' @source Calibrated from \code{\link{utah_covid_data}}
"abc_calibration_params"
