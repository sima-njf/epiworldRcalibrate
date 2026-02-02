#' Utah COVID-19 epidemic trends
#'
#' Daily COVID-19 epidemic indicators for the state of Utah, obtained from the
#' Utah Department of Health COVID-19 dashboard.
#'
#' @docType data
#' @format A data frame with 365 rows and 5 variables:
#' \describe{
#'   \item{Date}{Date of the observation (Date).}
#'   \item{Daily.Cases}{Number of newly reported COVID-19 cases (numeric).}
#'   \item{Smoothed.3.Day.Moving.Average}{Smoothed daily cases using a 3-day moving average (numeric).}
#'   \item{X3.Day.Moving.Average}{Alternative 3-day moving average of daily cases (numeric).}
#'   \item{Status}{Indicator of reporting or epidemic status (character or factor).}
#' }
#'
#' @source
#' Utah Department of Health COVID-19 Dashboard:
#' \url{https://coronavirus-dashboard.utah.gov/}
#'
#' @keywords datasets
"utah_covid_data"


#' ABC calibration results for COVID-19 SIR model
#'
#' Results from Approximate Bayesian Computation (ABC) calibration of an
#' SIR network model fitted to Utah COVID-19 incidence data.
#'
#' @docType data
#' @format A named list with the following elements:
#' \describe{
#'   \item{contact_rate}{Posterior median of the contact rate.}
#'   \item{recovery_rate}{Posterior median of the recovery rate.}
#'   \item{transmission_prob}{Posterior median of the transmission probability.}
#'   \item{R0}{Basic reproduction number computed from posterior medians.}
#'   \item{contact_rate_ci}{95 percent credible interval for the contact rate.}
#'   \item{recovery_rate_ci}{95 percent credible interval for the recovery rate.}
#'   \item{transmission_prob_ci}{95 percent credible interval for the transmission probability.}
#'   \item{calibration_time_seconds}{Total runtime of the ABC calibration (seconds).}
#'   \item{n_samples}{Number of MCMC samples used in calibration.}
#'   \item{burnin}{Number of burn-in iterations discarded.}
#'   \item{epsilon}{ABC tolerance parameter.}
#'   \item{seed}{Random seed used for reproducibility.}
#'   \item{posterior_samples}{Matrix of post-burn-in accepted parameter samples.}
#'   \item{acceptance_rate}{Acceptance rate of the ABC-MCMC algorithm (percent).}
#' }
#'
#' @source
#' Generated internally using the script
#' \code{data-raw/process_covid_calibration.R}.
#'
#' @keywords datasets
"abc_calibration_params"
