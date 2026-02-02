#' Utah COVID-19 epidemic trends
#'
#' Daily COVID-19 epidemic indicators for the state of Utah, obtained from the
#' Utah Department of Health COVID-19 dashboard.
#'
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
"utah_covid_data"
