#' Utah COVID-19 epidemic trends
#'
#' Daily COVID-19 epidemic indicators for the state of Utah, obtained from the
#' Utah Department of Health COVID-19 dashboard.
#'
#' @format A data frame with 365 rows and 6 variables:
#' \describe{
#'   \item{Date}{Date of the observation (Date).}
#'   \item{Daily.Cases}{Number of newly reported COVID-19 cases (numeric).}
#'   \item{Daily.Deaths}{Number of newly reported COVID-19 deaths (numeric).}
#'   \item{Daily.Hospitalizations}{Number of new hospitalizations (numeric).}
#'   \item{Seven.Day.Avg.Cases}{Seven-day moving average of daily cases (numeric).}
#'   \item{Seven.Day.Avg.Deaths}{Seven-day moving average of daily deaths (numeric).}
#' }
#'
#' @source
#' Utah Department of Health COVID-19 Dashboard:
#' \url{https://coronavirus-dashboard.utah.gov/}
#'
#' @details
#' The dataset is extracted from the \emph{Trends\_Epidemic} file contained in the
#' Utah COVID-19 data ZIP archive. The version included in this package corresponds
#' to the most recent 365 days of available data at the time of preparation.
#'
"utah_covid_data"
