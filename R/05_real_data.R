# =============================================================================
# 05_real_data.R
# Real epidemic data shipped with the package, and windows cut from them
#
# This file contains:
#   - the documentation of the datasets: covid_us_states, covid_utah,
#     admissions_utah, flu_baltimore_1918 and real_data_waves
#   - wave_data(): the daily counts of one wave of real_data_waves
#   - real_data_windows(): windows from the beginning, middle and end of the
#     waves, ready for calibrate() and compare_methods()
#
# The datasets are built by data-raw/real_data.R and data-raw/paper_data.R.
# =============================================================================

#' Daily counts of outbreaks used in the paper
#'
#' Daily new cases (by date of symptom onset) or hospital admissions of the
#' outbreaks of the paper's real-data analysis. Each is a data frame with
#' `date` and `cases` (`covid_admissions_utah`: `date` and `admissions`). The
#' name in [interventions]`$outbreak` is given for each, so that the dated
#' measures of an outbreak can be found with
#' `subset(interventions, outbreak == "Hong Kong SARS 2003")`.
#'
#' \describe{
#'   \item{covid_admissions_utah}{COVID-19 hospital admissions in Utah by day of
#'     admission, 6 March 2020 to 2025 (Utah Department of Health and Human
#'     Services). The paper uses four waves: spring 2020 (6 March to 30 June),
#'     fall-winter 2020 (1 September 2020 to 28 February 2021), Delta 2021 (15
#'     June to 30 November) and Omicron 2021-22 (1 December 2021 to 31 March
#'     2022). Outbreak: `"Utah COVID-19"`.}
#'   \item{sars_hongkong_2003}{SARS in Hong Kong, 2003, 107 days (Donnelly et
#'     al. 2003). Outbreak: `"Hong Kong SARS 2003"`.}
#'   \item{ebola_sierraleone_2014}{Ebola in Sierra Leone, 2014-15, 483 days (WHO
#'     Ebola Response Team). Outbreak: `"Sierra Leone Ebola 2014-15"`.}
#'   \item{sars_toronto_2003}{SARS in Toronto, 2003, 110 days (R package
#'     `outbreaks`, `sars_canada_2003`, all four sources of infection added).
#'     Outbreak: `"Toronto SARS 2003"`.}
#'   \item{mers_korea_2015}{MERS in South Korea, 2015, 36 days (`outbreaks`,
#'     `mers_korea_2015`, cases by date of onset). Outbreak: `"South Korea MERS
#'     2015"`.}
#'   \item{smallpox_kosovo_1972}{Smallpox in Kosovo, 1972, 57 days (R package
#'     `EpiEstim`, `Smallpox1972`). Outbreak: `"Kosovo smallpox 1972"`.}
#'   \item{flu_pennsylvania_2009}{Pandemic influenza in a Pennsylvania school,
#'     2009, 32 days (`EpiEstim`, `Flu2009`). Outbreak: `"Pennsylvania school
#'     influenza 2009"`.}
#'   \item{measles_hagelloch_1861}{Measles in Hagelloch, Germany, 1861, 87 days
#'     (`outbreaks`, `measles_hagelloch_1861`, by date of prodrome). No dated
#'     measures.}
#'   \item{ebola_kikwit_1995}{Ebola in Kikwit, Democratic Republic of the Congo,
#'     1995, 192 days (`outbreaks`, `ebola_kikwit_1995`). Outbreak: `"Kikwit
#'     Ebola 1995"`.}
#' }
#' The paper also uses [flu_baltimore_1918] (outbreak `"Baltimore influenza
#' 1918"`), the influenza admissions of [admissions_utah], the Utah cases of
#' [covid_utah] and the COVID-19 waves of [covid_waves_2020].
#'
#' @format Data frames with `date` (class `Date`) and `cases` (integer).
#' @source See each dataset above; built by `data-raw/paper_data.R`. Donnelly CA
#'   et al. (2003), Epidemiological determinants of spread of causal agent of
#'   severe acute respiratory syndrome in Hong Kong, The Lancet 361, 1761-1766.
#'   Jombart T et al. (2020), outbreaks: a compilation of disease outbreak data
#'   (R package). Cori A et al. (2013), American Journal of Epidemiology 178,
#'   1505-1512 (EpiEstim).
#' @seealso [interventions], [known_values()], [calibrate_curve()]
#' @name outbreak_data
#' @aliases covid_admissions_utah sars_hongkong_2003 ebola_sierraleone_2014 sars_toronto_2003 mers_korea_2015
#'   smallpox_kosovo_1972 flu_pennsylvania_2009 measles_hagelloch_1861 ebola_kikwit_1995
#' @examples
#' data(sars_hongkong_2003)
#' plot(sars_hongkong_2003$date, sars_hongkong_2003$cases, type = "h")
NULL

#' COVID-19 waves of 2020 with dated restrictions
#'
#' Daily COVID-19 cases (by date of report) of 27 waves of 2020 in countries,
#' Australian states, Canadian provinces and Hong Kong, chosen because their
#' restrictions are dated in the Oxford COVID-19 Government Response Tracker
#' (OxCGRT); the restrictions are in [interventions]. Negative corrections are
#' set to 0.
#'
#' @format A data frame with 2,371 rows:
#' \describe{
#'   \item{region}{Country, state, province or territory.}
#'   \item{wave}{For example `"first wave 2020"`.}
#'   \item{outbreak}{The name in [interventions]`$outbreak`, for example `"New
#'     Zealand COVID-19"`.}
#'   \item{date}{Date of report.}
#'   \item{cases}{Daily new cases.}
#' }
#' @source COVID-19 Data Repository by the Center for Systems Science and
#'   Engineering (CSSE) at Johns Hopkins University,
#'   \url{https://github.com/CSSEGISandData/COVID-19} (CC BY 4.0).
#' @seealso [interventions], [find_intervention()], [intervention_effect()]
#' @examples
#' data(covid_waves_2020)
#' unique(covid_waves_2020$region)
#' nz <- subset(covid_waves_2020, region == "New Zealand")
#' plot(nz$date, nz$cases, type = "h")
"covid_waves_2020"

#' Dated measures of the outbreaks
#'
#' Restrictions and relaxations with their dates, for the outbreaks in
#' [outbreak_data], [flu_baltimore_1918], Utah COVID-19 and the waves of
#' [covid_waves_2020]. The measures of the COVID-19 waves are the days, from 30
#' days before the wave to its end, on which OxCGRT raises school closing to
#' some or all schools, workplace closing to some or all non-essential
#' workplaces, restrictions on gatherings to at most 100 or 10 people, or a
#' stay-at-home requirement; the measures of the same day are joined. The other
#' measures come from news reports and the literature (the `source` of each).
#'
#' @format A data frame with 117 rows:
#' \describe{
#'   \item{outbreak}{The outbreak, for example `"Utah COVID-19"`, `"Hong Kong SARS
#'     2003"`, `"New Zealand COVID-19"`.}
#'   \item{date}{Date of the measure.}
#'   \item{measure}{What was done.}
#'   \item{kind}{`"restriction"`, `"relaxation"` or `"event"` (for example the
#'     first case diagnosed). [find_intervention()] uses the restrictions.}
#'   \item{scope}{`"national"`, `"state"`, `"county"`, ...}
#'   \item{source}{A link or reference.}
#' }
#' @source Hale T et al. (2021), A global panel database of pandemic policies
#'   (Oxford COVID-19 Government Response Tracker), Nature Human Behaviour 5,
#'   529-538, \url{https://github.com/OxCGRT/covid-policy-dataset} (CC BY 4.0);
#'   and the sources in the `source` column.
#' @seealso [find_intervention()], [intervention_effect()]
#' @examples
#' data(interventions)
#' subset(interventions, outbreak == "Utah COVID-19")
"interventions"

#' Incubation period and serial interval of the diseases of the paper
#'
#' Used by [known_values()] to give the known values of the models.
#'
#' @format A data frame with one row per disease:
#' \describe{
#'   \item{disease}{`"COVID-19"`, `"influenza"`, `"SARS"`, `"Ebola"`, `"MERS"`,
#'     `"smallpox"`, `"measles"`.}
#'   \item{incubation}{Mean incubation period (days).}
#'   \item{serial_interval, serial_interval_sd}{Mean and standard deviation of
#'     the serial interval (days).}
#'   \item{source}{References.}
#' }
#' @source See the `source` column.
#' @examples
#' data(disease_parameters)
#' disease_parameters[, 1:4]
"disease_parameters"

#' Daily COVID-19 cases in US states
#'
#' Daily new COVID-19 cases in the 50 US states and Washington, DC, from 22
#' January 2020 to 9 March 2023: the daily differences of the cumulative county
#' counts added up per state (negative corrections set to 0).
#'
#' @format A data frame with 58,293 rows:
#' \describe{
#'   \item{state}{State name.}
#'   \item{date}{Date of report.}
#'   \item{cases}{Daily new cases.}
#'   \item{population}{State population (sum of the county populations in the
#'     source).}
#' }
#' @source COVID-19 Data Repository by the Center for Systems Science and
#'   Engineering (CSSE) at Johns Hopkins University,
#'   \url{https://github.com/CSSEGISandData/COVID-19} (CC BY 4.0). Dong E, Du H,
#'   Gardner L (2020). An interactive web-based dashboard to track COVID-19 in
#'   real time. The Lancet Infectious Diseases 20(5), 533-534.
#' @seealso [real_data_waves], [real_data_windows()]
"covid_us_states"

#' Daily COVID-19 cases in Utah
#'
#' @format A data frame with 1,876 rows (18 March 2020 to 6 May 2025):
#' \describe{
#'   \item{date}{Date.}
#'   \item{cases}{Daily new cases.}
#' }
#' @source Utah Department of Health and Human Services,
#'   \url{https://coronavirus-dashboard.utah.gov/Utah_COVID19_data.zip}.
#' @seealso [real_data_waves], [real_data_windows()]
"covid_utah"

#' Daily hospital admissions for influenza and COVID-19 in Utah
#'
#' Confirmed influenza and COVID-19 (adult plus pediatric) hospital admissions
#' in Utah. The source reports the admissions of the previous day; `date` is
#' the day of admission.
#'
#' @format A data frame with 1,500 rows (13 March 2020 to 26 April 2024; one
#'   gap of 6 days in March 2020):
#' \describe{
#'   \item{date}{Day of admission.}
#'   \item{influenza}{Confirmed influenza admissions (`NA`: not reported).}
#'   \item{covid}{Confirmed COVID-19 admissions (`NA`: not reported).}
#' }
#' @source U.S. Department of Health and Human Services, COVID-19 Reported
#'   Patient Impact and Hospital Capacity by State Timeseries,
#'   \url{https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh}.
#' @seealso [real_data_waves], [real_data_windows()]
"admissions_utah"

#' Daily influenza cases in Baltimore, 1918
#'
#' Daily cases (onset of symptoms) of the 1918 influenza epidemic in Baltimore,
#' Maryland, over 92 days.
#'
#' @format A data frame with 92 rows:
#' \describe{
#'   \item{day}{Day of the epidemic.}
#'   \item{cases}{Daily new cases.}
#' }
#' @source Frost WH, Sydenstricker E (1919). Influenza in Maryland: preliminary
#'   statistics of certain localities. Public Health Reports 34(11), 491-504; as
#'   in the `Flu1918` data of the EpiEstim package (Cori A, Ferguson NM, Fraser
#'   C, Cauchemez S (2013). A new framework and software to estimate
#'   time-varying reproduction numbers during epidemics. American Journal of
#'   Epidemiology 178(9), 1505-1512).
#' @seealso [real_data_waves], [real_data_windows()]
"flu_baltimore_1918"

#' The real-data waves of the report
#'
#' The 35 waves used to compare the methods on real data: COVID-19 in 10 US
#' states that reported every day (fall 2020, Delta 2021 and Omicron 2021-22),
#' Utah COVID-19 cases (the last 365 days of the data and the summer 2023
#' wave), Utah influenza admissions (2022-23 and 2023-24 seasons) and Baltimore
#' 1918. A season is the part with the peak, between the last week without
#' cases before the peak and the first one after it.
#'
#' @format A data frame with one row per wave:
#' \describe{
#'   \item{wave}{Name of the wave.}
#'   \item{group}{`"US states"` or `"Utah and Baltimore"`.}
#'   \item{dataset, region, column}{Where the counts are: the dataset, the
#'     state (or `"Utah"`, `"Baltimore"`) and the column.}
#'   \item{start, end}{First and last day of the wave (`NA`: the whole
#'     series).}
#'   \item{n}{Population size given to the models.}
#'   \item{recov, incub}{Known recovery rate and incubation period (days):
#'     1/7 and 5 for COVID-19, 0.25 and 2 for influenza.}
#'   \item{disease}{`"COVID-19"` or `"influenza"`.}
#'   \item{per_10000}{`TRUE` when the counts are scaled to cases per 10,000
#'     people (US states, with `n` = 10,000).}
#' }
#' @seealso [wave_data()], [real_data_windows()]
"real_data_waves"

.dataset <- function(name) {
  e <- new.env()
  utils::data(list = name, package = "epiworldRcalibrate", envir = e)
  e[[name]]
}

# The full daily series of a wave's region (x: date or day, y: count, scaled
# per 10,000 people for US states) and which days make up the wave.
.wave_series <- function(row) {
  d <- .dataset(row$dataset)
  if (row$dataset == "covid_us_states") {
    d <- d[d$state == row$region, ]
    y <- round(d$cases * 10000 / d$population)
  } else {
    y <- d[[row$column]]
  }
  x <- if ("date" %in% names(d)) d$date else d$day
  idx <- if (is.na(row$start)) seq_along(x) else which(x >= row$start & x <= row$end)
  list(x = x, y = as.numeric(y), idx = idx)
}

.wave_row <- function(wave) {
  tab <- .dataset("real_data_waves")
  if (!all(wave %in% tab$wave)) {
    stop("Unknown wave(s): ", paste(setdiff(wave, tab$wave), collapse = ", "), ". See real_data_waves$wave.")
  }
  tab[match(wave, tab$wave), , drop = FALSE]
}

#' The daily counts of a real-data wave
#'
#' @param wave A name from `real_data_waves$wave`, for example `"New York
#'   Omicron 2021-22"`.
#' @return A data frame with `time` (date, or day for Baltimore 1918) and
#'   `count` (daily new cases or admissions; cases per 10,000 people for US
#'   states), with the known values `n`, `recov` and `incub` as attributes.
#' @examples
#' w <- wave_data("Utah flu 2022-23")
#' plot(w$time, w$count, type = "h")
#' @export
wave_data <- function(wave) {
  row <- .wave_row(wave[1])
  s <- .wave_series(row)
  structure(data.frame(time = s$x[s$idx], count = s$y[s$idx]), wave = row$wave, n = row$n, recov = row$recov,
            incub = row$incub)
}

#' Windows of real data for calibrate() and compare_methods()
#'
#' Cuts windows of the given lengths from the beginning, the middle (centered on
#' the peak of the 7-day average) and the end of each wave in
#' [real_data_waves], as in the report. With the defaults it returns the report's
#' 315 windows (270 from US states, 45 from Utah and Baltimore). Windows longer
#' than a wave are skipped.
#'
#' @param waves Names from `real_data_waves$wave`; `NULL` for all.
#' @param lengths Window lengths (days).
#' @param positions Any of `"beginning"`, `"middle"`, `"end"`.
#' @param ahead Days after the window kept as `future` (where the data
#'   continue).
#' @return A named list of windows, each with `cases`, `future`, the known
#'   values `n`, `recov` and `incub`, the labels `wave`, `group`, `disease` and
#'   `position`, and `time` (the dates of the window).
#' @examples
#' windows <- real_data_windows("New York Omicron 2021-22", lengths = 60)
#' w <- windows[[1]]
#' \dontrun{
#' fit <- calibrate(w$cases, window_calibrator("SEIR"), n = w$n, recov = w$recov, incub = w$incub)
#' results <- compare_methods(windows, "SEIR", cores = 3)
#' }
#' @export
real_data_windows <- function(waves = NULL, lengths = c(30L, 60L, 90L), positions = c("beginning", "middle", "end"),
                              ahead = 14L) {
  tab <- .dataset("real_data_waves")
  if (!is.null(waves)) tab <- .wave_row(waves)
  positions <- match.arg(positions, several.ok = TRUE)
  out <- list()
  for (i in seq_len(nrow(tab))) {
    row <- tab[i, ]
    s <- .wave_series(row)
    wy <- s$y[s$idx]
    pk <- which.max(as.numeric(stats::filter(wy, rep(1 / 7, 7), sides = 2)))
    for (L in lengths) {
      if (L > length(s$idx)) next
      for (pos in positions) {
        s0 <- switch(pos, beginning = 1, end = length(wy) - L + 1,
                     middle = min(max(1, pk - floor(L / 2)), length(wy) - L + 1))
        at <- s$idx[s0:(s0 + L - 1)]
        fut <- at[L] + seq_len(ahead)
        fut <- fut[fut <= length(s$y)]
        out[[sprintf("w%03d", length(out) + 1)]] <- list(
          cases = s$y[at], future = s$y[fut], n = row$n, recov = row$recov, incub = row$incub,
          wave = row$wave, group = row$group, disease = row$disease, position = pos, time = s$x[at])
      }
    }
  }
  out
}
