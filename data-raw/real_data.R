# Builds the real datasets shipped with the package (data/*.rda) from the raw
# files in examples/data, and the table of waves used in the report
# (examples/Real_data.Rmd). Run from the package root:
#   Rscript data-raw/real_data.R

DD <- "examples/data"

# Daily new COVID-19 cases per US state (Johns Hopkins University CSSE, county
# reports added up per state) and the state populations.
st <- read.csv(file.path(DD, "state_covid_daily_cases_jhu.csv"))
pop <- read.csv(file.path(DD, "state_population.csv"))
covid_us_states <- data.frame(state = st$state, date = as.Date(st$date), cases = as.numeric(st$daily_cases))
covid_us_states$population <- pop$population[match(covid_us_states$state, pop$state)]
covid_us_states <- covid_us_states[order(covid_us_states$state, covid_us_states$date), ]
rownames(covid_us_states) <- NULL

# Daily new COVID-19 cases in Utah (Utah Department of Health and Human Services).
cu <- read.csv(file.path(DD, "utah_covid_all_days.csv"))
covid_utah <- data.frame(date = as.Date(cu$Date), cases = as.numeric(cu$Daily.Cases))

# Daily hospital admissions in Utah (HHS): the file reports the admissions of
# the previous day, so the date is moved back one day.
hhs <- read.csv(file.path(DD, "utah_hhs_hospital_admissions.csv"))
covid_adm <- hhs$previous_day_admission_adult_covid_confirmed + hhs$previous_day_admission_pediatric_covid_confirmed
admissions_utah <- data.frame(date = as.Date(substr(hhs$date, 1, 10)) - 1,
                              influenza = as.numeric(hhs$previous_day_admission_influenza_confirmed),
                              covid = as.numeric(covid_adm))

# Daily influenza cases (onset of symptoms) in Baltimore, 1918, as in EpiEstim's Flu1918.
e1918 <- new.env(); load(file.path(DD, "Flu1918.rda"), envir = e1918)
flu_baltimore_1918 <- data.frame(day = seq_along(e1918$Flu1918$incidence), cases = as.numeric(e1918$Flu1918$incidence))

# The waves of the report -----------------------------------------------------------

# The part of a season with the peak: between the last week without cases
# before the peak and the first one after it.
season_range <- function(x, y, from, to) {
  s <- which(x >= as.Date(from) & x <= as.Date(to)); z <- y[s]
  ma <- as.numeric(stats::filter(z, rep(1 / 7, 7), sides = 2)); pk <- which.max(ma)
  zero_week <- vapply(seq_along(z), function(i) i >= 7 && all(z[(i - 6):i] == 0), logical(1))
  first <- max(c(0, which(zero_week & seq_along(z) < pk))) + 1
  after <- which(zero_week & seq_along(z) > pk)
  last <- if (length(after)) after[1] - 7 else length(z)
  x[s][c(first, last)]
}
wave <- function(group, name, dataset, region, column, from, to, n, disease, per_10000 = FALSE) {
  data.frame(wave = name, group = group, dataset = dataset, region = region, column = column,
             start = as.Date(from), end = as.Date(to), n = n,
             recov = if (disease == "COVID-19") 1 / 7 else 0.25, incub = if (disease == "COVID-19") 5 else 2,
             disease = disease, per_10000 = per_10000)
}
STATES_USED <- c("California", "Colorado", "Delaware", "Maryland", "New Jersey", "New York", "North Dakota",
                 "Pennsylvania", "Tennessee", "Texas")
STATE_WAVES <- list(`fall 2020` = c("2020-09-01", "2021-02-28"), `Delta 2021` = c("2021-06-15", "2021-11-30"),
                    `Omicron 2021-22` = c("2021-12-01", "2022-03-31"))
rows <- list()
for (s in STATES_USED) for (w in names(STATE_WAVES))
  rows[[length(rows) + 1]] <- wave("US states", paste(s, w), "covid_us_states", s, "cases", STATE_WAVES[[w]][1],
                                   STATE_WAVES[[w]][2], 10000, "COVID-19", per_10000 = TRUE)
r22 <- season_range(admissions_utah$date, admissions_utah$influenza, "2022-09-01", "2023-05-31")
r23 <- season_range(admissions_utah$date, admissions_utah$influenza, "2023-09-01", "2024-04-26")
r_s <- season_range(covid_utah$date, covid_utah$cases, "2023-06-01", "2023-10-15")
last365 <- tail(covid_utah$date, 365)
rows <- c(rows, list(
  wave("Utah and Baltimore", "Utah COVID-19 2024-25", "covid_utah", "Utah", "cases", last365[1], last365[365], 5000, "COVID-19"),
  wave("Utah and Baltimore", "Utah flu 2022-23", "admissions_utah", "Utah", "influenza", r22[1], r22[2], 5000, "influenza"),
  wave("Utah and Baltimore", "Utah flu 2023-24", "admissions_utah", "Utah", "influenza", r23[1], r23[2], 5000, "influenza"),
  wave("Utah and Baltimore", "Baltimore flu 1918", "flu_baltimore_1918", "Baltimore", "cases", NA, NA, 10000, "influenza"),
  wave("Utah and Baltimore", "Utah COVID-19 summer 2023", "covid_utah", "Utah", "cases", r_s[1], r_s[2], 10000, "COVID-19")))
real_data_waves <- do.call(rbind, rows)

save(covid_us_states, file = "data/covid_us_states.rda", compress = "xz", version = 2)
save(covid_utah, file = "data/covid_utah.rda", compress = "xz", version = 2)
save(admissions_utah, file = "data/admissions_utah.rda", compress = "xz", version = 2)
save(flu_baltimore_1918, file = "data/flu_baltimore_1918.rda", compress = "xz", version = 2)
save(real_data_waves, file = "data/real_data_waves.rda", compress = "xz", version = 2)
