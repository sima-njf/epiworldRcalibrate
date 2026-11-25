# Builds the datasets of the paper's real-data analysis (data/*.rda):
#   covid_admissions_utah, sars_hongkong_2003, ebola_sierraleone_2014, sars_toronto_2003, mers_korea_2015,
#   smallpox_kosovo_1972, flu_pennsylvania_2009, measles_hagelloch_1861, ebola_kikwit_1995,
#   covid_waves_2020, interventions and disease_parameters.
#
# Run from the package root with the raw files in RAW (by default data-raw/raw):
#   utah_covid_admissions_dhhs.csv   Utah DHHS COVID-19 hospital admissions by day of admission
#                                    (https://coronavirus-dashboard.utah.gov/Utah_COVID19_data.zip)
#   sars_hongkong_2003.csv           daily SARS cases by onset in Hong Kong, 2003 (Donnelly et al. 2003, Lancet 361:1761)
#   ebola_sierraleone_2014.csv       daily Ebola cases by onset in Sierra Leone, 2014-15 (WHO Ebola Response Team)
#   known_interventions.csv          dated measures of the outbreaks, with a source for each
#   jhu_global/time_series_covid19_confirmed_global.csv
#                                    https://github.com/CSSEGISandData/COVID-19 (CC BY 4.0)
#   oxcgrt/OxCGRT_compact_national_v1.csv, oxcgrt/OxCGRT_compact_subnational_v1.csv
#                                    https://github.com/OxCGRT/covid-policy-dataset (CC BY 4.0)
# The six further outbreaks come from the R packages outbreaks and EpiEstim.
RAW <- Sys.getenv("RAW", "data-raw/raw")

date_cases <- function(date, cases) data.frame(date = as.Date(date), cases = as.integer(round(cases)))
daily <- function(dates) {
  dates <- as.Date(dates[!is.na(dates)]); d <- seq(min(dates), max(dates), by = "day")
  date_cases(d, table(factor(as.character(dates), levels = as.character(d))))
}

adm <- read.csv(file.path(RAW, "utah_covid_admissions_dhhs.csv"))
covid_admissions_utah <- data.frame(date = as.Date(adm$date), admissions = as.integer(adm$admissions))

sars <- read.csv(file.path(RAW, "sars_hongkong_2003.csv"))
sars_hongkong_2003 <- date_cases(as.Date("2003-02-15") + sars$day - 1, sars$cases)

ebo <- read.csv(file.path(RAW, "ebola_sierraleone_2014.csv"))
ebola_sierraleone_2014 <- date_cases(ebo$date, ebo$cases)

sc <- outbreaks::sars_canada_2003
sars_toronto_2003 <- date_cases(sc$date, sc$cases_travel + sc$cases_household + sc$cases_healthcare + sc$cases_other)
mers_korea_2015 <- daily(outbreaks::mers_korea_2015$linelist$dt_onset)
ep <- new.env(); utils::data("Smallpox1972", "Flu2009", package = "EpiEstim", envir = ep)
smallpox_kosovo_1972 <- date_cases(as.Date("1972-02-16") + seq_along(ep$Smallpox1972$incidence) - 1, ep$Smallpox1972$incidence)
flu_pennsylvania_2009 <- date_cases(ep$Flu2009$incidence$dates, ep$Flu2009$incidence$I)
measles_hagelloch_1861 <- daily(outbreaks::measles_hagelloch_1861$date_of_prodrome)
ek <- outbreaks::ebola_kikwit_1995
ebola_kikwit_1995 <- date_cases(ek$date, ek$onset)

# The 27 COVID-19 waves of 2020 with dated national (or state, or territory) measures: daily cases by report date.
WAVES <- read.csv(text = "region,country,province,ox_country,ox_region,from,to,wave
Luxembourg,Luxembourg,,Luxembourg,,2020-03-01,2020-05-31,first wave 2020
Czechia,Czechia,,Czech Republic,,2020-03-01,2020-05-31,first wave 2020
Croatia,Croatia,,Croatia,,2020-03-01,2020-05-31,first wave 2020
Norway,Norway,,Norway,,2020-02-26,2020-05-31,first wave 2020
Estonia,Estonia,,Estonia,,2020-03-01,2020-05-31,first wave 2020
Hungary,Hungary,,Hungary,,2020-03-04,2020-05-31,first wave 2020
Cyprus,Cyprus,,Cyprus,,2020-03-09,2020-05-31,first wave 2020
Iceland,Iceland,,Iceland,,2020-02-28,2020-05-31,first wave 2020
New Zealand,New Zealand,,New Zealand,,2020-03-01,2020-05-31,first wave 2020
New South Wales,Australia,New South Wales,Australia,New South Wales,2020-03-01,2020-05-31,first wave 2020
Hong Kong,China,Hong Kong,Hong Kong,,2020-07-01,2020-09-30,third wave 2020
Denmark,Denmark,,Denmark,,2020-03-01,2020-04-30,first wave 2020
Finland,Finland,,Finland,,2020-03-01,2020-05-31,first wave 2020
Greece,Greece,,Greece,,2020-02-26,2020-05-31,first wave 2020
Slovenia,Slovenia,,Slovenia,,2020-03-04,2020-05-31,first wave 2020
Malaysia,Malaysia,,Malaysia,,2020-03-01,2020-05-31,first wave 2020
Thailand,Thailand,,Thailand,,2020-03-01,2020-05-31,first wave 2020
Victoria,Australia,Victoria,Australia,Victoria,2020-06-01,2020-10-31,second wave 2020
South Korea,\"Korea, South\",,South Korea,,2020-02-15,2020-03-31,first wave 2020
Lithuania,Lithuania,,Lithuania,,2020-03-01,2020-05-31,first wave 2020
Slovakia,Slovakia,,Slovak Republic,,2020-03-06,2020-05-31,first wave 2020
Bulgaria,Bulgaria,,Bulgaria,,2020-03-08,2020-05-31,first wave 2020
British Columbia,Canada,British Columbia,Canada,British Columbia,2020-03-01,2020-05-31,first wave 2020
Alberta,Canada,Alberta,Canada,Alberta,2020-03-05,2020-05-31,first wave 2020
Austria,Austria,,Austria,,2020-03-01,2020-04-30,first wave 2020
Israel,Israel,,Israel,,2020-03-01,2020-04-15,first wave 2020
Japan,Japan,,Japan,,2020-03-01,2020-05-31,first wave 2020", stringsAsFactors = FALSE, na.strings = "")
WAVES$province[is.na(WAVES$province)] <- ""; WAVES$ox_region[is.na(WAVES$ox_region)] <- ""
g <- read.csv(file.path(RAW, "jhu_global", "time_series_covid19_confirmed_global.csv"), check.names = FALSE)
gx <- as.Date(names(g)[-(1:4)], format = "%m/%d/%y")
covid_waves_2020 <- do.call(rbind, lapply(seq_len(nrow(WAVES)), function(k) {
  w <- WAVES[k, ]
  cum <- as.numeric(g[g[[2]] == w$country & g[[1]] == w$province, -(1:4)][1, ])
  y <- pmax(c(cum[1], diff(cum)), 0); i <- which(gx >= as.Date(w$from) & gx <= as.Date(w$to))
  data.frame(region = w$region, wave = w$wave, outbreak = paste(w$region, "COVID-19"), date = gx[i],
             cases = as.integer(y[i]))
}))

# The measures: the outbreaks' measures from known_interventions.csv, and those of every COVID-19 wave from OxCGRT: the
# days, from 30 days before the wave to its end, on which the tracker raises school closing (C1) to some (2) or all (3)
# schools, workplace closing (C2) to some (2) or all non-essential (3) workplaces, restrictions on gatherings (C4) to at
# most 100 (3) or 10 (4) people, or a stay-at-home requirement (C6) to a requirement (2) or a strict one (3); the
# measures of the same day are joined.
OXCGRT_URL <- "https://github.com/OxCGRT/covid-policy-dataset"
LEVELS <- list(C1M_School.closing = c(`2` = "Some schools closed", `3` = "All schools closed"),
               C2M_Workplace.closing = c(`2` = "Some workplaces closed", `3` = "Non-essential workplaces closed"),
               C4M_Restrictions.on.gatherings = c(`3` = "Gatherings of more than 100 people banned",
                                                  `4` = "Gatherings of more than 10 people banned"),
               C6M_Stay.at.home.requirements = c(`2` = "Stay-at-home requirement", `3` = "Strict stay-at-home requirement"))
ox <- read.csv(file.path(RAW, "oxcgrt/OxCGRT_compact_national_v1.csv"))
oxs <- read.csv(file.path(RAW, "oxcgrt/OxCGRT_compact_subnational_v1.csv"))
oxcgrt <- do.call(rbind, lapply(seq_len(nrow(WAVES)), function(k) {
  w <- WAVES[k, ]
  o <- if (w$ox_region == "") ox[ox$CountryName == w$ox_country & ox$Jurisdiction == "NAT_TOTAL", ] else
    oxs[oxs$CountryName == w$ox_country & oxs$RegionName == w$ox_region & oxs$Jurisdiction == "STATE_TOTAL", ]
  o <- o[order(o$Date), ]; o$date <- as.Date(as.character(o$Date), "%Y%m%d")
  m <- do.call(rbind, lapply(names(LEVELS), function(col) {
    v <- o[[col]]; v[is.na(v)] <- 0; flag <- o[[sub("_.*", "_Flag", col)]]
    i <- which(diff(c(0, v)) > 0 & as.character(v) %in% names(LEVELS[[col]]))
    if (!length(i)) return(NULL)
    data.frame(date = o$date[i], event = paste0(LEVELS[[col]][as.character(v[i])],
                                                ifelse(!is.na(flag[i]) & flag[i] == 0, " in some regions", "")))
  }))
  if (is.null(m)) return(NULL)
  m <- m[m$date >= as.Date(w$from) - 30 & m$date <= as.Date(w$to), ]
  if (!nrow(m)) return(NULL)
  agg <- stats::aggregate(event ~ date, m, function(e) paste(e, collapse = "; "))
  agg$event <- sub("^(.)", "\\U\\1", tolower(agg$event), perl = TRUE)
  data.frame(dataset = paste(w$region, "COVID-19"), date = format(agg$date), event = agg$event, kind = "restriction",
             scope = if (w$ox_region != "") "state" else if (w$region == "Hong Kong") "territory" else "national",
             source = OXCGRT_URL)
}))
k <- read.csv(file.path(RAW, "known_interventions.csv"), stringsAsFactors = FALSE)
k <- rbind(k[k$source != OXCGRT_URL, ], oxcgrt)
interventions <- data.frame(outbreak = k$dataset, date = as.Date(k$date), measure = k$event, kind = k$kind,
                            scope = k$scope, source = k$source, stringsAsFactors = FALSE)
interventions <- interventions[order(interventions$outbreak, interventions$date), ]
rownames(interventions) <- NULL

# Incubation period and serial interval (days) of each disease, as in the paper.
disease_parameters <- data.frame(
  disease = c("COVID-19", "influenza", "SARS", "Ebola", "MERS", "smallpox", "measles"),
  incubation = c(5.2, 1.4, 6.4, 11.4, 6.7, 12, 12.5),
  serial_interval = c(7.5, 2.6, 8.4, 15.3, 12.6, 22.4, 14.9),
  serial_interval_sd = c(3.4, 1.5, 3.8, 9.3, 2.8, 6.1, 3.9),
  source = c("Li et al. (2020), NEJM 382:1199", "Lessler et al. (2009), Lancet Infect Dis 9:291; Cori et al. (2013)",
             "Donnelly et al. (2003), Lancet 361:1761; Lipsitch et al. (2003), Science 300:1966",
             "WHO Ebola Response Team (2014), NEJM 371:1481", "Cowling et al. (2015), Euro Surveill 20(25):21163",
             "Fenner et al. (1988), Smallpox and its Eradication; Cori et al. (2013)",
             "Lessler et al. (2009); Cori et al. (2013)"),
  stringsAsFactors = FALSE)

for (nm in c("covid_admissions_utah", "sars_hongkong_2003", "ebola_sierraleone_2014", "sars_toronto_2003",
             "mers_korea_2015", "smallpox_kosovo_1972", "flu_pennsylvania_2009", "measles_hagelloch_1861",
             "ebola_kikwit_1995", "covid_waves_2020", "interventions", "disease_parameters")) {
  x <- get(nm); rownames(x) <- NULL
  assign(nm, x)
  save(list = nm, file = file.path("data", paste0(nm, ".rda")), compress = "xz")
  cat(sprintf("%-24s %6d rows  %s\n", nm, nrow(x), paste(names(x), collapse = ", ")))
}
