## code to prepare `utah_covid_data` dataset goes here

library(dplyr)

# Helper function to download and extract data from ZIP
get_data_from_url <- function(data_url, target_file) {
  tryCatch({
    temp_file <- tempfile()
    download.file(data_url, temp_file, mode = "wb")

    datafile_name <- grep(
      target_file,
      unzip(temp_file, list = TRUE)$Name,
      ignore.case = TRUE,
      value = TRUE
    )

    if (length(datafile_name) == 0L)
      stop("Target file not found in ZIP archive.")

    datafile <- unz(temp_file, datafile_name[1])
    collected_data <- read.csv(datafile)
    collected_data
  },
  error = function(cond) {
    message("Error: ", conditionMessage(cond))
    return(-1)
  },
  warning = function(cond) {
    message("Warning: ", conditionMessage(cond))
    return(-2)
  },
  finally = unlink(temp_file))
}

# Main function to get COVID data
get_covid_data <- function(n_days) {
  data_url <- "https://coronavirus-dashboard.utah.gov/Utah_COVID19_data.zip"
  target_file_regex <- "Trends_Epidemic+"

  covid_data <- get_data_from_url(data_url, target_file_regex)

  if (is.numeric(covid_data) && covid_data < 0)
    stop("Failed to download or read Utah COVID-19 data.")

  covid_data$Date <- as.Date(covid_data$Date)

  last_date <- max(covid_data$Date, na.rm = TRUE)
  covid_data <- subset(covid_data, Date > (last_date - n_days)) |>
    arrange(Date)

  stopifnot("Daily.Cases" %in% names(covid_data))
  covid_data
}

# Generate the dataset (adjust n_days as needed)
utah_covid_data <- get_covid_data(n_days = 365)

# Save to data/ folder
usethis::use_data(utah_covid_data, overwrite = TRUE)

