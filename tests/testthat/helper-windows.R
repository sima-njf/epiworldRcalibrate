# A short window of daily new infectious people of an SIR epidemic, for quick tests.
sir_window <- function(days = 40, seed = 3) {
  m <- epiworldR::ModelSIRCONN(name = "test", n = 5000, prevalence = 0.002, contact_rate = 3,
                               transmission_rate = 0.1, recovery_rate = 0.15)
  epiworldR::verbose_off(m)
  epiworldR::run(m, ndays = days + 1, seed = seed)
  tr <- epiworldR::get_hist_transition_matrix(m)
  tr <- tr[tr$state_from == "Susceptible" & tr$state_to == "Infected", ]
  y <- tr$counts[match(seq_len(days), tr$date)]
  y[is.na(y)] <- 0
  y
}

# DeepIMC needs Python with torch; its tests run only when asked for.
skip_without_python <- function() {
  if (!identical(Sys.getenv("EPIWORLDRCALIBRATE_PYTHON_TESTS"), "true"))
    skip("Set EPIWORLDRCALIBRATE_PYTHON_TESTS=true to run the DeepIMC tests (they need Python with torch).")
}
