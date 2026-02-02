library(epiworldR)
library(data.table)
library(parallel)

run_seir_simulations <- function(N, ndays, ncores, theta, seeds,
                                 disease_name = "Disease",
                                 output_file_csv = "incidence.csv") {

  cat("==============================================\n")
  cat("Running", N, "SEIR simulations on", ncores, "cores\n")
  cat("==============================================\n")

  # Run simulations in parallel
  incidence_list <- parallel::mclapply(1:N, FUN = function(i) {

    set.seed(seeds[i])

    # Create SEIR model
    m <- epiworldR::ModelSEIRCONN(
      name              = disease_name,
      n                 = theta$n[i],
      prevalence        = theta$preval[i],
      contact_rate      = theta$crate[i],
      incubation_days   = theta$incub[i],
      transmission_rate = theta$ptran[i],
      recovery_rate     = theta$recov[i]
    )

    # Turn off verbose output
    epiworldR::verbose_off(m)

    # Run the simulation
    epiworldR::run(m, ndays = ndays)

    # Extract incidence data
    incidence <- epiworldR::plot_incidence(m, plot = FALSE)
    incidence_dt <- data.table::as.data.table(incidence)

    # Extract only infected counts (as a vector)
    infected_vector <- incidence_dt$Infected[1:(ndays + 1)]

    return(infected_vector)

  }, mc.cores = ncores)

  # Combine all simulations
  cat("\n==============================================\n")
  cat("Combining results from all simulations...\n")
  cat("==============================================\n")

  # Convert list to matrix (each row = one simulation, each column = one day)
  infected_matrix <- do.call(rbind, incidence_list)

  # Convert to data.table
  infected_dt <- data.table::as.data.table(infected_matrix)

  # Rename columns to V1, V2, V3, ... V(ndays+1)
  colnames(infected_dt) <- paste0("V", 1:(ndays + 1))

  # Save as CSV
  cat("\n==============================================\n")
  cat("Saving incidence data to:", output_file_csv, "\n")
  cat("==============================================\n")

  data.table::fwrite(infected_dt, file = output_file_csv)

  cat("\n==============================================\n")
  cat("Simulation complete!\n")
  cat("Total simulations:", N, "\n")
  cat("Days per simulation:", ndays + 1, "\n")
  cat("Matrix dimensions:", dim(infected_dt), "\n")
  cat("==============================================\n")

  cat("\nInfected incidence preview:\n")
  print(head(infected_dt, 10))

  return(infected_dt)
}

# --------------------------
# 1. Load theta parameters from CSV
# --------------------------
theta_use <- data.table::fread("theta_use_seir.csv")

# --------------------------
# 2. Set number of simulations
# --------------------------
N_SIMS <- nrow(theta_use)  # Use all rows from theta.csv
#N_SIMS <- 2  # Uncomment to test with just 2 simulations

# --------------------------
# 3. Generate seeds
# --------------------------
set.seed(122)
seeds <- sample(1:1000000, N_SIMS)

# --------------------------
# 4. Run simulations
# --------------------------
incidence_data <- run_seir_simulations(
  N            = N_SIMS,
  ndays        = 60,  # This will create V1 to V61 (days 0-60)
  ncores       = 6,
  theta        = theta_use,
  seeds        = seeds,
  disease_name = "General Disease",
  output_file_csv = "incidence.csv"
)

# --------------------------
# 5. View results
# --------------------------
cat("\n==============================================\n")
cat("Final Results:\n")
cat("==============================================\n")

print(head(incidence_data))
print(dim(incidence_data))
