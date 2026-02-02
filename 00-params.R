# Load required libraries
library(epiworldR)
library(data.table)
library(parallel)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(gridExtra)
library(cowplot)

# --------------------------
# Global Simulation Settings
# --------------------------
model_ndays <- 60    # simulation duration (days)
model_seed  <- 122   # seed for reproducibility
N_SIMS      <- 20000  # number of simulations to run

# --------------------------
# Generate Parameter Sets using Theta (SEIR version)
# --------------------------
set.seed(model_seed)  # Ensure reproducibility

n_values <- sample(5000:10000, N_SIMS, replace = TRUE)  # population size range

theta <- data.table(
  n       = n_values,
  preval  = runif(N_SIMS, 0.007, 0.02),
  crate   = runif(N_SIMS, 1, 5),              # contact rate
  incub   = runif(N_SIMS, 3, 21),             # incubation days (E->I transition)
  recov   = runif(N_SIMS, 0.071, 0.25),        # recovery rate (14-21 days infectious period)
  R0      = runif(N_SIMS, 1.1, 5)           # basic reproduction number
)

# Calculate transmission probability (ptran)
# For SEIR: R0 = (ptran * crate) / recov
theta[, ptran := (R0 * recov / crate)]

# Final dataset with needed columns (including incubation)
theta_use <- theta[, .(n, preval, crate, incub, recov, ptran,R0)]

# Print summary
summary(theta_use)
cat("\nSample of parameter sets:\n")
print(head(theta_use, 10))

# Save the parameter sets
data.table::fwrite(theta_use, "theta_use_seir.csv")

# OR using base R:
write.csv(theta_use, "theta_use_seir.csv", row.names = FALSE)
