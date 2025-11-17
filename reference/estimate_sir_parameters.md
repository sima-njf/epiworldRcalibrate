# Estimate SIR Parameters from 61-day incidence

Estimate SIR Parameters from 61-day incidence

## Usage

``` r
estimate_sir_parameters(daily_cases, population_size, recovery_rate)
```

## Arguments

- daily_cases:

  Numeric vector of length 61 containing daily incidence counts for days
  0 to 60.

- population_size:

  Single numeric value giving the total population size used in the SIR
  model.

- recovery_rate:

  Single numeric value giving the recovery rate parameter of the SIR
  model.

## Value

Named numeric vector: `ptran`, `crate`, `R0`.
