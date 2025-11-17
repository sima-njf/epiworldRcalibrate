# Calibrate SIR Parameters (one-step)

This is a convenience wrapper that optionally initializes the BiLSTM
model and then calls
[`estimate_sir_parameters()`](https://sima-njf.github.io/epiworldRcalibrate/reference/estimate_sir_parameters.md)
on the provided data.

## Usage

``` r
calibrate_sir(
  daily_cases,
  population_size,
  recovery_rate,
  model_dir = NULL,
  auto_init = TRUE
)
```

## Arguments

- daily_cases:

  Numeric vector of length 61 containing daily incidence counts (day 0
  to day 60).

- population_size:

  Single numeric value giving the total population size.

- recovery_rate:

  Single numeric value giving the recovery rate parameter.

- model_dir:

  Optional path to the directory containing the trained BiLSTM model and
  scaler files. If `NULL`, the package's bundled assets are used.

- auto_init:

  Logical; if `TRUE` (default), automatically calls
  [`init_bilstm_model()`](https://sima-njf.github.io/epiworldRcalibrate/reference/init_bilstm_model.md)
  when the model is not yet loaded.

## Value

Named numeric vector: `ptran`, `crate`, `R0`.
