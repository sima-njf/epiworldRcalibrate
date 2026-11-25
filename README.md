# epiworldRcalibrate

<!-- badges: start -->

<!-- how-to-cite -->
> [!NOTE]
> **How to cite epiworldRcalibrate.** If you use **epiworldRcalibrate** in published work, please cite it:
>
> Najafzadehkhoei S, Vega Yon GG, Modenesi B, Meyer D (2025). Machine Generalize Learning in Agent-Based Models. *arXiv*:2509.07013. doi:[10.48550/arXiv.2509.07013](https://doi.org/10.48550/arXiv.2509.07013)
>
> Run `citation("epiworldRcalibrate")` in R for the BibTeX entry.
<!-- how-to-cite -->

<a href="https://github.com/EpiForeSITE"><img src="https://github.com/EpiForeSITE/software/raw/e82ed88f75e0fe5c0a1a3b38c2b94509f122019c/docs/assets/foresite-software-badge.svg"></a>
<a href="https://github.com/sima-njf/epiworldRcalibrate/actions/workflows/r-cmd-check.yml"><img src="https://github.com/sima-njf/epiworldRcalibrate/actions/workflows/r-cmd-check.yml/badge.svg"></a>
<a href="https://github.com/sima-njf/epiworldRcalibrate/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

<!-- badges: end -->

**epiworldRcalibrate** calibrates the SEIR and SIR agent-based models of
[epiworldR](https://github.com/UofUEpiBio/epiworldR) from daily counts of new
infections, cases or hospital admissions.

* **DeepIMC**, a trained bidirectional LSTM shipped with the package,
  calibrates a window of 15 to 365 days from any part of an epidemic in well
  under a second, with a 95% range for R0 and the state of the epidemic on
  the first day of the window.
* The four standard methods it is compared with in the paper, **ABC-SMC**,
  **ABC-MCMC**, **differential evolution** and **Nelder-Mead**, are set up the
  same way and called through the same function.
* Whole curves can be calibrated **with or without smoothing** and **with or
  without splitting** them where their growth rate changes (segmented
  regression), and the epidemic the estimates imply can be **recovered** by
  running the agent-based model again.
* **Interventions**: find the breakpoint that follows a known measure, and the
  cases expected without the measure.
* **Real data**: the curves of the paper (COVID-19, influenza, SARS, Ebola,
  MERS, smallpox, measles) and their dated measures, loaded with `data()`.

## Installation

```r
# install.packages("remotes")
remotes::install_github("sima-njf/epiworldRcalibrate", ref = "final-version")
```

DeepIMC runs its networks in Python through
[reticulate](https://rstudio.github.io/reticulate/). The package declares the
Python packages it needs (torch, numpy, pandas, scikit-learn, joblib), and
reticulate sets them up the first time DeepIMC is used. To use a Python you
already have, set `RETICULATE_PYTHON` before loading the package. The four
standard methods need no Python (differential evolution needs the R package
`DEoptim`).

## A window, five methods

```r
library(epiworldRcalibrate)

data(flu_baltimore_1918)
y <- flu_baltimore_1918$cases[1:40]

# DeepIMC, SEIR model, with the known values of influenza
fit <- calibrate_window(y, model = "SEIR", method = "DeepIMC", known = known_values("influenza", "SEIR"))
fit
plot(fit, ahead = 14)

# All five methods on the same window (the standard methods take minutes each)
fits <- calibrate_window(y, model = "SIR", method = c("DeepIMC", "ABC_SMC", "ABC_MCMC", "DE", "NELDERMEAD"),
                         known = known_values("influenza", "SIR"))
summary(fits)
plot(fits)
```

## A whole curve: smoothing, splitting, recovering it

```r
data(sars_hongkong_2003)
kv <- known_values("SARS", "SEIR")

# Split where the growth rate changes, smooth each part (7-day average), calibrate each part
fit <- calibrate_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date, model = "SEIR",
                       split = TRUE, smoothing = TRUE, known = kv)
fit

# The same curve in one window, raw counts, differential evolution
fit_de <- calibrate_curve(sars_hongkong_2003$cases, sars_hongkong_2003$date, model = "SEIR", method = "DE",
                          split = FALSE, smoothing = FALSE, known = kv)

# The epidemic the estimates imply, with the known measures marked
run <- recover_curve(fit)
plot(run, interventions = subset(interventions, outbreak == "Hong Kong SARS 2003"))
```

## Interventions

```r
data(covid_waves_2020)
nz <- subset(covid_waves_2020, region == "New Zealand")
fit <- calibrate_curve(nz$cases, nz$date, model = "SEIR", weekday = TRUE, known = known_values("COVID-19", "SEIR"))

find_intervention(fit, interventions, outbreak = "New Zealand COVID-19")   # the cut after a known restriction
eff <- intervention_effect(fit, interventions = interventions, outbreak = "New Zealand COVID-19")
eff          # R0 and Reff before and after, cases expected without the measure, cases averted
plot(eff)
```

## The main functions

| Function | What it does |
|---|---|
| `calibrate_window()` | one window with `"DeepIMC"`, `"ABC_SMC"`, `"ABC_MCMC"`, `"DE"` or `"NELDERMEAD"`, SEIR or SIR |
| `calibrate_curve()` | a whole curve, with or without smoothing (`smoothing =`) and splitting (`split =`) |
| `split_curve()`, `smooth_cases()` | the split (segmented regression) and the smoothers on their own |
| `recover_curve()` | the epidemic the estimates imply, run again with the agent-based model; `plot()` marks the measures |
| `find_intervention()` | the breakpoint 0 to 14 days after a known restriction where the growth rate falls |
| `intervention_effect()` | the part before the cut run on over the part after it: cases expected without the measure and averted |
| `predict_cases()`, `coverage()`, `interval_score()` | bands of daily cases, and their scores |
| `known_values()` | the known values of a disease, as in the paper |
| `window_calibrator()`, `window_baselines()`, `compare_methods()` | the calibrators themselves, and comparisons on many windows |
| `simulate_window_data()`, `train_window_calibrator()`, `tune_window_calibrator()` | train your own DeepIMC networks |

## Data

| Dataset | What |
|---|---|
| `flu_baltimore_1918`, `sars_hongkong_2003`, `sars_toronto_2003`, `mers_korea_2015`, `ebola_sierraleone_2014`, `ebola_kikwit_1995`, `smallpox_kosovo_1972`, `flu_pennsylvania_2009`, `measles_hagelloch_1861` | daily cases of outbreaks |
| `covid_admissions_utah`, `admissions_utah`, `covid_utah` | COVID-19 and influenza hospital admissions and COVID-19 cases in Utah |
| `covid_waves_2020` | daily COVID-19 cases of 27 waves of 2020 with dated restrictions |
| `covid_us_states` | daily COVID-19 cases in the US states, 2020-23 |
| `interventions` | the dated measures of these outbreaks |
| `disease_parameters` | incubation period and serial interval of each disease |

## Vignettes

* `vignette("calibrating-windows")`: the five methods on simulated SEIR and SIR windows with known parameters.
* `vignette("real-curves")`: the data, smoothing, splitting and recovering real curves, and the methods on a real curve.
* `vignette("interventions")`: where the curve changed after a known measure, and the epidemic without it, for 27 COVID-19 waves.

## Funding

This work was supported by the Centers for Disease Control and Prevention (award numbers 1U01CK000585 and
75D30121F00003).
