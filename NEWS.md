# epiworldRcalibrate 1.0.0

* DeepIMC: trained any-length bidirectional LSTMs for the SEIR and SIR models (`window_calibrator()`), for windows of
  15 to 365 days from any part of an epidemic, with a 95% range for R0 and the state on the first day of the window.
* `calibrate_window()`: one function for DeepIMC and the four standard methods (`"ABC_SMC"`, `"ABC_MCMC"`, `"DE"`,
  `"NELDERMEAD"`), for SEIR and SIR, one method or several at once, with `summary()` and `plot()` methods.
* `calibrate_curve()`: whole curves with or without smoothing (7-day moving average by default, Gaussian kernels,
  bandwidth chosen by cross-validation or by calibration) and with or without splitting where the growth rate changes
  (`split_curve()`, segmented regression).
* `recover_curve()` (also `run_epidemic()`): the epidemic the estimates imply, with plots that mark known measures.
* `find_intervention()` and `intervention_effect()`: the breakpoint that follows a known restriction, and the cases
  expected without it and averted.
* New datasets: `covid_admissions_utah`, `sars_hongkong_2003`, `ebola_sierraleone_2014`, `sars_toronto_2003`,
  `mers_korea_2015`, `smallpox_kosovo_1972`, `flu_pennsylvania_2009`, `measles_hagelloch_1861`, `ebola_kikwit_1995`,
  `covid_waves_2020`, `interventions`, `disease_parameters`; `known_values()`.
* Three vignettes: calibrating windows, real curves, and interventions.
* The functions of version 0.1 (`calibrate_sir()`, `estimate_sir_parameters()`, `init_bilstm_model()`, ...) are
  replaced by `calibrate_window(model = "SIR")`; the original 61-day SIR network is still available through
  `pretrained_sir_calibrator()`.

# epiworldRcalibrate 0.1.0

* First version
