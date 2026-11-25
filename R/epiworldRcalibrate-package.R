#' epiworldRcalibrate: calibrate agent-based epidemic models in seconds
#'
#' The package calibrates the SEIR and SIR agent-based models of
#' \pkg{epiworldR} from daily counts, with DeepIMC (a trained bidirectional
#' LSTM) or four standard methods, and works with whole curves and
#' interventions.
#'
#' The main functions:
#' * [calibrate_window()]: one window of daily counts, with `"DeepIMC"`,
#'   `"ABC_SMC"`, `"ABC_MCMC"`, `"DE"` or `"NELDERMEAD"`, for SEIR or SIR.
#' * [calibrate_curve()]: a whole curve, with or without smoothing and with or
#'   without splitting it where its growth rate changes ([split_curve()],
#'   [smooth_cases()]).
#' * [recover_curve()]: the epidemic the estimates imply, run again with the
#'   agent-based model, and its plot (with the known measures marked).
#' * [find_intervention()] and [intervention_effect()]: the breakpoint that
#'   follows a known measure, and the cases expected without it.
#' * [predict_cases()], [coverage()], [interval_score()]: bands and scores.
#' * Data: [outbreak_data], [covid_waves_2020], [interventions],
#'   [flu_baltimore_1918], [covid_utah], [admissions_utah], [covid_us_states],
#'   [disease_parameters] and [known_values()].
#'
#' See the vignettes: `vignette("calibrating-windows")`,
#' `vignette("real-curves")` and `vignette("interventions")`.
#'
#' @section How to cite:
#' If you use \pkg{epiworldRcalibrate} in published work, please cite it. Run
#' `citation("epiworldRcalibrate")` in R for the full entry.
#' @keywords internal
"_PACKAGE"
