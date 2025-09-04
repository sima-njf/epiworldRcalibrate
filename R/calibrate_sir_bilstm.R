#' BiLSTM-backed SIR calibration helpers
#'
#' Entry point: [calibrate_sir_bilstm()], which lazily initializes the Python model
#' the first time you call it. Back-compat wrappers [init_bilstm_model()] and
#' [predict_sir_bilstm()] are exported so existing code continues to work.
#'
#' Required files in `model_dir`:
#' - model4_bilstm.pt
#' - scaler_additional.pkl
#' - scaler_targets.pkl
#' - (optional) scaler_incidence.pkl
#'
#' @keywords internal

# ---- Private: process-scoped cache ----
.bilstm_env <- new.env(parent = emptyenv())
.bilstm_env$model_loaded <- FALSE
.bilstm_env$model_dir    <- NULL

# ---- Private: bundled Python implementation (single load into reticulate) ----
.bilstm_env$python_code <- "
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

_model = None
_scaler_add = None
_scaler_tgt = None
_scaler_inc = None
_device = torch.device('cpu')
INCIDENCE_MAX = 10000.0

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, additional_dim, output_dim, dropout):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim + additional_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, additional_inputs):
        _, (h_n, _) = self.bilstm(x)
        hid = torch.cat((h_n[-2], h_n[-1]), dim=1)
        combined = torch.cat((hid, additional_inputs), dim=1)
        x = torch.relu(self.fc1(combined))
        out = self.fc2(x)
        return torch.stack([
            self.sigmoid(out[:, 0]),
            self.softplus(out[:, 1]),
            self.softplus(out[:, 2])
        ], dim=1)

def _fit_fixed_incidence_scaler(n_features):
    # Fit a MinMaxScaler on synthetic [0, INCIDENCE_MAX] bounds so .transform works robustly
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    fake = np.vstack([np.zeros(n_features), np.ones(n_features) * INCIDENCE_MAX])
    scaler.fit(fake)
    return scaler

def load_model(model_path, scaler_add_path, scaler_tgt_path, scaler_inc_path=None):
    global _model, _scaler_add, _scaler_tgt, _scaler_inc
    _scaler_add = joblib.load(scaler_add_path)
    _scaler_tgt = joblib.load(scaler_tgt_path)

    if scaler_inc_path:
        try:
            _scaler_inc = joblib.load(scaler_inc_path)
        except Exception:
            _scaler_inc = None
    else:
        _scaler_inc = None

    _model = BiLSTMModel(input_dim=1, hidden_dim=160, num_layers=3,
                         additional_dim=2, output_dim=3, dropout=0.5)
    state = torch.load(model_path, map_location=_device)
    _model.load_state_dict(state)
    _model.to(_device).eval()

def predict(seq, additional_pair):
    global _scaler_inc
    x = np.asarray(seq, dtype=np.float32).reshape(1, -1, 1)

    if _scaler_inc is None:
        _scaler_inc = _fit_fixed_incidence_scaler(x.shape[1])

    x_scaled = _scaler_inc.transform(x.reshape(1, -1)).reshape(1, -1, 1)

    add_np = np.array([additional_pair], dtype=np.float32)
    add_scaled = _scaler_add.transform(add_np)

    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=_device)
    add_t = torch.tensor(add_scaled, dtype=torch.float32, device=_device)

    with torch.no_grad():
        out = _model(x_t, add_t).cpu().numpy()

    return _scaler_tgt.inverse_transform(out)[0].tolist()

def cleanup():
    global _model, _scaler_add, _scaler_tgt, _scaler_inc
    _model = None
    _scaler_add = None
    _scaler_tgt = None
    _scaler_inc = None
"

# ---- Private: loader ----
.ensure_bilstm_ready <- function(model_dir = NULL, force_reload = FALSE) {
  # Make sure Python is available
  if (!reticulate::py_available(initialize = TRUE)) {
    stop("Python is not available to reticulate. Configure reticulate::use_python()/use_virtualenv()/use_condaenv() first.")
  }

  # Resolve model_dir
  if (is.null(model_dir)) {
    # Prefer an installed package path (inst/models) if present
    pkg_dir <- system.file("inst/models", package = "epiworldRcalibrate")
    if (nzchar(pkg_dir)) {
      model_dir <- pkg_dir
    } else {
      # Fallback to your prior default (adjust if needed)
      model_dir <- "~/Desktop/epiworldRcalibrate_fixed/epiworldRcalibrate/inst/models"
    }
  }

  # Short-circuit if already loaded for the same dir (and not forcing reload)
  if (.bilstm_env$model_loaded && !force_reload && identical(.bilstm_env$model_dir, model_dir)) {
    return(invisible(TRUE))
  }

  # Validate directory + files
  if (!dir.exists(path.expand(model_dir))) {
    stop(sprintf("Model directory does not exist: %s", model_dir))
  }
  base_dir <- normalizePath(path.expand(model_dir), mustWork = TRUE)
  model_path      <- file.path(base_dir, "model4_bilstm.pt")
  scaler_add_path <- file.path(base_dir, "scaler_additional.pkl")
  scaler_tgt_path <- file.path(base_dir, "scaler_targets.pkl")
  scaler_inc_path <- file.path(base_dir, "scaler_incidence.pkl")

  required <- c(model_path, scaler_add_path, scaler_tgt_path)
  missing  <- required[!file.exists(required)]
  if (length(missing) > 0) {
    stop(sprintf("Required model files not found: %s", paste(missing, collapse = ", ")))
  }

  # Load Python code
  tryCatch(
    reticulate::py_run_string(.bilstm_env$python_code),
    error = function(e) stop(sprintf("Failed to initialize Python code: %s", e$message))
  )

  # Load weights + scalers
  tryCatch({
    reticulate::py$load_model(
      model_path      = normalizePath(model_path, mustWork = TRUE),
      scaler_add_path = normalizePath(scaler_add_path, mustWork = TRUE),
      scaler_tgt_path = normalizePath(scaler_tgt_path, mustWork = TRUE),
      scaler_inc_path = if (file.exists(scaler_inc_path)) normalizePath(scaler_inc_path, mustWork = TRUE) else NULL
    )
    .bilstm_env$model_loaded <- TRUE
    .bilstm_env$model_dir    <- model_dir
    invisible(TRUE)
  }, error = function(e) {
    .bilstm_env$model_loaded <- FALSE
    .bilstm_env$model_dir    <- NULL
    stop(sprintf("Failed to load model: %s", e$message))
  })
}

#' Calibrate SIR parameters using a pre-trained BiLSTM
#'
#' Lazily initializes the Python model on first use (unless `auto_init = FALSE`).
#'
#' @param time_series Numeric vector of length 61: incidence data
#' @param n Numeric (>0). Population size
#' @param recov Numeric (>0). Recovery rate
#' @param model_dir Optional path to model/scaler files (see Details)
#' @param auto_init Logical (default TRUE). If TRUE, auto-loads the model on first call
#' @return Named numeric vector: `c(ptran, crate, R0)`
#' @export
#' @examples
#' \dontrun{
#' # One-liner: auto-initializes, then predicts
#' res <- calibrate_sir_bilstm(abs(rnorm(61, 100, 20)), n = 5000, recov = 0.1)
#' }
calibrate_sir_bilstm <- function(time_series, n, recov, model_dir = NULL, auto_init = TRUE) {
  if (auto_init) .ensure_bilstm_ready(model_dir = model_dir)

  if (!isTRUE(.bilstm_env$model_loaded)) {
    stop("BiLSTM model not loaded. Call calibrate_sir_bilstm(..., auto_init = TRUE) or init_bilstm_model() first.")
  }

  stopifnot(is.numeric(time_series), length(time_series) == 61,
            is.numeric(n), n > 0,
            is.numeric(recov), recov > 0)

  ts_num <- as.numeric(time_series)

  out <- tryCatch({
    res <- reticulate::py$predict(ts_num, list(n, recov))
    names(res) <- c("ptran", "crate", "R0")
    res
  }, error = function(e) stop(sprintf("Prediction failed: %s", e$message)))

  return(out)
}

#' Initialize BiLSTM Model for SIR Calibration (back-compat)
#'
#' @param model_dir Character. Path to directory with model files.
#' @param force_reload Logical. Reload even if already loaded (default FALSE).
#' @return TRUE if model loaded successfully
#' @export
#' @examples
#' \dontrun{
#' init_bilstm_model()
#' }
init_bilstm_model <- function(model_dir = NULL, force_reload = FALSE) {
  .ensure_bilstm_ready(model_dir = model_dir, force_reload = force_reload)
  isTRUE(.bilstm_env$model_loaded)
}

#' Fast SIR parameter prediction (back-compat)
#'
#' @param time_series Numeric vector of length 61 (incidence)
#' @param n Numeric (>0). Population size
#' @param recov Numeric (>0). Recovery rate
#' @return Named numeric vector `c(ptran, crate, R0)`
#' @export
predict_sir_bilstm <- function(time_series, n, recov) {
  # Try to auto-init if not loaded (uses default model_dir resolution)
  if (!isTRUE(.bilstm_env$model_loaded)) {
    try(.ensure_bilstm_ready(model_dir = NULL, force_reload = FALSE), silent = TRUE)
  }
  if (!isTRUE(.bilstm_env$model_loaded)) {
    stop("BiLSTM model not loaded. Call init_bilstm_model() or use calibrate_sir_bilstm(..., auto_init = TRUE).")
  }
  calibrate_sir_bilstm(time_series, n, recov, auto_init = FALSE)
}

#' Is the BiLSTM model loaded?
#' @return Logical
#' @export
is_bilstm_loaded <- function() {
  isTRUE(.bilstm_env$model_loaded)
}

#' Free Python-side model and scalers
#' @return NULL (invisible)
#' @export
cleanup_bilstm_model <- function() {
  if (!isTRUE(.bilstm_env$model_loaded)) {
    message("No BiLSTM model loaded to clean up.")
    return(invisible(NULL))
  }
  tryCatch({
    reticulate::py$cleanup()
    .bilstm_env$model_loaded <- FALSE
    .bilstm_env$model_dir    <- NULL
    message("BiLSTM model cleaned up successfully.")
  }, error = function(e) {
    warning(sprintf("Error during cleanup: %s", e$message))
  })
  invisible(NULL)
}
