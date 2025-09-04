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

#' Resolve model directory path across platforms
#' @param model_dir User-provided path (optional)
#' @return Resolved, normalized path to model directory
#' @keywords internal
.resolve_model_dir <- function(model_dir = NULL) {
  if (!is.null(model_dir)) {
    # User provided a path - validate and normalize it
    expanded_path <- path.expand(model_dir)
    if (!dir.exists(expanded_path)) {
      stop(sprintf("Specified model directory does not exist: %s", model_dir))
    }
    return(normalizePath(expanded_path, mustWork = TRUE))
  }

  # Try multiple fallback locations in order of preference
  search_paths <- c(
    # 1. Installed package models directory (most preferred)
    system.file("models", package = "epiworldRcalibrate"),

    # 2. extdata directory (alternative package location)
    system.file("extdata", "models", package = "epiworldRcalibrate"),

    # 3. Current working directory + models
    file.path(getwd(), "models"),
    file.path(getwd(), "inst", "models"),

    # 4. User's home directory + models
    file.path(path.expand("~"), "epiworldRcalibrate_models"),

    # 5. Common system locations based on OS
    switch(Sys.info()["sysname"],
           "Windows" = c(
             file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "epiworldRcalibrate", "models"),
             file.path(Sys.getenv("LOCALAPPDATA"), "epiworldRcalibrate", "models")
           ),
           "Darwin" = c(  # macOS
             file.path(path.expand("~"), "Library", "Application Support", "epiworldRcalibrate", "models"),
             file.path("/usr/local/share/epiworldRcalibrate/models")
           ),
           "Linux" = c(
             file.path(path.expand("~"), ".local", "share", "epiworldRcalibrate", "models"),
             file.path("/usr/local/share/epiworldRcalibrate/models"),
             file.path("/opt/epiworldRcalibrate/models")
           ),
           # Default fallback for unknown systems
           character(0)
    )
  )

  # Remove empty strings and NULL entries
  search_paths <- search_paths[nzchar(search_paths) & !is.null(search_paths)]

  # Find the first existing directory
  for (path in search_paths) {
    if (dir.exists(path)) {
      # Verify it contains required files
      required_files <- c("model4_bilstm.pt", "scaler_additional.pkl", "scaler_targets.pkl")
      if (all(file.exists(file.path(path, required_files)))) {
        return(normalizePath(path, mustWork = TRUE))
      }
    }
  }

  # If nothing found, provide helpful error message
  stop(sprintf(
    "Could not locate model directory. Searched the following locations:\n%s\n\n%s",
    paste(sprintf("  - %s", search_paths), collapse = "\n"),
    "Please either:\n  1. Install the package properly with model files\n  2. Provide model_dir parameter explicitly\n  3. Place model files in one of the searched locations"
  ))
}

# ---- Private: loader ----
.ensure_bilstm_ready <- function(model_dir = NULL, force_reload = FALSE) {
  # Make sure Python is available
  if (!reticulate::py_available(initialize = TRUE)) {
    stop("Python is not available to reticulate. Configure reticulate::use_python()/use_virtualenv()/use_condaenv() first.")
  }

  # Resolve model_dir using robust cross-platform logic
  resolved_model_dir <- .resolve_model_dir(model_dir)

  # Short-circuit if already loaded for the same dir (and not forcing reload)
  if (.bilstm_env$model_loaded && !force_reload && identical(.bilstm_env$model_dir, resolved_model_dir)) {
    return(invisible(TRUE))
  }

  # Build file paths using file.path for cross-platform compatibility
  model_path      <- file.path(resolved_model_dir, "model4_bilstm.pt")
  scaler_add_path <- file.path(resolved_model_dir, "scaler_additional.pkl")
  scaler_tgt_path <- file.path(resolved_model_dir, "scaler_targets.pkl")
  scaler_inc_path <- file.path(resolved_model_dir, "scaler_incidence.pkl")

  # Verify required files exist
  required <- c(model_path, scaler_add_path, scaler_tgt_path)
  missing  <- required[!file.exists(required)]
  if (length(missing) > 0) {
    stop(sprintf("Required model files not found:\n%s\n\nIn directory: %s",
                 paste(sprintf("  - %s", basename(missing)), collapse = "\n"),
                 resolved_model_dir))
  }

  # Load Python code
  tryCatch(
    reticulate::py_run_string(.bilstm_env$python_code),
    error = function(e) stop(sprintf("Failed to initialize Python code: %s", e$message))
  )

  # Load weights + scalers with normalized paths
  tryCatch({
    reticulate::py$load_model(
      model_path      = normalizePath(model_path, mustWork = TRUE),
      scaler_add_path = normalizePath(scaler_add_path, mustWork = TRUE),
      scaler_tgt_path = normalizePath(scaler_tgt_path, mustWork = TRUE),
      scaler_inc_path = if (file.exists(scaler_inc_path)) normalizePath(scaler_inc_path, mustWork = TRUE) else NULL
    )
    .bilstm_env$model_loaded <- TRUE
    .bilstm_env$model_dir    <- resolved_model_dir
    message(sprintf("BiLSTM model loaded successfully from: %s", resolved_model_dir))
    invisible(TRUE)
  }, error = function(e) {
    .bilstm_env$model_loaded <- FALSE
    .bilstm_env$model_dir    <- NULL
    stop(sprintf("Failed to load model from %s: %s", resolved_model_dir, e$message))
  })
}

#' Calibrate SIR parameters using a pre-trained BiLSTM
#'
#' Lazily initializes the Python model on first use (unless `auto_init = FALSE`).
#' The model directory is automatically detected across platforms.
#'
#' @param time_series Numeric vector of length 61: incidence data
#' @param n Numeric (>0). Population size
#' @param recov Numeric (>0). Recovery rate
#' @param model_dir Optional path to model/scaler files. If NULL, will search
#'   standard locations across platforms.
#' @param auto_init Logical (default TRUE). If TRUE, auto-loads the model on first call
#' @return Named numeric vector: `c(ptran, crate, R0)`
#' @export
#' @examples
#' \dontrun{
#' # One-liner: auto-initializes, then predicts
#' res <- calibrate_sir_bilstm(abs(rnorm(61, 100, 20)), n = 5000, recov = 0.1)
#'
#' # Specify custom model directory
#' res <- calibrate_sir_bilstm(time_series, n = 5000, recov = 0.1,
#'                            model_dir = "/path/to/my/models")
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
#' @param model_dir Character. Path to directory with model files. If NULL,
#'   will search standard locations across platforms.
#' @param force_reload Logical. Reload even if already loaded (default FALSE).
#' @return TRUE if model loaded successfully
#' @export
#' @examples
#' \dontrun{
#' # Auto-detect model location
#' init_bilstm_model()
#'
#' # Use specific directory
#' init_bilstm_model("/path/to/models")
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

#' Get current model directory path
#' @return Character string with current model directory, or NULL if not loaded
#' @export
get_bilstm_model_dir <- function() {
  .bilstm_env$model_dir
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

#' Show searched model directory locations
#'
#' Utility function to help debug model loading issues by showing
#' all the directories that would be searched for model files.
#'
#' @return Character vector of searched paths
#' @export
show_model_search_paths <- function() {
  search_paths <- c(
    system.file("models", package = "epiworldRcalibrate"),
    system.file("extdata", "models", package = "epiworldRcalibrate"),
    file.path(getwd(), "models"),
    file.path(getwd(), "inst", "models"),
    file.path(path.expand("~"), "epiworldRcalibrate_models"),
    switch(Sys.info()["sysname"],
           "Windows" = c(
             file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "epiworldRcalibrate", "models"),
             file.path(Sys.getenv("LOCALAPPDATA"), "epiworldRcalibrate", "models")
           ),
           "Darwin" = c(
             file.path(path.expand("~"), "Library", "Application Support", "epiworldRcalibrate", "models"),
             file.path("/usr/local/share/epiworldRcalibrate/models")
           ),
           "Linux" = c(
             file.path(path.expand("~"), ".local", "share", "epiworldRcalibrate", "models"),
             file.path("/usr/local/share/epiworldRcalibrate/models"),
             file.path("/opt/epiworldRcalibrate/models")
           ),
           character(0)
    )
  )

  # Clean up and show status
  search_paths <- search_paths[nzchar(search_paths) & !is.null(search_paths)]

  cat("Model directory search paths (in order of preference):\n")
  for (i in seq_along(search_paths)) {
    path <- search_paths[i]
    exists_status <- if (dir.exists(path)) "EXISTS" else "not found"

    if (dir.exists(path)) {
      required_files <- c("model4_bilstm.pt", "scaler_additional.pkl", "scaler_targets.pkl")
      has_files <- all(file.exists(file.path(path, required_files)))
      file_status <- if (has_files) "(has required files)" else "(missing some files)"
    } else {
      file_status <- ""
    }

    cat(sprintf("%2d. %s [%s] %s\n", i, path, exists_status, file_status))
  }

  invisible(search_paths)
}
