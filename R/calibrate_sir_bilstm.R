#' @keywords internal
"_PACKAGE"

library(reticulate)

# Global environment tracking
.bilstm_env <- new.env()
.bilstm_env$model_loaded <- FALSE
.bilstm_env$model_dir <- NULL

#' Initialize BiLSTM Model for SIR Calibration
#'
#' @param model_dir Character. Path to directory containing model files.
#'                  If NULL, uses default "~/epiworldRcalibrate/epiworldRcalibrate/inst/models/LSTM_model"
#' @param force_reload Logical. Force reload even if model already loaded (default: FALSE)
#'
#' @details
#' This function loads the BiLSTM model and scalers into memory once. Call this before
#' using \code{\link{predict_sir_bilstm}} for optimal performance. The model stays loaded until
#' R session ends or \code{\link{cleanup_bilstm_model}} is called.
#'
#' Required files in \code{model_dir}:
#' \itemize{
#'   \item model4_bilstm.pt - PyTorch model weights
#'   \item scaler_additional.pkl - Scaler for additional inputs
#'   \item scaler_targets.pkl - Scaler for target outputs
#'   \item scaler_incidence.pkl - Scaler for incidence data
#' }
#'
#' @return Logical. TRUE if model loaded successfully
#' @export
#'
#' @examples
#' \dontrun{
#' # Initialize model (do this once)
#' init_bilstm_model()
#'
#' # Now make fast predictions
#' for(i in 1:100) {
#'   params <- predict_sir_bilstm(incidence_data, n = 5000, recov = 0.1)
#' }
#' }
init_bilstm_model <- function(model_dir = NULL, force_reload = FALSE) {

  if (is.null(model_dir)) {
    model_dir <- "~/Desktop/epiworldRcalibrate_fixed/epiworldRcalibrate/inst/models"
  }

  if (.bilstm_env$model_loaded && !force_reload && identical(.bilstm_env$model_dir, model_dir)) {
    message("BiLSTM model already loaded. Use force_reload=TRUE to reload.")
    return(TRUE)
  }

  if (!dir.exists(model_dir)) {
    stop(paste("Model directory does not exist:", model_dir))
  }

  base_dir <- normalizePath(model_dir)
  model_path <- normalizePath(file.path(base_dir, "model4_bilstm.pt"))
  scaler_add_path <- normalizePath(file.path(base_dir, "scaler_additional.pkl"))
  scaler_tgt_path <- normalizePath(file.path(base_dir, "scaler_targets.pkl"))
  scaler_inc_path <- normalizePath(file.path(base_dir, "scaler_incidence.pkl"))

  required_files <- c(model_path, scaler_add_path, scaler_tgt_path)
  missing_files <- required_files[!file.exists(required_files)]
  if (length(missing_files) > 0) {
    stop(paste("Required model files not found:", paste(missing_files, collapse = ", ")))
  }

  python_code <- "
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
INCIDENCE_MIN = 0
INCIDENCE_MAX = 10000

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

def create_fixed_incidence_scaler(shape):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = np.zeros(shape)
    scaler.data_max_ = np.ones(shape) * INCIDENCE_MAX
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = 1.0 / scaler.data_range_
    scaler.min_ = 0 - scaler.data_min_ * scaler.scale_
    return scaler

def load_model(model_path, scaler_add_path, scaler_tgt_path, scaler_inc_path=None):
    global _model, _scaler_add, _scaler_tgt, _scaler_inc
    _scaler_add = joblib.load(scaler_add_path)
    _scaler_tgt = joblib.load(scaler_tgt_path)
    if scaler_inc_path:
        try:
            _scaler_inc = joblib.load(scaler_inc_path)
        except:
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
        _scaler_inc = create_fixed_incidence_scaler(x.shape[1])
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

  tryCatch({
    py_run_string(python_code)
  }, error = function(e) {
    stop(paste("Failed to initialize Python environment:", e$message))
  })

  tryCatch({
    py$load_model(
      model_path = model_path,
      scaler_add_path = scaler_add_path,
      scaler_tgt_path = scaler_tgt_path,
      scaler_inc_path = if(file.exists(scaler_inc_path)) scaler_inc_path else NULL
    )
    .bilstm_env$model_loaded <- TRUE
    .bilstm_env$model_dir <- model_dir
    message("BiLSTM model loaded successfully!")
    return(TRUE)
  }, error = function(e) {
    .bilstm_env$model_loaded <- FALSE
    stop(paste("Failed to load model:", e$message))
  })
}

#' Fast SIR Parameter Prediction using BiLSTM
#'
#' @param time_series Numeric vector of length 61 containing incidence data
#' @param n Numeric. Population size
#' @param recov Numeric. Recovery rate
#'
#' @details
#' This function makes fast predictions using a pre-loaded BiLSTM model.
#' Call \code{\link{init_bilstm_model}} first to load the model into memory.
#'
#' @return Named numeric vector with components:
#' \describe{
#'   \item{ptran}{Transmission probability}
#'   \item{crate}{Contact rate}
#'   \item{R0}{Basic reproduction number}
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' init_bilstm_model()
#' incidence <- abs(rnorm(61, mean = 100, sd = 20))
#' predict_sir_bilstm(incidence, n = 5000, recov = 0.1)
#' }
predict_sir_bilstm <- function(time_series, n, recov) {
  if (!.bilstm_env$model_loaded) {
    stop("BiLSTM model not loaded. Call init_bilstm_model() first.")
  }
  stopifnot(
    length(time_series) == 61,
    is.numeric(n),
    is.numeric(recov),
    n > 0,
    recov > 0
  )
  tryCatch({
    time_series <- as.numeric(time_series)
    out <- py$predict(time_series, list(n, recov))
    names(out) <- c("ptran", "crate", "R0")
    return(out)
  }, error = function(e) {
    stop(paste("Prediction failed:", e$message))
  })
}

#' Check if BiLSTM Model is Loaded
#'
#' @return Logical indicating if the model is currently loaded
#' @export
is_bilstm_loaded <- function() {
  return(.bilstm_env$model_loaded)
}

#' Clean up BiLSTM Model
#'
#' @description
#' Frees memory by removing the loaded BiLSTM model and associated scalers.
#' Useful in long-running sessions where the model is no longer needed.
#'
#' @return None. Side-effect: model environment is cleared.
#' @export
cleanup_bilstm_model <- function() {
  if (.bilstm_env$model_loaded) {
    tryCatch({
      py$cleanup()
      .bilstm_env$model_loaded <- FALSE
      .bilstm_env$model_dir <- NULL
      message("BiLSTM model cleaned up successfully.")
    }, error = function(e) {
      warning(paste("Error during cleanup:", e$message))
    })
  } else {
    message("No BiLSTM model loaded to clean up.")
  }
}

#' Calibrate SIR Model Parameters using BiLSTM
#'
#' @param time_series Numeric vector of length 61 containing incidence data
#' @param n Numeric. Population size
#' @param recov Numeric. Recovery rate
#' @param model_dir Character. Optional. Path to model directory (used only if model is not already loaded)
#' @param auto_init Logical. If TRUE (default), automatically initializes the model if not yet loaded
#'
#' @details
#' This is a convenient wrapper that optionally initializes the model and predicts parameters
#' in a single call.
#'
#' @return Named numeric vector with components:
#' \describe{
#'   \item{ptran}{Transmission probability}
#'   \item{crate}{Contact rate}
#'   \item{R0}{Basic reproduction number}
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' calibrate_sir_bilstm(time_series = abs(rnorm(61)), n = 5000, recov = 0.1)
#' }
calibrate_sir_bilstm <- function(time_series, n, recov, model_dir = NULL, auto_init = TRUE) {
  if (!.bilstm_env$model_loaded && auto_init) {
    init_bilstm_model(model_dir = model_dir)
  }
  return(predict_sir_bilstm(time_series, n, recov))
}
