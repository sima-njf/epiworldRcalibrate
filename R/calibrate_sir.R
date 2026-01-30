#' @keywords internal
"_PACKAGE"

# =============================================================================
# Internal State
# =============================================================================

# Internal env to track BiLSTM model state
#' @keywords internal
.bilstm_env <- new.env(parent = emptyenv())
.bilstm_env$model_loaded <- FALSE
.bilstm_env$model_dir    <- NULL
.bilstm_env$venv_name    <- "epiworldRcalibrate"

# =============================================================================
# Python bootstrap (private)
# =============================================================================

#' Find available Python installation
#' @keywords internal
.find_python <- function() {
  user_py <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(user_py) && file.exists(user_py)) return(user_py)

  py_config <- tryCatch(reticulate::py_config(), error = function(e) NULL)
  if (!is.null(py_config) && !is.null(py_config$python))
    return(py_config$python)

  if (.Platform$OS.type == "windows") {
    candidates <- c("python.exe", "python3.exe",
                    file.path(Sys.getenv("LOCALAPPDATA"), "Programs/Python/*/python.exe"))
  } else {
    candidates <- c(
      "/usr/bin/python3", "/usr/local/bin/python3",
      "/usr/bin/python", "/usr/local/bin/python",
      "/opt/homebrew/bin/python3",
      paste0(Sys.getenv("HOME"), "/.pyenv/shims/python3")
    )
  }

  for (py in candidates) {
    if (file.exists(py)) return(py)
    if (grepl("\\*", py)) {
      matches <- Sys.glob(py)
      if (length(matches) > 0) return(matches[1])
    }
  }

  py_path <- tryCatch({
    if (.Platform$OS.type == "windows") {
      system2("where", "python", stdout = TRUE, stderr = FALSE)[1]
    } else {
      system2("which", "python3", stdout = TRUE, stderr = FALSE)[1]
    }
  }, error = function(e) NULL)

  if (!is.null(py_path) && nzchar(py_path) && file.exists(py_path))
    return(py_path)

  return(NULL)
}

#' Ensure Python + necessary modules are ready (FIXED VERSION)
#' @keywords internal
.ensure_python_ready <- function() {

  # Step 1: Initialize Python
  user_py <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(user_py)) {
    message("Using user-specified Python: ", user_py)
    if (!file.exists(user_py)) {
      stop("RETICULATE_PYTHON points to non-existent file: ", user_py)
    }
    reticulate::use_python(user_py, required = TRUE)
  } else {
    vname <- .bilstm_env$venv_name
    envs <- tryCatch(reticulate::virtualenv_list(), error = function(e) character())

    if (vname %in% envs) {
      message("Using existing virtual environment: ", vname)
      reticulate::use_virtualenv(vname, required = FALSE)
    } else {
      message("Creating new virtual environment: ", vname)
      python_path <- .find_python()
      if (is.null(python_path))
        stop("Could not find Python. Install Python 3.7+ or set RETICULATE_PYTHON.")

      message("Found Python at: ", python_path)

      tryCatch({
        reticulate::virtualenv_create(vname, python = python_path)
        reticulate::use_virtualenv(vname, required = FALSE)
      }, error = function(e) {
        stop("Failed to create virtual environment: ", conditionMessage(e))
      })
    }
  }

  # Step 2: Verify Python is available
  if (!reticulate::py_available(initialize = TRUE))
    stop("Python could not be initialized. Check your Python installation.")

  # Step 3: Define required packages with proper names
  required_packages <- list(
    list(check_name = "numpy", install_name = "numpy", min_version = NULL),
    list(check_name = "sklearn", install_name = "scikit-learn", min_version = NULL),
    list(check_name = "joblib", install_name = "joblib", min_version = NULL),
    list(check_name = "torch", install_name = "torch", min_version = NULL)
  )

  # Step 4: Check what needs installation
  needs_install <- character()

  for (pkg in required_packages) {
    available <- tryCatch({
      reticulate::py_module_available(pkg$check_name)
    }, error = function(e) FALSE)

    if (!available) {
      message("Package '", pkg$check_name, "' not found, will install")
      needs_install <- c(needs_install, pkg$install_name)
    } else {
      message("Package '", pkg$check_name, "' found")
    }
  }

  # Step 5: Install missing packages
  if (length(needs_install) > 0) {
    message("Installing missing Python packages...")

    # Install non-torch packages first
    non_torch <- setdiff(needs_install, "torch")
    if (length(non_torch) > 0) {
      message("Installing: ", paste(non_torch, collapse = ", "))
      tryCatch({
        reticulate::py_install(non_torch, pip = TRUE)
      }, error = function(e) {
        stop("Failed to install packages ", paste(non_torch, collapse = ", "),
             ": ", conditionMessage(e))
      })
    }

    # Install PyTorch with CPU-only version
    if ("torch" %in% needs_install) {
      message("Installing PyTorch (CPU version)...")
      tryCatch({
        reticulate::py_install(
          "torch",
          pip = TRUE,
          pip_options = c("--index-url", "https://download.pytorch.org/whl/cpu")
        )
      }, error = function(e) {
        stop("Failed to install PyTorch: ", conditionMessage(e))
      })
    }
  }

  # Step 6: Verify all packages can be imported (not just available)
  message("Verifying package installation...")

  verification_code <- '
import sys
def verify_imports():
    results = {}

    # Test numpy
    try:
        import numpy
        results["numpy"] = {"status": "ok", "version": numpy.__version__}
    except Exception as e:
        results["numpy"] = {"status": "error", "error": str(e)}

    # Test sklearn
    try:
        import sklearn
        results["sklearn"] = {"status": "ok", "version": sklearn.__version__}
    except Exception as e:
        results["sklearn"] = {"status": "error", "error": str(e)}

    # Test joblib
    try:
        import joblib
        results["joblib"] = {"status": "ok", "version": joblib.__version__}
    except Exception as e:
        results["joblib"] = {"status": "error", "error": str(e)}

    # Test torch
    try:
        import torch
        results["torch"] = {"status": "ok", "version": torch.__version__}
    except Exception as e:
        results["torch"] = {"status": "error", "error": str(e)}

    return results

verification_results = verify_imports()
'

  tryCatch({
    reticulate::py_run_string(verification_code)
    results <- reticulate::py$verification_results

    # Check results
    failed <- character()
    for (pkg_name in names(results)) {
      pkg_result <- results[[pkg_name]]
      if (pkg_result$status == "ok") {
        message("[OK] ", pkg_name, " v", pkg_result$version, " loaded successfully")
      } else {
        failed <- c(failed, pkg_name)
        message("[FAIL] ", pkg_name, " failed: ", pkg_result$error)
      }
    }

    if (length(failed) > 0) {
      stop("Failed to import the following packages: ",
           paste(failed, collapse = ", "),
           ". Try reinstalling with: reticulate::py_install(c('",
           paste(failed, collapse = "', '"), "'), pip = TRUE)")
    }

  }, error = function(e) {
    stop("Package verification failed: ", conditionMessage(e))
  })

  message("All required Python packages are ready!")
  invisible(TRUE)
}

# =============================================================================
# Embedded Python
# =============================================================================

#' @keywords internal
.get_python_model_code <- function() {
  '
import torch
import torch.nn as nn
import joblib
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_model = None
_scaler_add = None
_scaler_tgt = None
_scaler_inc = None
_device = torch.device("cpu")

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

def load_model_components(model_path, scaler_add_path, scaler_tgt_path, scaler_inc_path=None):
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

def predict_sir_parameters(seq, additional_pair):
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

def cleanup_model():
    global _model, _scaler_add, _scaler_tgt, _scaler_inc
    _model = None
    _scaler_add = None
    _scaler_tgt = None
    _scaler_inc = None
    return True
'
}

# =============================================================================
# Helpers
# =============================================================================

#' @keywords internal
.validate_model_directory <- function(model_dir) {
  if (!dir.exists(model_dir))
    stop("Model directory does not exist: ", model_dir)

  base_dir <- normalizePath(model_dir, winslash = "/", mustWork = TRUE)
  file_paths <- list(
    model = file.path(base_dir, "model4_bilstm.pt"),
    scaler_add = file.path(base_dir, "scaler_additional.pkl"),
    scaler_tgt = file.path(base_dir, "scaler_targets.pkl"),
    scaler_inc = file.path(base_dir, "scaler_incidence.pkl")
  )

  # Check required files (scaler_inc is optional)
  required <- c("model", "scaler_add", "scaler_tgt")
  missing <- required[!vapply(file_paths[required], file.exists, logical(1))]
  if (length(missing))
    stop("Missing model files: ", paste(missing, collapse = ", "))

  file_paths
}

#' @keywords internal
.get_model_directory <- function(model_dir = NULL) {
  if (!is.null(model_dir)) return(model_dir)

  pkg_model_dir <- system.file("models", package = "epiworldRcalibrate")
  if (pkg_model_dir == "") stop("Model directory not found.")
  pkg_model_dir
}

#' @keywords internal
.validate_sir_inputs <- function(daily_cases, population_size, recovery_rate) {
  if (!is.numeric(daily_cases)) stop("daily_cases must be numeric")
  if (length(daily_cases) != 61)
    stop("daily_cases must have length 61")
  if (any(daily_cases < 0)) stop("daily_cases cannot be negative")
  if (!is.numeric(population_size) || length(population_size) != 1)
    stop("population_size must be single numeric")
  if (population_size <= 0) stop("population_size must be positive")
  if (!is.numeric(recovery_rate) || length(recovery_rate) != 1)
    stop("recovery_rate must be single numeric")
  if (recovery_rate <= 0) stop("recovery_rate must be positive")
}

# =============================================================================
# Core API
# =============================================================================

#' Initialize BiLSTM Model for SIR Parameter Estimation
#'
#' @param model_dir Optional path to model directory. Defaults to the
#'   package's bundled model files.
#' @param force_reload Logical; reload even if already loaded.
#'
#' @return Invisibly returns `TRUE` on success.
#' @export
init_bilstm_model <- function(model_dir = NULL, force_reload = FALSE) {

  model_dir <- .get_model_directory(model_dir)

  if (.bilstm_env$model_loaded &&
      !force_reload &&
      identical(.bilstm_env$model_dir, model_dir)) {
    message("Model already loaded.")
    return(invisible(TRUE))
  }

  file_paths <- .validate_model_directory(model_dir)
  .ensure_python_ready()

  reticulate::py_run_string(.get_python_model_code())

  # Load with optional scaler_inc
  scaler_inc_path <- if (file.exists(file_paths$scaler_inc)) {
    file_paths$scaler_inc
  } else {
    NULL
  }

  reticulate::py$load_model_components(
    model_path = file_paths$model,
    scaler_add_path = file_paths$scaler_add,
    scaler_tgt_path = file_paths$scaler_tgt,
    scaler_inc_path = scaler_inc_path
  )

  .bilstm_env$model_loaded <- TRUE
  .bilstm_env$model_dir <- model_dir
  message("BiLSTM model loaded successfully.")
  invisible(TRUE)
}

#' Estimate SIR Parameters from 61-day incidence
#'
#' @param daily_cases Numeric vector of length 61 containing daily incidence
#'   counts for days 0 to 60.
#' @param population_size Single numeric value giving the total population size
#'   used in the SIR model.
#' @param recovery_rate Single numeric value giving the recovery rate parameter
#'   of the SIR model.
#'
#' @return Named numeric vector: `ptran`, `crate`, `R0`.
#' @export
estimate_sir_parameters <- function(daily_cases, population_size, recovery_rate) {
  if (!.bilstm_env$model_loaded)
    stop("Model not loaded. Call init_bilstm_model().")

  .validate_sir_inputs(daily_cases, population_size, recovery_rate)

  out <- reticulate::py$predict_sir_parameters(
    as.numeric(daily_cases),
    list(as.numeric(population_size), as.numeric(recovery_rate))
  )
  names(out) <- c("ptran", "crate", "R0")

  # Recalculate crate based on R0, recovery_rate, and ptran
  out["crate"] <- out["R0"] * recovery_rate / out["ptran"]

  out
}

#' Calibrate SIR Parameters (one-step)
#'
#' This is a convenience wrapper that optionally initializes the BiLSTM model
#' and then calls [estimate_sir_parameters()] on the provided data.
#'
#' @param daily_cases Numeric vector of length 61 containing daily incidence
#'   counts (day 0 to day 60).
#' @param population_size Single numeric value giving the total population size.
#' @param recovery_rate Single numeric value giving the recovery rate parameter.
#' @param model_dir Optional path to the directory containing the trained
#'   BiLSTM model and scaler files. If `NULL`, the package's bundled assets
#'   are used.
#' @param auto_init Logical; if `TRUE` (default), automatically calls
#'   [init_bilstm_model()] when the model is not yet loaded.
#'
#' @return Named numeric vector: `ptran`, `crate`, `R0`.
#' @export
calibrate_sir <- function(daily_cases,
                          population_size,
                          recovery_rate,
                          model_dir = NULL,
                          auto_init = TRUE) {

  if (!.bilstm_env$model_loaded && auto_init)
    init_bilstm_model(model_dir)

  estimate_sir_parameters(daily_cases, population_size, recovery_rate)
}

# =============================================================================
# Utilities
# =============================================================================

#' Check model status
#'
#' @return A list describing model load status and Python config.
#' @export
check_model_status <- function() {
  list(
    loaded = .bilstm_env$model_loaded,
    model_directory = .bilstm_env$model_dir,
    python_config = tryCatch(
      utils::capture.output(reticulate::py_config()),
      error = function(e) NULL
    )
  )
}

#' Check Python and package installation status
#'
#' @return A list with Python installation details and package status
#' @export
check_python_setup <- function() {
  result <- list(
    python_available = FALSE,
    python_path = NULL,
    python_version = NULL,
    virtualenv = NULL,
    packages = list()
  )

  # Check if Python is available
  result$python_available <- tryCatch({
    reticulate::py_available(initialize = TRUE)
  }, error = function(e) FALSE)

  if (!result$python_available) {
    message("Python is not available")
    return(result)
  }

  # Get Python config
  py_config <- tryCatch(reticulate::py_config(), error = function(e) NULL)
  if (!is.null(py_config)) {
    result$python_path <- py_config$python
    result$python_version <- py_config$version
    result$virtualenv <- py_config$virtualenv
  }

  # Check packages
  packages <- c("numpy", "sklearn", "joblib", "torch")
  for (pkg in packages) {
    result$packages[[pkg]] <- list(
      available = tryCatch(
        reticulate::py_module_available(pkg),
        error = function(e) FALSE
      )
    )

    # Try to get version if available
    if (result$packages[[pkg]]$available) {
      version <- tryCatch({
        reticulate::import(pkg)$`__version__`
      }, error = function(e) "unknown")
      result$packages[[pkg]]$version <- version
    }
  }

  result
}

#' Reinstall all Python dependencies
#'
#' @param force Logical; if TRUE, removes and recreates the virtual environment
#' @return Invisibly returns TRUE on success
#' @export
reinstall_python_deps <- function(force = FALSE) {
  vname <- .bilstm_env$venv_name

  if (force) {
    message("Removing existing virtual environment...")
    envs <- tryCatch(reticulate::virtualenv_list(), error = function(e) character())
    if (vname %in% envs) {
      reticulate::virtualenv_remove(vname, confirm = FALSE)
    }
  }

  # Clean up current state
  if (.bilstm_env$model_loaded) {
    cleanup_model()
  }

  .bilstm_env$model_loaded <- FALSE
  .bilstm_env$model_dir <- NULL

  # Reinstall
  message("Setting up Python environment...")
  .ensure_python_ready()

  message("Python dependencies reinstalled successfully!")
  invisible(TRUE)
}

#' Unload the model from memory
#'
#' @return Invisibly `TRUE` on success.
#' @export
cleanup_model <- function() {
  if (!.bilstm_env$model_loaded) {
    message("No model loaded.")
    return(invisible(TRUE))
  }

  try(reticulate::py$cleanup_model(), silent = TRUE)
  .bilstm_env$model_loaded <- FALSE
  .bilstm_env$model_dir <- NULL
  message("Model cleaned up.")
  invisible(TRUE)
}
