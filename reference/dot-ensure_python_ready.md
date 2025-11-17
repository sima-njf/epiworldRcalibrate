# Ensure Python is usable and required modules are present.

- Respects RETICULATE_PYTHON if the user has set it.

- Otherwise creates/uses a private virtualenv 'epiworldRcalibrate'

- Installs numpy, joblib, torch (CPU) if missing.

Ensure Python is usable and required modules are present.

- Respects RETICULATE_PYTHON if the user has set it.

- Otherwise creates/uses a private virtualenv 'epiworldRcalibrate'

- Installs numpy, joblib, torch (CPU) if missing.

## Usage

``` r
.ensure_python_ready()
```
