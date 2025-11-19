
# epiworldRcalibrate

<!-- badges: start -->
<a href="https://github.com/EpiForeSITE"><img src="https://github.com/EpiForeSITE/software/raw/e82ed88f75e0fe5c0a1a3b38c2b94509f122019c/docs/assets/foresite-software-badge.svg"></a>
<a href="https://github.com/sima-njf/epiworldRcalibrate/actions/workflows/r-cmd-check.yml"><img src="https://github.com/sima-njf/epiworldRcalibrate/actions/workflows/r-cmd-check.yml/badge.svg"></a>
<a href="https://github.com/sima-njf/epiworldRcalibrate/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

<!-- badges: end -->

### **Deep Learning Calibration for epiworldR (BiLSTM-Based)**

**epiworldRcalibrate** provides fast, data-driven calibration of SIR epidemic parameters using a pretrained **Bidirectional LSTM (BiLSTM)** model.
Given a single incidence time series, the package estimates:

* **Transmission rate** (`ptran`)
* **Contact rate** (`crate`)
* **Basic reproduction number** (`R0`)

The package is fully integrated with **epiworldR** and requires **no external Python setup**.

---

## 🚀 Features

* **One-line calibration** via `calibrate_sir()`
* **Automatic deep learning backend** (initialized on demand)
* **Compatible with all epiworldR SIR simulations**
* **Transparent preprocessing** via `show_preprocessing()`
* **Designed for reproducible epidemic modeling workflows**

---

## 📦 Installation

```r
# Install from GitHub
devtools::install_github("sima-njf/epiworldRcalibrate")
```

---

## 🔧 Quick Example

```r
library(epiworldR)
library(epiworldRcalibrate)

# simulate SIR model
m <- ModelSIRCONN("sim", n=8000, prevalence=0.01,
                  contact_rate=3, transmission_rate=0.25,
                  recovery_rate=0.1)
run(m, ndays = 60)

inc <- plot_incidence(m)[,1]

# one-line calibration
calibrate_sir(
  daily_cases = inc,
  population_size = 8000,
  recovery_rate = 0.1
)
```

---

## 🎯 What This Package Is For

Use **epiworldRcalibrate** when you want to:

* Extract SIR parameters **directly from simulated incidence curves**
* Compare **ground-truth vs. calibrated** dynamics
* Avoid heavy Bayesian/likelihood-based fitting
* Teach or study calibration in infectious disease modeling
* Perform **fast approximate inference** in simulation studies

---

## 📘 Documentation

Full website, reference, and vignette:
👉 **[https://sima-njf.github.io/epiworldRcalibrate/](https://sima-njf.github.io/epiworldRcalibrate/)**

---

## 👤 Author

Developed by **Sima Najafzadehkhoei**
🔗 [https://github.com/sima-njf](https://github.com/sima-njf)


