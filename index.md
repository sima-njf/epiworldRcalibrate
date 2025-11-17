📦 epiworldRcalibrate BiLSTM-Based Parameter Calibration for epiworldR

epiworldRcalibrate provides fast, deep-learning–based calibration of SIR
model parameters from epidemic time series generated with epiworldR.

The package uses a pretrained Bidirectional LSTM (BiLSTM) neural network
to estimate:

Transmission rate (ptran)

Contact rate (crate)

Reproduction number (R0)

Directly from a single incidence curve.

🌟 Key Features

One-line calibration using calibrate_sir()

No Python setup needed — the model initializes internally

Works seamlessly with epiworldR simulations

Interpretable preprocessing via show_preprocessing()

Enables comparison of true vs calibrated SIR dynamics

🚀 Basic Usage library(epiworldR) library(epiworldRcalibrate)

model \<- ModelSIRCONN(“sim”, n = 8000, prevalence = 0.01, contact_rate
= 3, transmission_rate = 0.25, recovery_rate = 0.1) run(model, ndays =
60)

inc \<- plot_incidence(model)\[,1\]

calibrate_sir( daily_cases = inc, population_size = 8000, recovery_rate
= 0.1 )

📘 What It’s For

This package is designed for:

Epidemic modelers who want automatic SIR parameter recovery

Researchers comparing mechanistic vs. learned models

Students learning calibration methods for infectious disease modeling

Practitioners needing fast approximate inference from incidence data

🔗 Website

Full documentation and vignette: ➡️
<https://sima-njf.github.io/epiworldRcalibrate/>
