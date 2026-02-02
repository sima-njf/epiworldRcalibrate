
library(reticulate)
library(data.table)


# Point to your Python environment
# use_python("/usr/bin/python3")  # Uncomment if needed

# Import Python packages
torch <- import("torch")
torch_nn <- import("torch.nn")
np <- import("numpy")
joblib <- import("joblib")
os <- import("os")
# --------------------------
# 1. Fix the path - use path.expand() for ~
# --------------------------
OUTPUT_DIR <- path.expand("~/sima/epiworldRcalibrate_SEIR/model")

# Check if path exists and see what files are there
cat("Path exists:", dir.exists(OUTPUT_DIR), "\n")
cat("Files in directory:\n")
print(list.files(OUTPUT_DIR))
# --------------------------
# 2. Load scalers (path is now correct)
# --------------------------
scaler_incidence  <- joblib$load(file.path(OUTPUT_DIR, "scaler_incidence_seir2.pkl"))
scaler_additional <- joblib$load(file.path(OUTPUT_DIR, "scaler_additional_seir2.pkl"))
scaler_targets    <- joblib$load(file.path(OUTPUT_DIR, "scaler_targets_seir2.pkl"))

cat("✅ Scalers loaded!\n")

# --------------------------
# 3. Load model in Python via reticulate
# --------------------------
py_run_string(paste0("
import torch
import torch.nn as nn
import joblib

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 additional_dim, output_dim, dropout):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(2 * hidden_dim + additional_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, add_inputs):
        _, (h_n, _) = self.bilstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h = torch.relu(self.fc1(torch.cat((h, add_inputs), dim=1)))
        out = self.fc2(h)
        out = torch.stack([
            self.sigmoid(out[:, 0]),
            self.softplus(out[:, 1]),
            self.softplus(out[:, 2]),
        ], dim=1)
        return out

OUTPUT_DIR = '", OUTPUT_DIR, "'

# Load checkpoint and detect hidden_dim
checkpoint = torch.load(OUTPUT_DIR + '/model4_bilstm_seir2.pt', map_location='cpu')
hidden_dim = checkpoint['bilstm.weight_ih_l0'].shape[0] // 4
print(f'Detected hidden_dim: {hidden_dim}')

# Load model
model = BiLSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=3,
                    additional_dim=3, output_dim=3, dropout=0.5)
model.load_state_dict(checkpoint)
model.eval()

# Load scalers
scaler_incidence  = joblib.load(OUTPUT_DIR + '/scaler_incidence_seir2.pkl')
scaler_additional = joblib.load(OUTPUT_DIR + '/scaler_additional_seir2.pkl')
scaler_targets    = joblib.load(OUTPUT_DIR + '/scaler_targets_seir2.pkl')

print('Model and scalers loaded successfully!')
"))

cat("✅ Model loaded!\n")

# --------------------------
# 4. Prediction function
# --------------------------
predict_params <- function(incidence_vec, n, incub, recov) {

  py_run_string(paste0("
import numpy as np
import torch

incidence_vec = np.array([", paste(incidence_vec, collapse=", "), "])

X_scaled = scaler_incidence.transform(incidence_vec.reshape(1, -1))
X_scaled = X_scaled.reshape(1, ", length(incidence_vec), ", 1)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

add_input = np.array([[", n, ", ", incub, ", ", recov, "]])
add_scaled = scaler_additional.transform(add_input)
add_tensor = torch.tensor(add_scaled, dtype=torch.float32)

with torch.no_grad():
    pred_scaled = model(X_tensor, add_tensor).numpy()
    pred = scaler_targets.inverse_transform(pred_scaled)

ptran = float(pred[0, 0])
crate = float(pred[0, 1])
R0    = float(pred[0, 2])
  "))

  data.table(
    ptran = py$ptran,
    crate = py$crate,
    R0    = py$R0
  )
}

# --------------------------
# 5. Run prediction
# --------------------------
incidence_vec <- c(
  103, 37, 60, 74, 108, 125, 138, 186, 215, 276, 318, 331, 414, 402, 446,
  454, 405, 401, 373, 334, 285, 241, 219, 156, 140, 108, 93, 82, 73, 78,
  48, 38, 34, 22, 27, 22, 20, 11, 14, 8, 11, 14, 8, 6, 6, 2, 0, 7, 2, 4,
  4, 1, 1, 2, 2, 1, 0, 2, 1, 1, 0
)

result <- predict_params(
  incidence_vec = incidence_vec,
  n     = 7087,
  incub = 7.5,
  recov = 0.203
)

cat("\n==============================================\n")
cat("Predicted Parameters:\n")
cat("==============================================\n")
cat(sprintf("  ptran: %.6f\n", result$ptran))
cat(sprintf("  crate: %.4f\n", result$crate))
cat(sprintf("  R0:    %.4f\n", result$R0))

# Verify R0
calculated_R0 <- (result$ptran * result$crate) / 0.203
cat("\nVerification:\n")
cat(sprintf("  R0 from model:  %.4f\n", result$R0))
cat(sprintf("  R0 calculated:  %.4f\n", calculated_R0))
cat(sprintf("  Difference:     %.6f\n", abs(result$R0 - calculated_R0)))

