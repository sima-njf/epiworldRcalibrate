"""Compatibility loader for the original pretrained SIR BiLSTM."""

from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn


class ExistingSIRBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=160,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(322, 64)
        self.fc2 = nn.Linear(64, 3)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, known):
        _, (h, _) = self.bilstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = torch.cat((h, known), dim=1)
        h = torch.relu(self.fc1(h))
        out = self.fc2(h)
        return torch.stack([
            self.sigmoid(out[:, 0]),
            self.softplus(out[:, 1]),
            self.softplus(out[:, 2]),
        ], dim=1)


def predict_pretrained_sir(epicurve, population_size, recovery_rate, model_dir):
    model_dir = Path(model_dir)

    model = ExistingSIRBiLSTM()
    state = torch.load(model_dir / "model4_bilstm.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    incidence_scaler = joblib.load(model_dir / "scaler_incidence.pkl")
    additional_scaler = joblib.load(model_dir / "scaler_additional.pkl")
    target_scaler = joblib.load(model_dir / "scaler_targets.pkl")

    curve = np.asarray(epicurve, dtype=np.float32).reshape(1, -1)
    if curve.shape[1] != 61:
        raise ValueError("The original pretrained SIR model requires exactly 61 incidence values.")

    curve = incidence_scaler.transform(curve).reshape(1, 61, 1)
    known = additional_scaler.transform(
        np.asarray([[population_size, recovery_rate]], dtype=np.float32)
    )

    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(curve, dtype=torch.float32),
            torch.tensor(known, dtype=torch.float32),
        ).numpy()

    pred = target_scaler.inverse_transform(pred_scaled)[0]
    return {
        "ptran": float(pred[0]),
        "crate": float(pred[1]),
        "R0": float(pred[2]),
    }
