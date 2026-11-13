from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CLASSES = ["Low", "Medium", "High"]  # ordinal: 0, 1, 2
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}

NUM_COLS = [
    "Age",
    "Monthly_Income_KES",
    "Credit_Score",
    "Total_BNPL_Active_Loans",
    "Total_BNPL_Debt_KES",
    "Average_Transaction_Value_KES",
    "Debt_to_Income",
]
CAT_COLS = ["Employment_Status", "Late_Payment_History", "Shopping_Category_Most_Frequent"]
FEATURES = NUM_COLS + CAT_COLS
RAW_INPUT_COLS = [c for c in FEATURES if c != "Debt_to_Income"]

EMPLOYMENT_OPTIONS = ["Employed", "Freelancer", "Student", "Unemployed"]
SHOPPING_OPTIONS = ["Electronics", "Fashion", "Groceries/Essentials", "Home/Furniture", "Travel"]
YES_NO = ["No", "Yes"]

FEATURE_LABELS = {
    "Age": "Age",
    "Monthly_Income_KES": "Monthly income",
    "Credit_Score": "Credit score",
    "Total_BNPL_Active_Loans": "Active BNPL loans",
    "Total_BNPL_Debt_KES": "Outstanding BNPL debt",
    "Average_Transaction_Value_KES": "Average purchase value",
    "Debt_to_Income": "Debt relative to income",
    "Employment_Status": "Employment status",
    "Late_Payment_History": "Late payment history",
    "Shopping_Category_Most_Frequent": "Most frequent purchase category",
}

TIER_COLORS = {"Low": "#2FA36B", "Medium": "#E9A23B", "High": "#D1453B"}

POLICY = {
    "Low": {
        "limit_pct": 0.40,
        "action": "Approve at standard terms. Monitor repayment as usual and review the limit "
        "after three clean repayment cycles.",
    },
    "Medium": {
        "limit_pct": 0.20,
        "action": "Approve with a reduced limit and shorter repayment cycles. Do not raise the "
        "limit automatically. Re-score after the next two payments.",
    },
    "High": {
        "limit_pct": 0.05,
        "action": "Decline or offer a starter limit only. Verify income, and send to a credit "
        "officer for manual review before any approval.",
    },
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Input must contain the raw input columns."""
    out = df.copy()
    income = pd.to_numeric(out["Monthly_Income_KES"], errors="coerce").clip(lower=1)
    debt = pd.to_numeric(out["Total_BNPL_Debt_KES"], errors="coerce")
    out["Debt_to_Income"] = debt / income
    return out


def age_band(age: pd.Series) -> pd.Series:
    bins = [0, 24, 34, 44, 54, 200]
    labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
    return pd.cut(age, bins=bins, labels=labels)



@dataclass
class Artifacts:
    pipeline: object
    calibrator: object
    meta: dict

    @property
    def thresholds(self) -> dict:
        return self.meta["thresholds"]

    @property
    def model_name(self) -> str:
        return self.meta["model_name"]


def load_artifacts(base: str | Path) -> Artifacts:
    base = Path(base)
    pipeline = joblib.load(base / "models" / "risk_model.joblib")
    calibrator = joblib.load(base / "models" / "calibrator.joblib")
    meta = json.loads((base / "models" / "meta.json").read_text())
    return Artifacts(pipeline, calibrator, meta)


def calibrate(calibrator, raw_proba: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(raw_proba, 1e-6, 1.0))
    return calibrator.predict_proba(z)


def assign_tier(p_medium: np.ndarray, p_high: np.ndarray, thresholds: dict) -> np.ndarray:
    score = np.asarray(p_medium) + np.asarray(p_high)
    return np.where(
        np.asarray(p_high) >= thresholds["high"],
        "High",
        np.where(score >= thresholds["medium"], "Medium", "Low"),
    )


def score_customers(art: Artifacts, df: pd.DataFrame) -> pd.DataFrame:
    X = add_features(df)[FEATURES]
    raw = art.pipeline.predict_proba(X)
    p = calibrate(art.calibrator, raw)
    out = pd.DataFrame(
        {"P_Low": p[:, 0], "P_Medium": p[:, 1], "P_High": p[:, 2]}, index=df.index
    )
    out["Risk_Score"] = 100 * (out["P_Medium"] + out["P_High"])
    out["Risk_Tier"] = assign_tier(out["P_Medium"].values, out["P_High"].values, art.thresholds)
    return out


def _original_feature(encoded_name: str) -> str:
    name = encoded_name.split("__", 1)[-1]
    for col in NUM_COLS:
        if name == col:
            return col
    for col in CAT_COLS:
        if name.startswith(col + "_"):
            return col
    return name


def explain(art: Artifacts, df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-customer feature contributions (exact TreeSHAP from XGBoost).

    Positive values push the customer toward the Medium/High classes and away from Low.
    Returns None if the deployed model is not XGBoost.
    """
    model = art.pipeline.named_steps["model"]
    if not hasattr(model, "get_booster"):
        return None
    import xgboost as xgb

    X = add_features(df)[FEATURES]
    prep = art.pipeline.named_steps["prep"]
    Xt = prep.transform(X)
    contribs = model.get_booster().predict(xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)
    low, med, high = contribs[:, 0, :-1], contribs[:, 1, :-1], contribs[:, 2, :-1]
    pressure = 0.5 * ((med - low) + (high - low))
    names = [_original_feature(n) for n in prep.get_feature_names_out()]
    frame = pd.DataFrame(pressure, columns=names, index=df.index)
    return frame.T.groupby(level=0).sum().T[FEATURES]



def business_curve(
    risk_score: np.ndarray,
    actual: np.ndarray,
    cutoffs: np.ndarray,
    exposure: float,
    lgd: float,
    margin: float,
    medium_weight: float,
) -> pd.DataFrame:
    """Expected portfolio economics when approving customers whose risk score is below a cutoff.

    `actual` holds the observed risk class (Low/Medium/High). A High customer is treated as a
    full default, a Medium customer as a default with probability `medium_weight`.
    """
    score = np.asarray(risk_score, dtype=float)
    w = pd.Series(actual).map({"Low": 0.0, "Medium": float(medium_weight), "High": 1.0}).values
    per_customer = exposure * ((1 - w) * margin - w * lgd)
    loss_each = exposure * lgd * w
    order = np.argsort(score)
    score_sorted = score[order]
    cum_profit = np.cumsum(per_customer[order])
    cum_loss = np.cumsum(loss_each[order])
    cum_bad_w = np.cumsum(w[order])
    n = len(score)
    rows = []
    for c in cutoffs:
        k = int(np.searchsorted(score_sorted, c, side="left"))
        approved = k
        rows.append(
            {
                "Cutoff": float(c),
                "Approved": approved,
                "Approval_Rate": approved / n,
                "Expected_Profit": float(cum_profit[k - 1]) if k > 0 else 0.0,
                "Expected_Loss": float(cum_loss[k - 1]) if k > 0 else 0.0,
                "Bad_Rate": float(cum_bad_w[k - 1] / k) if k > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def kes(x: float, decimals: int = 0) -> str:
    """Format a number as Kenya shillings."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"KSh {x:,.{decimals}f}"


def kes_short(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return f"KSh {x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"KSh {x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"KSh {x / 1e3:.0f}K"
    return f"KSh {x:,.0f}"
