import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_optimization_models(csv_path="data/video_dataset.csv", model_dir="models"):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    X = df[['si_mean', 'si_std', 'ti_mean', 'ti_std']]
    y_qp = df['target_qp']
    y_gop = df['target_gop']

    model_qp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_gop = RandomForestRegressor(n_estimators=100, random_state=42)

    model_qp.fit(X, y_qp)
    model_gop.fit(X, y_gop)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model_qp, os.path.join(model_dir, 'model_qp.pkl'))
    joblib.dump(model_gop, os.path.join(model_dir, 'model_gop.pkl'))

if __name__ == "__main__":
    train_optimization_models()