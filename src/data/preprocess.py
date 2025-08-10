from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def build_preprocessor(scale_numeric: bool = True):
    steps = []
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps) if steps else None
