from typing import Tuple
import pandas as pd
from sklearn import datasets

def load_iris() -> Tuple[pd.DataFrame, pd.Series]:
    ds = datasets.load_iris(as_frame=True)
    X, y = ds.data, ds.target
    y.name = "species"
    return X, y
