import numpy as np
from typing import Union, List

# Type alias for array-like inputs
ArrayLike = Union[List[float], np.ndarray]

def calculate_mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Computes Mean Squared Error (MSE):
    MSE = (1/n) * sum((y_true - y_pred) ** 2)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_t - y_p) ** 2))

def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Computes Root Mean Squared Error (RMSE):
    RMSE = sqrt(MSE)
    """
    return float(np.sqrt(calculate_mse(y_true, y_pred)))

def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Computes Mean Absolute Error (MAE):
    MAE = (1/n) * sum(abs(y_true - y_pred))
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))

def calculate_r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Computes R-squared (Coefficient of Determination) score:
    R^2 = 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - mean(y_true)) ** 2))
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    
    if ss_tot == 0.0:
        return 0.0  # Avoid division by zero if all ground-truth values are constant
        
    return float(1.0 - (ss_res / ss_tot))

def print_regression_report(y_true: ArrayLike, y_pred: ArrayLike, title: str = "Regression Evaluation Report"):
    """
    Prints a formatted evaluation report for the regression metrics.
    """
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(f"Mean Absolute Error (MAE):    {mae:.6f}")
    print(f"Mean Squared Error (MSE):     {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R-squared Score (R^2):        {r2:.6f}")
    print("=" * len(title))

# Scikit-learn equivalents for reference/cross-validation
def compare_with_sklearn(y_true: ArrayLike, y_pred: ArrayLike):
    """
    Verifies metric outputs against standard scikit-learn library implementations.
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        custom_mse = calculate_mse(y_true, y_pred)
        custom_mae = calculate_mae(y_true, y_pred)
        custom_r2 = calculate_r2(y_true, y_pred)
        
        sk_mse = mean_squared_error(y_true, y_pred)
        sk_mae = mean_absolute_error(y_true, y_pred)
        sk_r2 = r2_score(y_true, y_pred)
        
        print("Comparison with scikit-learn:")
        print(f"  MSE:  Custom={custom_mse:.6f} | scikit-learn={sk_mse:.6f} (Diff: {abs(custom_mse - sk_mse):.2e})")
        print(f"  MAE:  Custom={custom_mae:.6f} | scikit-learn={sk_mae:.6f} (Diff: {abs(custom_mae - sk_mae):.2e})")
        print(f"  R^2:  Custom={custom_r2:.6f} | scikit-learn={sk_r2:.6f} (Diff: {abs(custom_r2 - sk_r2):.2e})")
    except ImportError:
        print("scikit-learn is not installed in the current environment.")
