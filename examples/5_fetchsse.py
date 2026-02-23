import numpy as np
import pandas as pd 

def main():
    path = "results/calibration_NVDA_20260128_222958_prices.csv"
    df = pd.read_csv(path)

    # Filtering logic
    df_filt = df 
    print(f"Dropped rows: {np.size(df) - np.size(df_filt)}")

    # Price Space RMSE
    rmse_a = np.sqrt(np.mean(df_filt["Err_A"]**2))
    rmse_mc = np.sqrt(np.mean(df_filt["Err_MC"]**2))

    # Weighted SSE (Objective Function Space)
    denominator = (1e-5 + df_filt["Mkt"])**2
    rel_sse_a = np.sum(df_filt["Err_A"]**2 / denominator)
    rel_sse_mc = np.sum(df_filt["Err_MC"]**2 / denominator)

    # Implied Volatility Metrics
    rmse_iv_mc = np.sqrt(np.mean((df_filt["IV_MC"] - df_filt["IV_Mkt"])**2))

    # Output
    results = {
        "RMSE_A": rmse_a,
        "RMSE_MC": rmse_mc,
        "RelSSE_A": rel_sse_a,
        "RelSSE_MC": rel_sse_mc,
        "RMSE_IV_MC": rmse_iv_mc
    }

    for key, val in results.items():
        print(f"{key:<12}: {val:.6f}")

if __name__ == "__main__":
    main()