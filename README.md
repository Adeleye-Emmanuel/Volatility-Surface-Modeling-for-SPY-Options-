# 🧠 Volatility Surface Modeling for SPY Options 

This project develops an end-to-end framework for modeling the **implied volatility surface** of SPY (S&P 500 ETF) options using both **parametric** and **non-parametric (machine learning)** methods.

The IV surface is a critical tool in options pricing, capturing how market-expected volatility varies across strike prices and maturities. This notebook uses real market data to:

- Visualize implied volatility patterns (smile, skew, term structure)
- Fit the SVI model to extract an arbitrage-free volatility surface
- Train machine learning models to approximate the surface
- Analyze local behavior through gradient extraction from neural networks

---

## 📁 Project Structure

### 1. 🔍 **Data Fetching & Cleaning**

- Fetched SPY options data from a local options dataset
- Computed:
  - **Log-moneyness**: \( \log(K / S) \)
  - **Time to maturity** (TTM): in years
- Filtered invalid or noisy observations, especially those with:
  - Implausibly high IVs (>2.0)
  - Near-zero TTM
- Final dataset contained ~1,100 clean observations

---

### 2. 📊 **Volatility Smile and Surface Visualization**

#### Single Maturity:
- Plotted **IV vs. Strike** and **IV vs. Log-moneyness**
- Observed the classic smile pattern:
  - Higher IVs for deep ITM calls and OTM puts

#### Multiple Maturities:
- Created subplot grid of IV vs strike for all expiries
- Identified high IV spikes and traced them to specific contracts
- Found potential causes:
  - Illiquid OTM contracts
  - Stale or sparse bid/ask data

---

### 3. 🔺 **SVI Model Fitting**

- Parameterized total implied variance as:

  \[
  w(k) = a + b\left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)
  \]

- Fit the SVI model **per expiry**
- Applied no-arbitrage constraints:
  - \( b > 0 \), \( \sigma > 0 \), \( |\rho| < 1 \)
- Filtered & retained expiries with good SVI fits
- Interpolated fitted parameters across maturity to reconstruct full 2D surface

---

### 4. 📈 **SVI Surface Visualization**

- Plotted the SVI-derived surface: \( \text{IV} = \sqrt{w(k)/T} \)
- Captured skew and curvature that aligns with market observations

---

### 5. 🤖 **Machine Learning Surface Fitting**

#### Random Forest Regressor:
- Trained a `RandomForestRegressor` to learn:

  \[
  (\log(K/F), T) \rightarrow \text{IV}
  \]

- Used real market IV data (not synthetic or SVI-generated)
- RMSE used for performance tracking
- Produced a surface close to market IVs

#### Neural Network:
- Created a feedforward neural network (2 hidden layers, ReLU)
- Trained on scaled features (`log_moneyness`, `TTM`)
- Output = predicted implied volatility
- Resulted in a **smooth, differentiable surface**

---

### 6. 🧮 **Gradient Analysis with Neural Network**

- Used TensorFlow's `GradientTape` to extract:

  - \( \frac{\partial \text{IV}}{\partial \log(K/F)} \): Local skew
  - \( \frac{\partial \text{IV}}{\partial T} \): Term structure slope

- Plotted both partial derivative surfaces:
  - **∂IV/∂k** shows skew dynamics across time/moneyness
  - **∂IV/∂T** shows steepening/flattenting term behavior

---

### 7. 📌 Conclusion

- Built an arbitrage-aware volatility surface using SVI
- Trained machine learning models on **real market IVs**
- Used gradients to interpret skew and term structure behavior
- Created a modular and reproducible notebook for volatility surface research

---

## 🛠 Tools & Libraries

- Python, NumPy, pandas
- Matplotlib & seaborn
- SciPy (for SVI fitting)
- scikit-learn (Random Forest)
- TensorFlow/Keras (Neural Network & Gradient Analysis)

---

## 📊 Output Visuals

- IV Smile (Call/Put)
- Term Structure of ATM IVs
- Volatility Surface: SVI, RF, Neural Net
- Gradient Surfaces (Skew, Term Structure)

---

## 🧾 Author Notes

This project was built to understand the **behavioral patterns of implied volatility** using real market data and a mix of **model-driven and data-driven approaches**. It provides deep insight into option pricing dynamics and practical IV surface fitting.
