# Cryptocurrency SDE Analysis System

A sophisticated mathematical finance project implementing Stochastic Differential Equations (SDEs) for cryptocurrency price modeling. Uses advanced models including Jump-Diffusion processes, Heston Stochastic Volatility, and Bayesian Parameter Uncertainty Quantification.

Download data from https://drive.google.com/drive/folders/1Bx5W9At5UhXaFVLex58CwEmtr5vcf1RU?usp=sharing 
## **Project Overview**

### **Mathematical Foundation**
Implements the Jump-Diffusion SDE:
```
dq_i(t) = μ_i dt + σ_ij dW_j + J_i dN_i(t)
```

Where:
- `q_i = log(S_i/S_0)` - Log-price coordinates
- `μ_i` - Drift parameters (expected returns) 
- `σ_ij` - Volatility matrix (correlation structure)
- `J_i dN_i` - Jump-diffusion process (sudden price movements)

### **Advanced Features**
- **Heston Stochastic Volatility**: `dv_t = κ(θ - v_t)dt + ξ√v_t dZ_t`
- **Bayesian Uncertainty Quantification**: Variational inference with ELBO optimization
- **GPU Training**: Google Colab with CUDA acceleration
- **Euler-Maruyama Integration**: Proper SDE numerical methods

## **File Structure**

### **Core Pipeline (4 Steps)**
1. **`step1_data_preprocessing.py`** - Convert raw crypto data to geometric coordinates
2. **`step2_sde_data_export.py`** - Export processed data for Google Colab training  
3. **`colab_training_notebook.py`** - Google Colab GPU training (918 lines)
4. **`step3_CORRECTED_analysis.py`** - Analysis of trained model results

### **Configuration Files**
- **`config.yaml`** - Model parameters and training configuration
- **`requirements.txt`** - Python dependencies (PyTorch, NumPy, etc.)

### **Data Directories**
- **`crypto_data/`** - Raw 1-minute OHLCV data (2017-2025, ~4M records per asset)
  - BTC, ETH, SOL, DOGE, BNB, XRP klines from Binance
- **`processed_data/`** - Geometric coordinates and CSV exports for Colab
- **`colab_results/`** - Trained model outputs and analysis reports
- **`v1/`** - Python virtual environment

## **Usage Workflow**

### **Step 1: Data Preprocessing**
```bash
python step1_data_preprocessing.py
```
**What it does:**
- Loads raw CSV files from `crypto_data/`
- Extracts close prices for 6 cryptocurrencies
- Synchronizes to common length (~2.67M timesteps)
- Converts to log coordinates: `q_i(t) = log(S_i(t)/S_i(0))`
- Calculates velocities: `dq/dt` via finite differences
- **Outputs:** `geometric_data.pkl` + CSV files for Colab

### **Step 2: Data Export**
```bash
python step2_sde_data_export.py
```
**What it does:**
- Validates data quality (no NaN, reasonable scales)
- Re-exports data as CSV with proper column names
- Creates `data_summary.json` with validation statistics
- Generates `COLAB_SDE_INSTRUCTIONS.txt`
- **Outputs:** Clean CSV files ready for Google Colab

### **Step 3: Google Colab Training**
1. Upload `processed_data/` folder to Google Drive
2. Open `colab_training_notebook.py` in Google Colab
3. Set GPU runtime (A100/V100 recommended)
4. Run all cells to train SDE model

**What it does:**
- Loads CSV data from Google Drive
- Implements `JumpDiffusionDynamics` class with all parameters
- Trains via Maximum Likelihood Estimation using `EulerMaruyamaIntegrator`
- Uses `VariationalLoss` for Bayesian uncertainty quantification
- **Outputs:** `advanced_sde_model.pth`, `advanced_training_results.pkl`

### **Step 4: Results Analysis**
```bash
python step3_CORRECTED_analysis.py
```
**What it does:**
- Loads trained model from `colab_results/`
- Analyzes learned SDE parameters
- Calculates annual volatility estimates
- Generates trading signal assessments
- **Outputs:** Comprehensive analysis with uncertainty quantification

## 📈 **Model Components**

### **Core Classes**
- **`JumpDiffusionDynamics`** - Main SDE model with learnable parameters
- **`EulerMaruyamaIntegrator`** - Numerical SDE integration 
- **`LogLikelihoodLoss`** - Maximum likelihood objective
- **`VariationalLoss`** - Bayesian uncertainty quantification

### **Parameters Learned**
- **Drift rates (μ)**: Expected log-returns per minute
- **Volatility matrix (Σ)**: Cholesky-factorized covariance  
- **Jump parameters**: Intensity (λ), mean (μ_J), std (σ_J)
- **Heston parameters**: Mean reversion (κ), long-term variance (θ), vol-of-vol (ξ)
- **Bayesian posteriors**: Parameter uncertainty distributions

## **Actual Results**

### **Training Metrics** 
- **Final Loss:** -35.996 (converged)
- **Numerical Stability:** Condition number 4.25 (excellent)
- **Training Data:** 80% of ~2.67M timesteps

### **Model Outputs**
| Asset | Expected Return | Volatility | Confidence | Recommendation |
|-------|----------------|------------|------------|----------------|
| BTC   | +42.4%         | 73.3%      | 50%        | UNCERTAIN      |
| ETH   | -38.0%         | 73.6%      | 50%        | UNCERTAIN      |
| SOL   | +37.4%         | 125.4%     | 50%        | UNCERTAIN      |
| DOGE  | -37.6%         | 128.2%     | 50%        | UNCERTAIN      |
| BNB   | -13.9%         | 74.1%      | 50%        | UNCERTAIN      |
| XRP   | +50.8%         | 93.9%      | 50%        | UNCERTAIN      |

### **Parameter Uncertainty**
- **Drift uncertainty:** ±100× larger than signal
- **All predictions:** Marked as "UNCERTAIN"
- **Model confidence:** 50% (random guessing level)


## **Technical Requirements**

### **Dependencies**
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0
```

### **Hardware**
- **Local**: CPU sufficient for preprocessing
- **Training**: GPU required (Google Colab A100/V100)
- **Memory**: Large datasets (~4M records per asset)

### **Environment**
- Python 3.12+ 
- Virtual environment (`v1/` directory)
- Google Colab account for GPU training




## **Academic Value**

This project excellently demonstrates:
- **Proper SDE Implementation**: Mathematically sound approach
- **Uncertainty Quantification**: Bayesian methods showing model limitations  
- **Scientific Honesty**: Model admits when it cannot make predictions
- **Advanced Mathematics**: Jump-diffusion, stochastic volatility, variational inference

The system's greatest strength is **mathematical integrity** - it correctly implements sophisticated models and honestly reports when they fail to provide actionable insights.


---

*Last Updated: September 2025*  
*Data Period: 2017-2025*  
*Assets: BTC, ETH, SOL, DOGE, BNB, XRP*
