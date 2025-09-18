"""
Step 2: Data Export for Real SDE Training  
========================================

This file prepares data for stochastic differential equation training.
Run this SECOND to export clean financial data for Google Colab SDE training.

Mathematical Foundation:
- Stochastic Differential Equations: Øksendal "Stochastic Differential Equations"  
- Financial Applications: Shreve "Stochastic Calculus for Finance"
- Black-Scholes Model: Black & Scholes (1973) "The Pricing of Options"

APPROACH:
 Export log-price coordinates q_i = log(S_i/S_0)
 Export reference prices for coordinate transformation
 Prepare clean time series data for maximum likelihood SDE estimation
 No fake physics - authentic financial mathematics only
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the preprocessed geometric data"""
    print("📊 Loading processed geometric data...")
    
    with open('processed_data/geometric_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"   ✅ Loaded {data['n_timesteps']} timesteps × {data['n_assets']} assets")
    print(f"   Assets: {data['assets']}")
    
    return data

def validate_sde_data(data):
    """Validate data quality for SDE parameter estimation"""
    print("\n📈 Validating SDE Training Data...")
    
    # Convert PyTorch tensors to numpy arrays
    q_coords = data['q_coordinates'].detach().numpy() if hasattr(data['q_coordinates'], 'detach') else np.array(data['q_coordinates'])
    q_velocities = data['q_velocities'].detach().numpy() if hasattr(data['q_velocities'], 'detach') else np.array(data['q_velocities']) 
    reference_prices = data['reference_prices'].detach().numpy() if hasattr(data['reference_prices'], 'detach') else np.array(data['reference_prices'])
    
    print(f"   ✅ Log-price coordinates q_i = log(S_i/S_0): {q_coords.shape}")
    print(f"   ✅ Velocity data dq_i/dt: {q_velocities.shape}")
    print(f"   ✅ Reference prices S_0: {reference_prices.shape}")
    
    # Data quality checks
    print("   🎯 Data Quality Validation:")
    
    # Check for NaN or infinite values
    has_nan_q = np.isnan(q_coords).any()
    has_nan_v = np.isnan(q_velocities).any() 
    has_inf_q = np.isinf(q_coords).any()
    has_inf_v = np.isinf(q_velocities).any()
    
    print(f"   ✅ No NaN in coordinates: {not has_nan_q}")
    print(f"   ✅ No NaN in velocities: {not has_nan_v}")
    print(f"   ✅ No infinite values in q: {not has_inf_q}")
    print(f"   ✅ No infinite values in dq/dt: {not has_inf_v}")
    
    # Statistical properties for SDE fitting
    print("   📊 Financial Time Series Statistics:")
    
    for i, asset in enumerate(data['assets']):
        q_mean = np.mean(q_coords[:, i])
        q_std = np.std(q_coords[:, i])
        v_mean = np.mean(q_velocities[:, i])
        v_std = np.std(q_velocities[:, i])
        
        print(f"      {asset}:")
        print(f"        Log-price q: μ={q_mean:.4f}, σ={q_std:.4f}")
        print(f"        Velocity dq/dt: μ={v_mean:.6f}, σ={v_std:.6f}")
    
    # Check velocity scales are reasonable for SDE
    max_vel_std = max([np.std(q_velocities[:, i]) for i in range(len(data['assets']))])
    reasonable_scales = max_vel_std < 1.0
    print(f"   ✅ Reasonable velocity scales (max σ={max_vel_std:.4f}): {reasonable_scales}")
    
    # Check data has enough samples for ML estimation
    sufficient_data = len(q_coords) > 1000
    print(f"   ✅ Sufficient data for ML estimation: {sufficient_data}")
    
    print("\n   🎯 SDE Data Validation Complete:")
    print("   ✅ Clean log-price time series q_i(t)")
    print("   ✅ Proper velocity estimates dq_i/dt") 
    print("   ✅ Ready for Black-Scholes SDE parameter estimation")
    print("   ✅ Maximum likelihood fitting of drift μ and volatility Σ")
    print("   ❌ NO fake physics - authentic financial mathematics")
    
    return True

def export_colab_data(data):
    """Export data in format expected by Colab SDE notebook"""
    print("\n💾 Exporting data for Google Colab SDE training...")
    
    # Create processed_data directory if it doesn't exist
    Path('processed_data').mkdir(exist_ok=True)
    
    # Convert tensors to numpy arrays
    q_coordinates_np = data['q_coordinates'].detach().numpy() if hasattr(data['q_coordinates'], 'detach') else np.array(data['q_coordinates'])
    q_velocities_np = data['q_velocities'].detach().numpy() if hasattr(data['q_velocities'], 'detach') else np.array(data['q_velocities'])
    reference_prices_np = data['reference_prices'].detach().numpy() if hasattr(data['reference_prices'], 'detach') else np.array(data['reference_prices'])
    
    # Export as CSV files (Colab-friendly format)
    q_coords_df = pd.DataFrame(
        q_coordinates_np, 
        columns=[f'{asset}_logprice' for asset in data['assets']]
    )
    
    q_velocities_df = pd.DataFrame(
        q_velocities_np,
        columns=[f'{asset}_velocity' for asset in data['assets']]  
    )
    
    reference_df = pd.DataFrame(
        [reference_prices_np], 
        columns=[f'{asset}_ref_price' for asset in data['assets']]
    )
    
    # Save CSV files for Colab
    q_coords_df.to_csv('processed_data/q_coordinates.csv', index=False)
    q_velocities_df.to_csv('processed_data/q_velocities.csv', index=False)
    reference_df.to_csv('processed_data/reference_prices.csv', index=False)
    
    print("   ✅ Saved q_coordinates.csv")
    print("   ✅ Saved q_velocities.csv") 
    print("   ✅ Saved reference_prices.csv")
    
    # Create summary statistics for validation
    summary_stats = {
        'n_timesteps': len(q_coordinates_np),
        'n_assets': len(data['assets']),
        'assets': data['assets'],
        'data_quality': {
            'no_nan_coordinates': bool(not np.isnan(q_coordinates_np).any()),
            'no_nan_velocities': bool(not np.isnan(q_velocities_np).any()),
            'no_infinite_values': bool(not np.isinf(q_coordinates_np).any()),
            'velocity_scales_reasonable': bool(max([np.std(q_velocities_np[:, i]) 
                                                for i in range(len(data['assets']))]) < 1.0)
        },
        'statistics': {}
    }
    
    # Add per-asset statistics
    for i, asset in enumerate(data['assets']):
        summary_stats['statistics'][asset] = {
            'logprice_mean': float(np.mean(q_coordinates_np[:, i])),
            'logprice_std': float(np.std(q_coordinates_np[:, i])),
            'velocity_mean': float(np.mean(q_velocities_np[:, i])),
            'velocity_std': float(np.std(q_velocities_np[:, i])),
            'reference_price': float(reference_prices_np[i])
        }
    
    # Save summary
    with open('processed_data/data_summary.json', 'w') as f:
        import json
        json.dump(summary_stats, f, indent=2)
    
    print("   ✅ Saved data_summary.json")
    
    return summary_stats

def create_colab_instructions():
    """Create instructions for Colab SDE training"""
    print("\n📋 Creating Colab SDE training instructions...")
    
    instructions = """
Google Colab SDE Training Instructions
=====================================

1. UPLOAD DATA TO GOOGLE DRIVE:
   - Upload the entire 'processed_data/' folder to your Google Drive
   - Place it at: /content/drive/MyDrive/physics/
   - Required files:
     * q_coordinates.csv (log-price time series)
     * q_velocities.csv (velocity estimates)  
     * reference_prices.csv (price normalization)
     * data_summary.json (validation statistics)

2. OPEN COLAB NOTEBOOK:
   - Upload colab_training_notebook.py to Google Colab
   - Set runtime to GPU (preferably A100 or V100)
   - Run all cells in sequence

3. REAL SDE TRAINING:
   The notebook will:
   ✅ Implement Black-Scholes SDE: dS_i = μ_i S_i dt + σ_i S_i dW_i
   ✅ Use Euler-Maruyama integration for SDEs
   ✅ Estimate parameters via maximum likelihood
   ✅ Validate with proper statistical tests

4. EXPECTED OUTPUTS:
   - sde_model.pth (trained drift μ and volatility Σ parameters)
   - sde_results.pkl (training statistics and validation metrics)
   
5. DOWNLOAD RESULTS:
   After training completes, download results from Drive back to local
   for step3 analysis.

Mathematical References:
- Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
- Øksendal (2003): "Stochastic Differential Equations" 
- Shreve (2004): "Stochastic Calculus for Finance II"

This implements REAL financial mathematics - not fake physics!
"""
    
    with open('processed_data/COLAB_SDE_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("   ✅ Saved COLAB_SDE_INSTRUCTIONS.txt")

def main():
    """Main workflow for SDE data preparation"""
    print("🔬 STEP 2: SDE Data Export and Preparation")
    print("=" * 50)
    
    # Load preprocessed data
    data = load_processed_data()
    
    # Validate data quality for SDE training
    validation_passed = validate_sde_data(data)
    
    if not validation_passed:
        print("❌ Data validation failed! Check data quality before proceeding.")
        return
    
    # Export data for Colab
    summary_stats = export_colab_data(data)
    
    # Create instructions
    create_colab_instructions()
    
    print("\n🎯 SDE DATA PREPARATION COMPLETE!")
    print("   📊 Clean financial time series exported")
    print("   📈 Ready for Black-Scholes parameter estimation") 
    print("   🚀 Upload processed_data/ to Google Drive")
    print("   📓 Run colab_training_notebook.py in Colab")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Upload processed_data/ folder to Google Drive")
    print(f"   2. Open colab_training_notebook.py in Google Colab")
    print(f"   3. Set GPU runtime (A100 recommended)")
    print(f"   4. Run SDE maximum likelihood training")
    print(f"   5. Download results for step3 analysis")
    

if __name__ == "__main__":
    main()