"""
Step 1: Data Preprocessing for Geometric Neural Networks
=====================================================

This file loads raw crypto data and converts it to geometric coordinates.
Run this FIRST to prepare data for physics training.

Outputs: processed_data.pkl (for local use) and CSV files (for Colab)
"""

import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_crypto_data():
    """Load all cryptocurrency OHLCV data"""
    print("📊 Loading cryptocurrency data...")
    
    data_path = Path("crypto_data")
    
    # File mapping
    files = {
        'BTC': 'BTC_BTCUSDT_1m_klines.csv',
        'ETH': 'ETH_ETHUSDT_1m_klines.csv', 
        'SOL': 'SOL_SOLUSDT_1m_klines.csv',
        'DOGE': 'DOGE_DOGEUSDT_1m_klines.csv',
        'BNB': 'BNB_BNBUSDT_1m_klines.csv',
        'XRP': 'XRP_XRPUSDT_1m_klines.csv'
    }
    
    all_data = {}
    
    for asset, filename in files.items():
        filepath = data_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Standardize column names
            if 'close_price' not in df.columns and 'close' in df.columns:
                df['close_price'] = df['close']
            elif 'Close' in df.columns:
                df['close_price'] = df['Close']
            
            all_data[asset] = df
            print(f"   ✅ {asset}: {len(df)} records")
        else:
            print(f"   ❌ {asset}: File not found - {filepath}")
    
    return all_data

def synchronize_data(all_data):
    """Synchronize all assets to same time indices"""
    print("\n🔄 Synchronizing time series...")
    
    # Find common time range
    min_length = min(len(df) for df in all_data.values())
    print(f"   Common length: {min_length} timesteps")
    
    # Extract close prices and align
    synchronized_prices = {}
    for asset, df in all_data.items():
        # Take the last min_length records (most recent data)
        prices = df['close_price'].iloc[-min_length:].values
        synchronized_prices[asset] = prices
    
    # Convert to DataFrame
    price_df = pd.DataFrame(synchronized_prices)
    print(f"   ✅ Synchronized data shape: {price_df.shape}")
    
    return price_df

def convert_to_geometric_coordinates(price_df):
    """Convert prices to geometric manifold coordinates"""
    print("\n🌐 Converting to geometric coordinates...")
    
    # Convert to torch tensors
    prices_tensor = torch.tensor(price_df.values, dtype=torch.float32)
    reference_prices = prices_tensor[0]  # First timestamp as reference
    
    # Geometric coordinates: q_i(t) = log(S_i(t) / S_i(0))
    q_coords = torch.log(prices_tensor / (reference_prices + 1e-8))
    
    # Velocities: dq/dt (finite difference)
    q_velocities = torch.diff(q_coords, dim=0)
    
    print(f"   Position coordinates q: {q_coords.shape}")
    print(f"   Velocity coordinates dq/dt: {q_velocities.shape}")
    print(f"   Coordinate ranges: [{q_coords.min():.3f}, {q_coords.max():.3f}]")
    
    return q_coords, q_velocities, reference_prices

def save_processed_data(q_coords, q_velocities, reference_prices, price_df):
    """Save processed data for both local and Colab use"""
    print("\n💾 Saving processed data...")
    
    # Create output directory
    Path("processed_data").mkdir(exist_ok=True)
    
    # 1. For local Python use (pickle)
    processed_data = {
        'q_coordinates': q_coords,
        'q_velocities': q_velocities,  
        'reference_prices': reference_prices,
        'original_prices': price_df,
        'assets': list(price_df.columns),
        'n_timesteps': len(q_coords),
        'n_assets': len(price_df.columns)
    }
    
    with open('processed_data/geometric_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    print("   ✅ Saved: processed_data/geometric_data.pkl")
    
    # 2. For Colab use (CSV files)
    # Save coordinates as CSV
    q_df = pd.DataFrame(q_coords.numpy(), 
                       columns=[f'q_{asset}' for asset in price_df.columns])
    q_df.to_csv('processed_data/q_coordinates.csv', index=False)
    
    # Save velocities as CSV  
    q_vel_df = pd.DataFrame(q_velocities.numpy(),
                           columns=[f'dq_{asset}' for asset in price_df.columns])
    q_vel_df.to_csv('processed_data/q_velocities.csv', index=False)
    
    # Save reference prices
    ref_df = pd.DataFrame([reference_prices.numpy()], 
                         columns=[f'ref_{asset}' for asset in price_df.columns])
    ref_df.to_csv('processed_data/reference_prices.csv', index=False)
    
    print("    Saved CSV files for Colab:")
    print("      - q_coordinates.csv")
    print("      - q_velocities.csv") 
    print("      - reference_prices.csv")
    
    return processed_data

def main():
    """Main preprocessing pipeline"""
    print("🚀 STEP 1: Data Preprocessing for Geometric Neural Networks")
    print("=" * 60)
    
    # Load raw data
    all_data = load_crypto_data()
    
    if len(all_data) < 6:
        print(f"\n Warning: Only {len(all_data)}/6 assets loaded!")
        print("   Check that all CSV files exist in crypto_data/")
    
    # Synchronize timestamps
    price_df = synchronize_data(all_data)
    
    # Convert to geometric coordinates
    q_coords, q_velocities, reference_prices = convert_to_geometric_coordinates(price_df)
    
    # Save processed data
    processed_data = save_processed_data(q_coords, q_velocities, reference_prices, price_df)
    
    print(f"\n🎯 PREPROCESSING COMPLETE!")
    print(f"   📊 Data shape: {processed_data['n_timesteps']} timesteps × {processed_data['n_assets']} assets")
    print(f"   🌐 Manifold dimension: {processed_data['n_assets']}D")
    print(f"   💾 Output files ready for next step")
    
    print(f"\n📝 NEXT STEP: Run 'python step2_physics_setup.py'")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()