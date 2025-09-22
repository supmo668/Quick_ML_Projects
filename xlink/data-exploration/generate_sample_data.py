#!/usr/bin/env python3
"""
Generate sample customer churn dataset for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

def generate_sample_data(n_rows=1000, output_file='sample_data.csv'):
    """Generate sample customer churn dataset"""
    
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'account_id': [f'ACC{str(i).zfill(6)}' for i in range(1, n_rows + 1)],
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'seniorcitizen': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], n_rows),
        'dependents': np.random.choice(['Yes', 'No'], n_rows, p=[0.3, 0.7]),
        'months_with_provider': np.random.randint(1, 73, n_rows),
        'phone_service': np.random.choice(['Yes', 'No'], n_rows, p=[0.9, 0.1]),
        'extra_lines': np.random.choice(['No', 'One', 'Multiple'], n_rows, p=[0.5, 0.3, 0.2]),
        'internet_plan': np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows, p=[0.35, 0.45, 0.2]),
        'addon_security': np.random.choice(['Yes', 'No', 'No internet'], n_rows, p=[0.3, 0.5, 0.2]),
        'addon_backup': np.random.choice(['Yes', 'No', 'No internet'], n_rows, p=[0.35, 0.45, 0.2]),
        'addon_device_protect': np.random.choice(['Yes', 'No', 'No internet'], n_rows, p=[0.4, 0.4, 0.2]),
        'addon_techsupport': np.random.choice(['Yes', 'No', 'No internet'], n_rows, p=[0.3, 0.5, 0.2]),
        'stream_tv': np.random.choice(['Yes', 'No', 'No internet'], n_rows, p=[0.45, 0.35, 0.2]),
        'stream_movies': np.random.choice(['Yes', 'No', 'No internet'], n_rows, p=[0.45, 0.35, 0.2]),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows, p=[0.55, 0.25, 0.2]),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_rows, p=[0.6, 0.4]),
        'payment_method': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
            n_rows, p=[0.35, 0.2, 0.25, 0.2]
        ),
        'monthly_fee': np.round(np.random.uniform(18.25, 118.75, n_rows), 2),
        'lifetime_spend': np.round(np.random.uniform(18.25, 8684.80, n_rows), 2),
        'churned': np.random.choice([0, 1], n_rows, p=[0.73, 0.27]),
        'customer_hash': [hashlib.md5(f'customer_{i}'.encode()).hexdigest()[:16] for i in range(n_rows)],
        'marketing_opt_in': np.random.choice([True, False], n_rows, p=[0.4, 0.6])
    }
    
    df = pd.DataFrame(data)
    
    # Add some logical consistency
    # No internet users shouldn't have internet services
    no_internet_mask = df['internet_plan'] == 'No'
    for col in ['addon_security', 'addon_backup', 'addon_device_protect', 
                'addon_techsupport', 'stream_tv', 'stream_movies']:
        df.loc[no_internet_mask, col] = 'No internet'
    
    # Lifetime spend should be correlated with months_with_provider and monthly_fee
    df['lifetime_spend'] = df['monthly_fee'] * df['months_with_provider'] * np.random.uniform(0.9, 1.1, n_rows)
    df['lifetime_spend'] = df['lifetime_spend'].round(2)
    
    # Higher churn for month-to-month contracts
    month_to_month_mask = df['contract_type'] == 'Month-to-month'
    df.loc[month_to_month_mask, 'churned'] = np.random.choice([0, 1], month_to_month_mask.sum(), p=[0.55, 0.45])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Sample data generated: {output_file}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Churn rate: {df['churned'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    generate_sample_data()
