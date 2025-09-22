#!/usr/bin/env python3
"""
Create inference examples from actual customer data
"""

import pandas as pd
import json
from pathlib import Path

def create_inference_examples():
    """Create inference examples from actual customer data"""
    
    # Load the actual customer data
    data_path = Path(__file__).parent.parent / "data" / "customer_churn.csv"
    if not data_path.exists():
        print(f"❌ Customer data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} customer records")
    
    # Select diverse examples
    examples = []
    
    # High-risk: Month-to-month with electronic check
    high_risk = df[
        (df['contract_type'] == 'Month-to-month') & 
        (df['payment_method'] == 'Electronic check') &
        (df['churned'] == 1)
    ].head(1)
    
    if not high_risk.empty:
        customer = high_risk.iloc[0]
        # Remove target variable and non-feature columns
        features = customer.drop(['churned', 'account_id', 'customer_hash']).to_dict()
        
        examples.append({
            "name": "high_risk_customer",
            "description": "Month-to-month customer with electronic check payment (actual churned customer)",
            "features": {k: (int(v) if isinstance(v, (int, bool)) else float(v) if isinstance(v, float) else str(v)) 
                        for k, v in features.items() if pd.notna(v)},
            "expected_churn": True,
            "actual_churned": True
        })
    
    # Low-risk: Long-term contract with automatic payment
    low_risk = df[
        (df['contract_type'].isin(['One year', 'Two year'])) & 
        (df['payment_method'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])) &
        (df['churned'] == 0) &
        (df['months_with_provider'] > 24)
    ].head(1)
    
    if not low_risk.empty:
        customer = low_risk.iloc[0]
        features = customer.drop(['churned', 'account_id', 'customer_hash']).to_dict()
        
        examples.append({
            "name": "low_risk_customer", 
            "description": "Long-term customer with automatic payment (actual retained customer)",
            "features": {k: (int(v) if isinstance(v, (int, bool)) else float(v) if isinstance(v, float) else str(v)) 
                        for k, v in features.items() if pd.notna(v)},
            "expected_churn": False,
            "actual_churned": False
        })
    
    # Medium-risk: One year contract, mixed payment
    medium_risk = df[
        (df['contract_type'] == 'One year') & 
        (df['months_with_provider'].between(12, 36))
    ].head(1)
    
    if not medium_risk.empty:
        customer = medium_risk.iloc[0]
        features = customer.drop(['churned', 'account_id', 'customer_hash']).to_dict()
        
        examples.append({
            "name": "medium_risk_customer",
            "description": "One-year contract customer (actual customer)",
            "features": {k: (int(v) if isinstance(v, (int, bool)) else float(v) if isinstance(v, float) else str(v)) 
                        for k, v in features.items() if pd.notna(v)},
            "expected_churn": None,  # Unknown expectation
            "actual_churned": bool(customer['churned'])
        })
    
    inference_data = {
        "description": "Real customer inference examples from actual data",
        "format": "Features extracted directly from customer_churn.csv",
        "examples": examples
    }
    
    return inference_data

if __name__ == "__main__":
    inference_data = create_inference_examples()
    if inference_data:
        output_file = Path(__file__).parent / "real_inference_examples.json"
        with open(output_file, 'w') as f:
            json.dump(inference_data, f, indent=2)
        print(f"✅ Created {len(inference_data['examples'])} real inference examples")
        print(f"📄 Saved to: {output_file}")
        
        # Print examples for verification
        for example in inference_data['examples']:
            print(f"\n🔍 {example['name']}:")
            print(f"   📝 {example['description']}")
            print(f"   🎯 Actual churned: {example['actual_churned']}")
            print(f"   📊 Features: {len(example['features'])} attributes")
    else:
        print("❌ Failed to create inference examples")