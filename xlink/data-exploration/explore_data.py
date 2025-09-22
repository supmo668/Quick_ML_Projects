#!/usr/bin/env python3
"""
Data Exploration Script for Customer Churn Dataset
Generates comprehensive summary statistics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class DataExplorer:
    def __init__(self, data_path=None):
        """Initialize the DataExplorer with optional data path"""
        self.data = None
        self.summary_stats = {}
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def generate_summary(self):
        """Generate comprehensive summary statistics"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": self._get_dataset_info(),
            "column_types": self._analyze_column_types(),
            "missing_values": self._analyze_missing_values(),
            "numerical_stats": self._analyze_numerical_columns(),
            "categorical_stats": self._analyze_categorical_columns(),
            "correlations": self._analyze_correlations(),
            "churn_analysis": self._analyze_churn() if 'churned' in self.data.columns else None
        }
        
        self.summary_stats = summary
        return summary
    
    def _get_dataset_info(self):
        """Get basic dataset information"""
        return {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "columns": list(self.data.columns)
        }
    
    def _analyze_column_types(self):
        """Analyze and categorize column types"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Check for binary columns
        binary_cols = []
        for col in self.data.columns:
            unique_vals = self.data[col].nunique()
            if unique_vals == 2:
                binary_cols.append(col)
        
        return {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "binary": binary_cols,
            "dtype_distribution": {str(k): int(v) for k, v in self.data.dtypes.value_counts().to_dict().items()}
        }
    
    def _analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        return {
            "total_missing": int(missing.sum()),
            "columns_with_missing": missing[missing > 0].to_dict(),
            "missing_percentage": missing_pct[missing_pct > 0].to_dict()
        }
    
    def _analyze_numerical_columns(self):
        """Analyze numerical columns"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numerical_cols:
            col_data = self.data[col].dropna()
            stats[col] = {
                "count": int(col_data.count()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "25%": float(col_data.quantile(0.25)),
                "50%": float(col_data.quantile(0.50)),
                "75%": float(col_data.quantile(0.75)),
                "max": float(col_data.max()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "outliers_iqr": self._detect_outliers_iqr(col_data)
            }
        
        return stats
    
    def _analyze_categorical_columns(self):
        """Analyze categorical columns"""
        categorical_cols = self.data.select_dtypes(include=['object', 'bool']).columns
        stats = {}
        
        for col in categorical_cols:
            value_counts = self.data[col].value_counts()
            stats[col] = {
                "unique_values": int(self.data[col].nunique()),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "frequency_most_frequent": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "distribution": {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()}
            }
        
        return stats
    
    def _analyze_correlations(self):
        """Analyze correlations between numerical columns"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = self.data[numerical_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_value)
                        })
            
            return {
                "correlation_matrix": {str(col): {str(row): float(val) for row, val in corr_matrix[col].to_dict().items()} for col in corr_matrix.columns},
                "strong_correlations": strong_correlations
            }
        return None
    
    def _analyze_churn(self):
        """Analyze churn patterns if churned column exists"""
        if 'churned' not in self.data.columns:
            return None
        
        churn_rate = self.data['churned'].mean()
        
        # Analyze churn by different features
        churn_by_features = {}
        
        for col in self.data.columns:
            if col != 'churned' and col not in ['account_id', 'customer_hash']:
                try:
                    if self.data[col].dtype in ['object', 'bool'] or self.data[col].nunique() < 10:
                        churn_by_feature = self.data.groupby(col)['churned'].agg(['mean', 'count'])
                        churn_by_features[col] = {
                            'mean': {str(k): float(v) for k, v in churn_by_feature['mean'].to_dict().items()},
                            'count': {str(k): int(v) for k, v in churn_by_feature['count'].to_dict().items()}
                        }
                except:
                    pass
        
        return {
            "overall_churn_rate": float(churn_rate),
            "churned_count": int(self.data['churned'].sum()),
            "retained_count": int(len(self.data) - self.data['churned'].sum()),
            "churn_by_features": churn_by_features
        }
    
    def _detect_outliers_iqr(self, data):
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(data)) * 100
        }
    
    def create_visualizations(self):
        """Create and save visualization plots"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Distribution plots for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            fig, axes = plt.subplots(
                nrows=(len(numerical_cols) + 3) // 4, 
                ncols=min(4, len(numerical_cols)),
                figsize=(16, 4 * ((len(numerical_cols) + 3) // 4))
            )
            axes = axes.flatten() if len(numerical_cols) > 1 else [axes]
            
            for idx, col in enumerate(numerical_cols[:len(axes)]):
                self.data[col].hist(ax=axes[idx], bins=30, edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'numerical_distributions.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.data[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # 3. Churn analysis visualization
        if 'churned' in self.data.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Churn distribution
            self.data['churned'].value_counts().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Churn Distribution')
            axes[0, 0].set_xlabel('Churned')
            axes[0, 0].set_ylabel('Count')
            
            # Churn by contract type if exists
            if 'contract_type' in self.data.columns:
                churn_by_contract = self.data.groupby('contract_type')['churned'].mean()
                churn_by_contract.plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Churn Rate by Contract Type')
                axes[0, 1].set_xlabel('Contract Type')
                axes[0, 1].set_ylabel('Churn Rate')
            
            # Monthly fee distribution by churn status
            if 'monthly_fee' in self.data.columns:
                self.data.boxplot(column='monthly_fee', by='churned', ax=axes[1, 0])
                axes[1, 0].set_title('Monthly Fee by Churn Status')
                axes[1, 0].set_xlabel('Churned')
                axes[1, 0].set_ylabel('Monthly Fee')
            
            # Tenure distribution by churn status
            if 'months_with_provider' in self.data.columns:
                self.data.boxplot(column='months_with_provider', by='churned', ax=axes[1, 1])
                axes[1, 1].set_title('Tenure by Churn Status')
                axes[1, 1].set_xlabel('Churned')
                axes[1, 1].set_ylabel('Months with Provider')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'churn_analysis.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Visualizations saved to {self.output_dir}")
    
    def export_summary(self, formats=['json', 'csv', 'html']):
        """Export summary statistics in multiple formats"""
        if not self.summary_stats:
            self.generate_summary()
        
        # JSON export
        if 'json' in formats:
            with open(self.output_dir / 'summary_statistics.json', 'w') as f:
                json.dump(self.summary_stats, f, indent=2, default=str)
            print(f"✓ JSON summary exported to {self.output_dir / 'summary_statistics.json'}")
        
        # CSV export
        if 'csv' in formats:
            self._export_csv_summary()
            print(f"✓ CSV summary exported to {self.output_dir / 'summary_statistics.csv'}")
        
        # Excel export (keeping for backward compatibility)
        if 'excel' in formats:
            with pd.ExcelWriter(self.output_dir / 'summary_statistics.xlsx') as writer:
                # Dataset info
                pd.DataFrame([self.summary_stats['dataset_info']]).to_excel(
                    writer, sheet_name='Dataset Info', index=False
                )
                
                # Numerical statistics
                if self.summary_stats['numerical_stats']:
                    pd.DataFrame(self.summary_stats['numerical_stats']).T.to_excel(
                        writer, sheet_name='Numerical Stats'
                    )
                
                # Categorical statistics
                if self.summary_stats['categorical_stats']:
                    cat_df = pd.DataFrame(self.summary_stats['categorical_stats']).T
                    cat_df.to_excel(writer, sheet_name='Categorical Stats')
            
            print(f"✓ Excel summary exported to {self.output_dir / 'summary_statistics.xlsx'}")
        
        # HTML export
        if 'html' in formats:
            self._export_html_report()
            print(f"✓ HTML report exported to {self.output_dir / 'summary_report.html'}")
        
        # Text summary
        self._export_text_summary()
        print(f"✓ Text summary exported to {self.output_dir / 'summary.txt'}")
    
    def _export_csv_summary(self):
        """Export comprehensive summary statistics to CSV format"""
        summary_rows = []
        
        # Dataset Information
        summary_rows.append(['Section', 'Metric', 'Value', 'Details'])
        summary_rows.append(['Dataset Info', 'Total Rows', self.summary_stats['dataset_info']['total_rows'], ''])
        summary_rows.append(['Dataset Info', 'Total Columns', self.summary_stats['dataset_info']['total_columns'], ''])
        summary_rows.append(['Dataset Info', 'Memory Usage (MB)', f"{self.summary_stats['dataset_info']['memory_usage_mb']:.2f}", ''])
        summary_rows.append(['', '', '', ''])  # Empty row for separation
        
        # Column Types
        summary_rows.append(['Column Types', 'Numerical Columns', len(self.summary_stats['column_types']['numerical']), str(self.summary_stats['column_types']['numerical'])])
        summary_rows.append(['Column Types', 'Categorical Columns', len(self.summary_stats['column_types']['categorical']), str(self.summary_stats['column_types']['categorical'])])
        summary_rows.append(['Column Types', 'Binary Columns', len(self.summary_stats['column_types']['binary']), str(self.summary_stats['column_types']['binary'])])
        summary_rows.append(['', '', '', ''])  # Empty row for separation
        
        # Missing Values
        summary_rows.append(['Missing Values', 'Total Missing', self.summary_stats['missing_values']['total_missing'], ''])
        for col, count in self.summary_stats['missing_values']['columns_with_missing'].items():
            pct = self.summary_stats['missing_values']['missing_percentage'][col]
            summary_rows.append(['Missing Values', col, count, f'{pct:.2f}%'])
        summary_rows.append(['', '', '', ''])  # Empty row for separation
        
        # Numerical Statistics
        if self.summary_stats['numerical_stats']:
            summary_rows.append(['Numerical Stats', 'Column', 'Statistic', 'Value'])
            for col, stats in self.summary_stats['numerical_stats'].items():
                summary_rows.append(['Numerical Stats', col, 'Count', stats['count']])
                summary_rows.append(['Numerical Stats', col, 'Mean', f"{stats['mean']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Std', f"{stats['std']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Min', f"{stats['min']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Q1 (25%)', f"{stats['25%']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Median (50%)', f"{stats['50%']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Q3 (75%)', f"{stats['75%']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Max', f"{stats['max']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Skewness', f"{stats['skewness']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Kurtosis', f"{stats['kurtosis']:.4f}"])
                summary_rows.append(['Numerical Stats', col, 'Outliers (%)', f"{stats['outliers_iqr']['percentage']:.2f}%"])
                summary_rows.append(['', '', '', ''])  # Empty row between columns
        
        # Categorical Statistics
        if self.summary_stats['categorical_stats']:
            summary_rows.append(['Categorical Stats', 'Column', 'Metric', 'Value'])
            for col, stats in self.summary_stats['categorical_stats'].items():
                summary_rows.append(['Categorical Stats', col, 'Unique Values', stats['unique_values']])
                summary_rows.append(['Categorical Stats', col, 'Most Frequent', stats['most_frequent']])
                summary_rows.append(['Categorical Stats', col, 'Frequency of Most Frequent', stats['frequency_most_frequent']])
                # Add top categories
                for category, count in list(stats['distribution'].items())[:5]:  # Top 5 categories
                    summary_rows.append(['Categorical Stats', col, f'Category: {category}', count])
                summary_rows.append(['', '', '', ''])  # Empty row between columns
        
        # Correlations
        if self.summary_stats.get('correlations') and self.summary_stats['correlations']['strong_correlations']:
            summary_rows.append(['Strong Correlations', 'Variable 1', 'Variable 2', 'Correlation'])
            for corr in self.summary_stats['correlations']['strong_correlations']:
                summary_rows.append(['Strong Correlations', corr['var1'], corr['var2'], f"{corr['correlation']:.4f}"])
            summary_rows.append(['', '', '', ''])  # Empty row for separation
        
        # Churn Analysis
        if self.summary_stats.get('churn_analysis'):
            summary_rows.append(['Churn Analysis', 'Overall Churn Rate', f"{self.summary_stats['churn_analysis']['overall_churn_rate']:.4f}", f"{self.summary_stats['churn_analysis']['overall_churn_rate']:.2%}"])
            summary_rows.append(['Churn Analysis', 'Churned Customers', self.summary_stats['churn_analysis']['churned_count'], ''])
            summary_rows.append(['Churn Analysis', 'Retained Customers', self.summary_stats['churn_analysis']['retained_count'], ''])
            summary_rows.append(['', '', '', ''])  # Empty row for separation
            
            # Churn by features (top insights)
            summary_rows.append(['Churn by Feature', 'Feature', 'Category', 'Churn Rate'])
            for feature, data in self.summary_stats['churn_analysis']['churn_by_features'].items():
                if feature in ['contract_type', 'internet_plan', 'payment_method', 'seniorcitizen']:  # Key features
                    for category, rate in data['mean'].items():
                        summary_rows.append(['Churn by Feature', feature, category, f"{rate:.4f} ({rate:.2%})"])
        
        # Save to CSV
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.output_dir / 'summary_statistics.csv', index=False, header=False)
    
    def _export_html_report(self):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Exploration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Data Exploration Report</h1>
            <p>Generated: {self.summary_stats['timestamp']}</p>
            
            <h2>Dataset Overview</h2>
            <div class="metric">
                <p><strong>Total Rows:</strong> {self.summary_stats['dataset_info']['total_rows']}</p>
                <p><strong>Total Columns:</strong> {self.summary_stats['dataset_info']['total_columns']}</p>
                <p><strong>Memory Usage:</strong> {self.summary_stats['dataset_info']['memory_usage_mb']:.2f} MB</p>
            </div>
            
            <h2>Missing Values</h2>
            <p><strong>Total Missing Values:</strong> {self.summary_stats['missing_values']['total_missing']}</p>
            
            <h2>Column Types</h2>
            <p><strong>Numerical Columns:</strong> {len(self.summary_stats['column_types']['numerical'])}</p>
            <p><strong>Categorical Columns:</strong> {len(self.summary_stats['column_types']['categorical'])}</p>
            <p><strong>Binary Columns:</strong> {len(self.summary_stats['column_types']['binary'])}</p>
        """
        
        if self.summary_stats.get('churn_analysis'):
            html_content += f"""
            <h2>Churn Analysis</h2>
            <div class="metric">
                <p><strong>Overall Churn Rate:</strong> {self.summary_stats['churn_analysis']['overall_churn_rate']:.2%}</p>
                <p><strong>Churned Customers:</strong> {self.summary_stats['churn_analysis']['churned_count']}</p>
                <p><strong>Retained Customers:</strong> {self.summary_stats['churn_analysis']['retained_count']}</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(self.output_dir / 'summary_report.html', 'w') as f:
            f.write(html_content)
    
    def _export_text_summary(self):
        """Export text summary"""
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA EXPLORATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {self.summary_stats['timestamp']}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            for key, value in self.summary_stats['dataset_info'].items():
                if key != 'columns':
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("COLUMN TYPES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Numerical: {len(self.summary_stats['column_types']['numerical'])} columns\n")
            f.write(f"Categorical: {len(self.summary_stats['column_types']['categorical'])} columns\n")
            f.write(f"Binary: {len(self.summary_stats['column_types']['binary'])} columns\n\n")
            
            f.write("MISSING VALUES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total missing values: {self.summary_stats['missing_values']['total_missing']}\n")
            if self.summary_stats['missing_values']['columns_with_missing']:
                f.write("Columns with missing values:\n")
                for col, count in self.summary_stats['missing_values']['columns_with_missing'].items():
                    f.write(f"  {col}: {count}\n")
            else:
                f.write("No missing values found\n")
            f.write("\n")
            
            if self.summary_stats.get('churn_analysis'):
                f.write("CHURN ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall churn rate: {self.summary_stats['churn_analysis']['overall_churn_rate']:.2%}\n")
                f.write(f"Churned customers: {self.summary_stats['churn_analysis']['churned_count']}\n")
                f.write(f"Retained customers: {self.summary_stats['churn_analysis']['retained_count']}\n")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore and analyze customer data')
    parser.add_argument('--data', type=str, help='Path to the CSV data file', required=True)
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--formats', nargs='+', default=['json', 'csv', 'html'],
                       choices=['json', 'csv', 'excel', 'html'],
                       help='Output formats for summary statistics')
    
    args = parser.parse_args()
    
    # Initialize explorer
    explorer = DataExplorer()
    
    # Load data
    explorer.load_data(args.data)
    
    # Generate summary
    print("\n📊 Generating summary statistics...")
    summary = explorer.generate_summary()
    
    # Create visualizations
    if not args.no_viz:
        print("\n📈 Creating visualizations...")
        explorer.create_visualizations()
    
    # Export summary
    print("\n💾 Exporting summary statistics...")
    explorer.export_summary(formats=args.formats)
    
    print("\n✅ Data exploration complete!")
    print(f"📁 All outputs saved to: {explorer.output_dir.absolute()}")

if __name__ == "__main__":
    main()
