# Data Exploration Toolkit

A comprehensive data exploration and analysis toolkit for customer churn datasets.

## 📋 Features

- **Automated Data Profiling**: Generate comprehensive statistics for all columns
- **Missing Value Analysis**: Identify and report missing data patterns
- **Statistical Summary**: Calculate mean, median, standard deviation, and more
- **Correlation Analysis**: Identify relationships between variables
- **Churn Analysis**: Analyze customer churn patterns and factors
- **Visualization Generation**: Create insightful plots and charts
- **Multiple Export Formats**: JSON, Excel, HTML, and text reports

## 🚀 Quick Start

### Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

### Basic Usage

1. **Generate sample data** (if you don't have a dataset):
```bash
python generate_sample_data.py
```

2. **Run data exploration**:
```bash
python explore_data.py --data sample_data.csv
```

3. **Advanced options**:
```bash
# Skip visualizations
python explore_data.py --data your_data.csv --no-viz

# Specify output formats
python explore_data.py --data your_data.csv --formats json excel
```

## 📊 Dataset Schema

The toolkit expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | String | Unique customer identifier |
| `gender` | Categorical | Customer gender |
| `seniorcitizen` | Binary (0/1) | Senior citizen status |
| `partner` | Categorical | Has partner (Yes/No) |
| `dependents` | Categorical | Has dependents (Yes/No) |
| `months_with_provider` | Numeric | Tenure in months |
| `phone_service` | Categorical | Phone service subscription |
| `extra_lines` | Categorical | Additional phone lines |
| `internet_plan` | Categorical | Internet service type |
| `addon_security` | Categorical | Online security addon |
| `addon_backup` | Categorical | Online backup addon |
| `addon_device_protect` | Categorical | Device protection addon |
| `addon_techsupport` | Categorical | Tech support addon |
| `stream_tv` | Categorical | TV streaming service |
| `stream_movies` | Categorical | Movie streaming service |
| `contract_type` | Categorical | Contract duration |
| `paperless_billing` | Categorical | Paperless billing enabled |
| `payment_method` | Categorical | Payment method used |
| `monthly_fee` | Numeric | Monthly charges |
| `lifetime_spend` | Numeric | Total charges |
| `churned` | Binary (0/1) | Customer churned |
| `customer_hash` | String | Hashed customer ID |
| `marketing_opt_in` | Boolean | Marketing opt-in status |

## 📁 Output Files

After running the exploration script, you'll find the following files in the `output/` directory:

### 1. **summary_statistics.json**
Comprehensive JSON file containing:
- Dataset metadata (rows, columns, memory usage)
- Column type analysis
- Missing value statistics
- Numerical column statistics (mean, std, quartiles, outliers)
- Categorical column distributions
- Correlation matrix
- Churn analysis (if applicable)

### 2. **summary_statistics.xlsx**
Excel workbook with multiple sheets:
- **Dataset Info**: Basic dataset information
- **Numerical Stats**: Statistical measures for numeric columns
- **Categorical Stats**: Frequency distributions for categorical columns

### 3. **summary_report.html**
Interactive HTML report featuring:
- Dataset overview
- Missing value summary
- Column type distribution
- Churn analysis metrics
- Formatted tables and metrics

### 4. **summary.txt**
Plain text summary including:
- Dataset dimensions
- Column types breakdown
- Missing value report
- Key statistics
- Churn metrics

### 5. **Visualizations** (PNG files)
- `numerical_distributions.png`: Histograms of numerical columns
- `correlation_heatmap.png`: Correlation matrix visualization
- `churn_analysis.png`: Churn-related visualizations (if churned column exists)

## 📈 Understanding the Output

### Key Metrics Explained

**Numerical Statistics:**
- **Skewness**: Measure of distribution asymmetry (0 = symmetric)
- **Kurtosis**: Measure of distribution tail heaviness
- **Outliers (IQR)**: Points beyond 1.5×IQR from quartiles

**Churn Analysis:**
- **Overall Churn Rate**: Percentage of customers who left
- **Churn by Features**: Breakdown by categorical variables
- **Contract Type Impact**: Churn rates by contract duration

**Correlation Analysis:**
- Values range from -1 to 1
- Strong correlations: |r| > 0.5
- Perfect correlation: |r| = 1

## 🔧 Customization

The `DataExplorer` class can be extended for custom analyses:

```python
from explore_data import DataExplorer

# Initialize with custom output directory
explorer = DataExplorer()
explorer.output_dir = Path("custom_output")

# Load and analyze data
explorer.load_data("your_data.csv")
summary = explorer.generate_summary()

# Access specific statistics
print(summary['numerical_stats']['monthly_fee'])
```

## 📝 Requirements

- Python 3.9+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- tabulate >= 0.9.0
- openpyxl >= 3.1.0
- jinja2 >= 3.1.0

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.
