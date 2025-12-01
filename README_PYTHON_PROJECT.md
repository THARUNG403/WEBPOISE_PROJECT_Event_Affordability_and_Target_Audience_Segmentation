# Event Affordability & Target Audience Segmentation

A comprehensive Python data analysis project that identifies customer segments, analyzes event affordability, and provides strategic recommendations for event organizers.

## 📋 Project Overview

This project analyzes event participation and demographic/economic data to:
- Identify who can afford premium, mid-range, or budget events
- Determine which audience segments prefer different event types
- Suggest event targeting strategies based on affordability and preferences
- Build predictive models for attendance forecasting

## 🚀 Quick Start

### Option 1: Run in Jupyter Notebook (Recommended)

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

2. **Create a new notebook**:
   ```bash
   jupyter notebook
   ```

3. **Copy the code** from `event_affordability_analysis.py` and paste it into notebook cells, or convert the script:
   ```bash
   # Install conversion tool
   pip install jupytext

   # Convert to notebook
   jupytext --to notebook event_affordability_analysis.py
   ```

4. **Run all cells** and view the interactive visualizations!

### Option 2: Run in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → New Notebook**
3. Copy and paste the code from `event_affordability_analysis.py`
4. Run the cells - all dependencies are pre-installed!

### Option 3: Run as Python Script

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python event_affordability_analysis.py
```

## 📦 Dependencies

All required libraries:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📊 What's Included

### 1. Synthetic Dataset Generation
- 1000 samples with realistic demographic and economic data
- Includes: Age, Gender, Income, Event Type, Ticket Price, Attendance

### 2. Data Exploration & Visualization
- Distribution analysis for all key variables
- Demographic breakdowns
- Income vs. pricing relationships
- Event popularity by age groups

### 3. Feature Engineering
- **Affordability Index**: Income/Ticket_Price ratio
- **Budget Categories**: Low, Medium, High based on income terciles
- Encoded categorical variables for modeling

### 4. Clustering Analysis (K-Means)
- Optimal cluster identification using Elbow Method
- Customer segmentation based on income, pricing, and behavior
- Cluster-specific event preferences

### 5. Predictive Modeling
- **Logistic Regression**: Attendance prediction
- **Decision Tree Classifier**: Attendance prediction with feature importance
- Model evaluation with accuracy, precision, recall metrics
- Confusion matrices for performance visualization

### 6. Strategic Insights
- Budget segment preferences and targeting strategies
- Age-based event recommendations
- Pricing optimization guidelines
- Marketing channel recommendations

## 📈 Output Files

The script generates:

### Visualizations (PNG files):
1. `01_data_distributions.png` - Income, age, price, event type distributions
2. `02_demographics.png` - Gender and attendance pie charts
3. `03_income_vs_price.png` - Scatter plot colored by attendance
4. `04_event_by_age.png` - Event popularity across age groups
5. `05_elbow_method.png` - Optimal cluster number determination
6. `06_cluster_visualization.png` - Customer segment visualization
7. `07_cluster_event_preferences.png` - Event preferences by cluster
8. `08_confusion_matrices.png` - Model performance comparison
9. `09_feature_importance.png` - Key factors affecting attendance
10. `10_budget_event_preferences.png` - Event preferences by budget

### Data Export:
- `event_affordability_dataset.csv` - Complete dataset with all engineered features

## 🎯 Key Findings

### Customer Segments (4 Clusters)
Each cluster represents a distinct customer segment with unique:
- Income levels
- Price sensitivity
- Event preferences
- Attendance patterns

### Budget Categories
- **Low Budget**: Community-focused, price-sensitive
- **Medium Budget**: Balanced preferences, moderate spending
- **High Budget**: Premium experiences, high willingness to pay

### Predictive Insights
- Affordability Index is the strongest predictor of attendance
- Event type alignment with demographics significantly impacts attendance
- Age influences event preferences more than gender

## 💡 Strategic Recommendations

The analysis provides actionable recommendations for:
1. **Segment-specific targeting** - Which events to promote to which audiences
2. **Pricing strategies** - Optimal price points per segment
3. **Marketing channels** - Where to reach each segment effectively
4. **Event programming** - What types of events to offer
5. **Revenue optimization** - Balancing volume and premium offerings

## 🔧 Customization

You can easily modify the script to:
- **Change sample size**: Modify `n_samples` parameter in `generate_synthetic_dataset()`
- **Adjust income ranges**: Modify `mean` and `sigma` in log-normal distribution
- **Add new event types**: Update `event_types` and `event_price_ranges` dictionaries
- **Try different cluster numbers**: Change `optimal_k` variable
- **Experiment with models**: Add RandomForest, SVM, or other classifiers

Example:
```python
# Generate more samples
df = generate_synthetic_dataset(n_samples=5000, random_state=42)

# Try 5 clusters instead of 4
optimal_k = 5
```

## 📝 Project Structure

```
project/
│
├── event_affordability_analysis.py    # Main analysis script
├── README_PYTHON_PROJECT.md          # This file
│
└── Generated outputs:
    ├── 01_data_distributions.png
    ├── 02_demographics.png
    ├── ... (all visualization files)
    └── event_affordability_dataset.csv
```

## 🎓 Educational Value

This project demonstrates:
- ✅ Data generation and preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature engineering techniques
- ✅ Unsupervised learning (K-Means clustering)
- ✅ Supervised learning (Classification)
- ✅ Model evaluation and comparison
- ✅ Business insight generation
- ✅ Professional visualization practices

## 📫 Usage Tips

### For Jupyter Notebook:
- Split the code into logical cells (marked by comments with `====`)
- Run cells sequentially
- Add markdown cells between sections for documentation
- Experiment with parameters interactively

### For Presentation:
- All visualizations are high-resolution (300 DPI)
- Perfect for reports, presentations, or portfolios
- Clear titles and labels on all charts
- Professional color schemes

### For GitHub:
- The script is self-contained and reproducible
- No external data files required
- All dependencies are standard Python libraries
- Well-documented with comprehensive comments

## 🚀 Next Steps

To extend this project:
1. **Real data integration**: Replace synthetic data with actual event data
2. **Time series analysis**: Add temporal patterns (seasonality, trends)
3. **Geographic segmentation**: Include location-based analysis
4. **A/B testing framework**: Test pricing strategies
5. **Recommendation system**: Suggest events to users
6. **Dashboard creation**: Build interactive Streamlit/Dash app
7. **API integration**: Connect to ticketing platforms

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

Built using industry-standard data science libraries:
- Pandas for data manipulation
- NumPy for numerical computing
- Matplotlib & Seaborn for visualization
- Scikit-learn for machine learning

---

**Ready to analyze event affordability?** Just run the script and explore the insights! 🎉
