# K-Fold Cross-Validation Analysis for Loan Approval Prediction Challenge

This challenge consists of an automatic learning competition. The task is to apply the techniques learned in class to a competition to obtain the best model. To this end, a dataset of SME enterprises that have applied for a loan is provided, and their mission will be to build a classifier that determines whether the loan should be granted or denied.

This script performs a comprehensive k-fold cross-validation analysis on a binary classification dataset, implementing multiple machine learning classifiers with advanced preprocessing, class balancing, and evaluation metrics.

## Features

### Data Processing
- **Data Sampling**: Uses 20% of the full dataset for development/testing
- **Missing Value Handling**: SimpleImputer with mean strategy
- **Feature Encoding**: Automatic one-hot encoding for categorical variables
- **Standardization**: StandardScaler applied before PCA
- **Dimensionality Reduction**: PCA with 100 components
- **Class Imbalance**: SMOTE resampling within cross-validation pipeline

### Model Pipeline
Each classifier uses an imbalanced-learn Pipeline with:
1. Missing value imputation
2. Feature scaling
3. PCA transformation
4. SMOTE resampling
5. Model fitting

### Classifiers
- K-Nearest Neighbors (KNN)
- Logistic Regression (balanced weights)
- Random Forest (balanced weights)
- Multi-Layer Perceptron (MLP)

### Evaluation Metrics
- F1-Score (mean and std)
- ROC AUC Score (mean and std)
- Precision
- Recall
- Confusion Matrices (averaged across folds)
- Training vs Testing F1 Score comparison

### Visualizations
- Confusion matrix plots for each classifier
- Learning curves showing training vs cross-validation scores
- Progress tracking with timestamps

## Dependencies

```python
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0 
matplotlib-inline>=0.1.3
```

## Usage

1. Ensure the dataset is in the `/data/train.csv` directory
2. Run the script:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python kfold.py
   ```

## Output

The script generates:
- A timestamped log file (e.g., 'kfold_results_20240220_143022.log') containing:
  - All console outputs
  - Detailed progress logs with timing information
  - Data processing steps
  - Model evaluation results
  - Final performance metrics
- Learning curve plots for each classifier
- Confusion matrix visualizations
- Comprehensive performance metrics including:
  - Model-wise F1 scores across folds
  - ROC AUC scores
  - Precision and Recall metrics
  - Training vs Testing performance comparison


## Implementation Details

### Cross-Validation
- Uses StratifiedKFold with 5 splits
- Maintains class distribution across folds
- Implements proper data leakage prevention

### Pipeline Structure
```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('sampling', SMOTE()),
    ('clf', Classifier())
])
```

### Performance Tracking
- Timestamps for each major operation
- Elapsed time tracking
- Progress indicators for fold processing

## Best Practices Implemented

1. **Data Leakage Prevention**:
   - Preprocessing steps within cross-validation
   - SMOTE applied per fold
   - Proper train/test separation

2. **Robust Evaluation**:
   - Multiple performance metrics
   - Cross-validated scores
   - Confusion matrix analysis
   - Learning curve visualization

3. **Class Imbalance Handling**:
   - SMOTE resampling
   - Balanced class weights
   - Stratified fold splitting

4. **Memory Efficiency**:
   - Dimensionality reduction
   - Proper memory cleanup
   - Efficient data processing

## Notes

- The script is configured to use 20% of the data for faster testing
- All preprocessing steps are included in the pipeline to prevent data leakage
- Learning curves help identify overfitting/underfitting
- Confusion matrices are averaged across all folds for stability

## Results

The script outputs:
1. Best performing model based on F1-score
2. Best performing model based on ROC AUC
3. Detailed metrics for each classifier
4. Visual representations of model performance 