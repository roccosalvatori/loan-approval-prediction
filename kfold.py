import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, make_scorer, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys
import logging

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Set up logging
log_filename = f'kfold_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_time(message, start_time):
    current_time = time.time()
    elapsed = current_time - start_time
    log_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message} (Time elapsed: {elapsed:.2f}s)"
    logging.info(log_message)

# Start timing
start_time = time.time()

logging.info("Loading dataset...")
# Load dataset
df = pd.read_csv("./data/train.csv")
log_time("Dataset loaded", start_time)

# Take X% of the data for testing
df = df.sample(frac=0.1, random_state=42)
logging.info(f"\nUsing {len(df)} samples for testing")

# Separate features and target
X = df.drop(columns=['id', 'Accept'])
y = df['Accept']

# Print initial class distribution
logging.info("\nInitial class distribution:")
logging.info(y.value_counts(normalize=True))
log_time("Class distribution analyzed", start_time)

logging.info("\nPreprocessing data...")
X = pd.get_dummies(X)
logging.info(f"Feature shape after encoding: {X.shape}")
log_time("Dummy encoding completed", start_time)

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
log_time("Missing values imputed", start_time)

# Apply PCA before SMOTE
logging.info("Applying PCA to reduce dimensionality...")
logging.info(f"Initial feature shape: {X.shape}")

# First standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(random_state=42)
X_pca_initial = pca.fit_transform(X_scaled)

# Plot cumulative explained variance
logging.info("Creating PCA variance plot...")
plt.figure(figsize=(12, 6))

# Calculate explained variance ratios
explained_variance_ratio = pca.explained_variance_ratio_
cumsum = np.cumsum(explained_variance_ratio)

# Print first few components' explained variance
logging.info("\nVariance explained by first 10 components:")
for i, var in enumerate(explained_variance_ratio[:10], 1):
    logging.info(f"Component {i}: {var:.4f} ({cumsum[i-1]:.4f} cumulative)")

# Create the main plot
plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=6, label='Cumulative')
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'ro-', linewidth=2, markersize=6, label='Individual')

# Add 95% variance line
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
plt.axvline(x=np.argmax(cumsum >= 0.95) + 1, color='g', linestyle='--')

# Customize the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('PCA Explained Variance Analysis', fontsize=14, pad=20)
plt.legend(fontsize=10)

plt.ylim(0, 1.1)

plt.annotate(f'95% variance at {np.argmax(cumsum >= 0.95) + 1} components',
            xy=(np.argmax(cumsum >= 0.95) + 1, 0.95),
            xytext=(np.argmax(cumsum >= 0.95) + 5, 0.95),
            arrowprops=dict(facecolor='black', shrink=0.05))

# Save the plot 
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.close()
log_time("PCA variance plot created", start_time)

logging.info(f"\nNumber of components needed for 95% variance: {np.argmax(cumsum >= 0.95) + 1}")
logging.info(f"Total variance explained: {cumsum[-1]:.4f}")
logging.info(f"Variance explained by first component: {explained_variance_ratio[0]:.4f}")
logging.info(f"Variance explained by first 5 components: {cumsum[4]:.4f}")
log_time("PCA components analyzed", start_time)

# Apply SMOTE after PCA
logging.info("Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca_initial, y)
log_time("SMOTE resampling completed", start_time)

# Print resampled class distribution
logging.info("\nResampled class distribution:")
logging.info(pd.Series(y_resampled).value_counts(normalize=True))
log_time("Resampled distribution analyzed", start_time)

# Define scoring metrics
scoring = {
    'f1': make_scorer(f1_score),
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score)
}

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Classifiers with hyperparameters
models = {
    "KNN": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    
    "LogReg": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', LogisticRegression(class_weight="balanced", max_iter=1000))
    ]),
    
    "SVC_rbf": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', SVC(kernel='rbf', class_weight="balanced", probability=True))
    ]),
    
    "SVC_poly": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', SVC(kernel='poly', class_weight="balanced", probability=True))
    ]),
    
    "SVC_sigmoid": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', SVC(kernel='sigmoid', class_weight="balanced", probability=True))
    ]),
    
    "LinearSVC": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', LinearSVC(class_weight="balanced", max_iter=10000))
    ]),
    
    "GaussianNB": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', GaussianNB())
    ]),
    
    "DecisionTree": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', DecisionTreeClassifier(class_weight="balanced"))
    ]),
    
    "RandomForest": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', RandomForestClassifier(class_weight="balanced", n_estimators=100))
    ]),
    
    "MLP": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=42)),
        ('pca', PCA(n_components=57)),
        ('clf', MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50)))
    ])
}

# Evaluate all models
results = {}
model_start_time = time.time()

for name, pipeline in models.items():
    print(f"\nEvaluating {name}...")
    model_iter_start = time.time()
    
    # Get cross-validation scores first
    print(f"  Starting cross-validation for {name}...")
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    log_time(f"  Cross-validation completed for {name}", model_iter_start)
    
    # Calculate confusion matrix for each fold
    conf_matrices = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_start = time.time()
        print(f"  Processing fold {fold_idx + 1}/5...")
        
        # Get fold data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Fit pipeline on training data
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Predict on original (non-SMOTE) test data
        y_pred_fold = pipeline.predict(X_test_fold)
        
        # Calculate confusion matrix on original distribution
        conf_matrices.append(confusion_matrix(y_test_fold, y_pred_fold, labels=[0, 1]))
        log_time(f"  Fold {fold_idx + 1} completed", fold_start)
    
    # Average confusion matrix
    avg_conf_matrix = np.mean(conf_matrices, axis=0)
    
    # Store all results
    results[name] = {
        "F1 scores": scores['test_f1'],
        "F1 mean": np.mean(scores['test_f1']),
        "F1 std": np.std(scores['test_f1']),
        "ROC AUC": scores['test_roc_auc'],
        "ROC AUC mean": np.mean(scores['test_roc_auc']),
        "ROC AUC std": np.std(scores['test_roc_auc']),
        "Precision mean": np.mean(scores['test_precision']),
        "Recall mean": np.mean(scores['test_recall']),
        "Train F1 mean": np.mean(scores['train_f1']),
        "Test F1 mean": np.mean(scores['test_f1']),
        "Confusion Matrix": avg_conf_matrix
    }
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Average Confusion Matrix - {name}')
    plt.colorbar()
    
    # Add labels
    classes = ['Rejected (0)', 'Accepted (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = avg_conf_matrix.max() / 2
    for i in range(avg_conf_matrix.shape[0]):
        for j in range(avg_conf_matrix.shape[1]):
            plt.text(j, i, f'{avg_conf_matrix[i, j]:.0f}',
                    horizontalalignment="center",
                    color="white" if avg_conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()
    
    log_time(f"Model {name} evaluation completed", model_iter_start)

# Show detailed results
logging.info("\nDetailed Results:")
for model_name, score_dict in results.items():
    logging.info(f"\nModel: {model_name}")
    logging.info(f"F1 Score: {score_dict['F1 mean']:.4f} (±{score_dict['F1 std']:.4f})")
    logging.info(f"ROC AUC: {score_dict['ROC AUC mean']:.4f} (±{score_dict['ROC AUC std']:.4f})")
    logging.info(f"Precision: {score_dict['Precision mean']:.4f}")
    logging.info(f"Recall: {score_dict['Recall mean']:.4f}")
    logging.info(f"Train/Test F1 difference: {score_dict['Train F1 mean'] - score_dict['Test F1 mean']:.4f}")

# Find best models
best_f1 = max(results.items(), key=lambda x: x[1]['F1 mean'])
best_auc = max(results.items(), key=lambda x: x[1]['ROC AUC mean'])

logging.info("\nBest Models:")
logging.info(f"Best F1-Score: {best_f1[0]} (Mean: {best_f1[1]['F1 mean']:.4f})")
logging.info(f"Best ROC-AUC: {best_auc[0]} (Mean: {best_auc[1]['ROC AUC mean']:.4f})")

log_time("Script completed", start_time)
