import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def log_time(message, start_time):
    current_time = time.time()
    elapsed = current_time - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message} (Time elapsed: {elapsed:.2f}s)")

# Start timing
start_time = time.time()

print("Loading dataset...")
# Load dataset
df = pd.read_csv("./data/train.csv")
log_time("Dataset loaded", start_time)

# Take 20% of the data for testing
df = df.sample(frac=0.05, random_state=42)
print(f"\nUsing {len(df)} samples for testing")

# Separate features and target
X = df.drop(columns=['id', 'Accept'])
y = df['Accept']

# Print initial class distribution
print("\nInitial class distribution:")
print(y.value_counts(normalize=True))
log_time("Class distribution analyzed", start_time)

print("\nPreprocessing data...")
# Optional: Encode object columns if any
X = pd.get_dummies(X)
log_time("Dummy encoding completed", start_time)

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
log_time("Missing values imputed", start_time)

print("Applying SMOTE for class balancing...")
# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
log_time("SMOTE resampling completed", start_time)

# Print resampled class distribution
print("\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts(normalize=True))
log_time("Resampled distribution analyzed", start_time)

# Apply PCA after SMOTE
print("Applying PCA to reduce dimensionality...")
pca = PCA(random_state=42)
X_pca = pca.fit_transform(X_resampled)
log_time("PCA transformation completed", start_time)

# Plot cumulative explained variance
print("Creating PCA variance plot...")
plt.figure(figsize=(10, 6))
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
plt.grid(True)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.close()
log_time("PCA variance plot created", start_time)

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"\nNumber of components needed for 95% variance: {n_components_95}")
print(f"Total variance explained: {cumsum[-1]:.4f}")
log_time("PCA components analyzed", start_time)

# Now apply PCA with the optimal number of components
pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_resampled)

# Define K-Fold and scoring
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'f1': make_scorer(f1_score),
    'roc_auc': 'roc_auc'
}

# Classifiers with imbalance handling
models = {
    "KNN": KNeighborsClassifier(),
    "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000),
    "SVC_rbf": SVC(kernel='rbf', class_weight="balanced", probability=True),
    "SVC_poly": SVC(kernel='poly', class_weight="balanced", probability=True),
    "SVC_sigmoid": SVC(kernel='sigmoid', class_weight="balanced", probability=True),
    "LinearSVC": LinearSVC(class_weight="balanced", max_iter=10000),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
    "RandomForest": RandomForestClassifier(class_weight="balanced"),
    "MLP": MLPClassifier(max_iter=1000)
}

# Evaluate all models
results = {}
model_start_time = time.time()

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    model_iter_start = time.time()
    
    # Build pipeline
    if name == "GaussianNB":  # GaussianNB doesn't like scaled data
        pipeline = Pipeline([
            ('clf', model)
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
    
    # Get cross-validation scores
    print(f"  Starting cross-validation for {name}...")
    scores = cross_validate(pipeline, X_pca, y_resampled, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    log_time(f"  Cross-validation completed for {name}", model_iter_start)
    
    # Calculate confusion matrix for each fold
    conf_matrices = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_pca, y_resampled)):
        fold_start = time.time()
        print(f"  Processing fold {fold_idx + 1}/5...")
        X_train_fold = X_pca[train_idx]
        y_train_fold = y_resampled[train_idx]
        X_test_fold = X_pca[test_idx]
        y_test_fold = y_resampled[test_idx]
        
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred_fold = pipeline.predict(X_test_fold)
        conf_matrices.append(confusion_matrix(y_test_fold, y_pred_fold))
        log_time(f"  Fold {fold_idx + 1} completed", fold_start)
    
    # Average confusion matrix
    avg_conf_matrix = np.mean(conf_matrices, axis=0)
    
    results[name] = {
        "F1 scores": scores['test_f1'],
        "F1 mean": np.mean(scores['test_f1']),
        "F1 std": np.std(scores['test_f1']),
        "ROC AUC": scores['test_roc_auc'],
        "ROC AUC mean": np.mean(scores['test_roc_auc']),
        "ROC AUC std": np.std(scores['test_roc_auc']),
        "Confusion Matrix": avg_conf_matrix
    }

    # Plot confusion matrix
    print(f"  Creating confusion matrix plot for {name}...")
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Average Confusion Matrix - {name}')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()
    log_time(f"Model {name} evaluation completed", model_iter_start)

# Show results
print("\nDetailed Results:")
for model_name, score_dict in results.items():
    print(f"\nModel: {model_name}")
    print("F1 Scores (folds):", score_dict['F1 scores'])
    print(f"F1 Mean: {round(score_dict['F1 mean'], 4)} (±{round(score_dict['F1 std'], 4)})")
    print("ROC AUC (folds):", score_dict['ROC AUC'])
    print(f"ROC AUC Mean: {round(score_dict['ROC AUC mean'], 4)} (±{round(score_dict['ROC AUC std'], 4)})")

# Find best models based on F1-score and ROC-AUC
best_f1 = max(results.items(), key=lambda x: x[1]['F1 mean'])
best_auc = max(results.items(), key=lambda x: x[1]['ROC AUC mean'])

print("\nBest Models:")
print(f"Best F1-Score: {best_f1[0]} (Mean: {round(best_f1[1]['F1 mean'], 4)})")
print(f"Best ROC-AUC: {best_auc[0]} (Mean: {round(best_auc[1]['ROC AUC mean'], 4)})")

log_time("Script completed", start_time)
