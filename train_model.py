import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the lung cancer dataset"""
    print("🔍 Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv("survey-lung-cancer.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
    
    # Map GENDER to numeric (M=1, F=0)
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    
    # Define yes/no columns to convert
    yes_no_cols = [
        'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
        'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
    
    # Convert YES/NO to 1/0
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({'YES': 1, 'NO': 0, '1': 1, '0': 0})
    
    # Define feature columns
    feature_cols = ['GENDER', 'AGE'] + yes_no_cols
    target_col = 'LUNG_CANCER'
    
    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[target_col].astype(str).str.upper().map({'YES': 1, 'NO': 0, '1': 1, '0': 0})
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols

def train_models(X, y, feature_cols):
    """Train multiple models and select the best one"""
    print("\n🧠 Starting model training...")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    # Balance classes using SMOTE
    print("⚖️ Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_imputed, y)
    print(f"After SMOTE - Positive: {sum(y_res)}, Negative: {len(y_res) - sum(y_res)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {}
    
    # 1. XGBoost Model
    print("\n🚀 Training XGBoost...")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    xgb_grid = GridSearchCV(
        xgb_model, xgb_param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    
    best_xgb = xgb_grid.best_estimator_
    xgb_score = accuracy_score(y_test, best_xgb.predict(X_test))
    models['XGBoost'] = {'model': best_xgb, 'score': xgb_score}
    print(f"XGBoost Test Accuracy: {xgb_score:.3f}")
    
    # 2. Random Forest Model
    print("\n🌲 Training Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(
        rf_model, rf_param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    rf_score = accuracy_score(y_test, best_rf.predict(X_test))
    models['RandomForest'] = {'model': best_rf, 'score': rf_score}
    print(f"Random Forest Test Accuracy: {rf_score:.3f}")
    
    # 3. Gradient Boosting Model
    print("\n📈 Training Gradient Boosting...")
    gb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(
        gb_model, gb_param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=0
    )
    gb_grid.fit(X_train, y_train)
    
    best_gb = gb_grid.best_estimator_
    gb_score = accuracy_score(y_test, best_gb.predict(X_test))
    models['GradientBoosting'] = {'model': best_gb, 'score': gb_score}
    print(f"Gradient Boosting Test Accuracy: {gb_score:.3f}")
    
    # Select best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['score'])
    best_model = models[best_model_name]['model']
    best_score = models[best_model_name]['score']
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_score:.3f})")
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    print(f"\n📊 Final Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model, imputer, feature_cols

def save_model_components(model, imputer, feature_cols):
    """Save all model components"""
    print("\n💾 Saving model components...")
    
    # Save main model
    joblib.dump(model, 'lung_cancer_model.pkl')
    print("✅ Saved: lung_cancer_model.pkl")
    
    # Save imputer
    joblib.dump(imputer, 'imputer.pkl')
    print("✅ Saved: imputer.pkl")
    
    # Save feature columns
    joblib.dump(feature_cols, 'feature_cols.pkl')
    print("✅ Saved: feature_cols.pkl")
    
    print("\n🎉 All components saved successfully!")

def main():
    """Main training pipeline"""
    print("🚀 Starting Lung Cancer Prediction Model Training\n")
    
    try:
        # Load and preprocess data
        X, y, feature_cols = load_and_preprocess_data()
        
        # Train models
        best_model, imputer, feature_cols = train_models(X, y, feature_cols)
        
        # Save components
        save_model_components(best_model, imputer, feature_cols)
        
        print("\n✨ Training completed successfully!")
        print("You can now run the Flask app with: python app.py")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()