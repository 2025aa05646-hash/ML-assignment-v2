import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# Load dataset
df = pd.read_csv("dataset/adult.csv", header=None)

# Assign column names
df.columns = [
'age','workclass','fnlwgt','education','education-num',
'marital-status','occupation','relationship','race',
'sex','capital-gain','capital-loss','hours-per-week',
'native-country','income'
]

# Encode categorical variables
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('income', axis=1)
y = df['income']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models dictionary
models = {

"Logistic Regression": LogisticRegression(),

"Decision Tree": DecisionTreeClassifier(),

"KNN": KNeighborsClassifier(),

"Naive Bayes": GaussianNB(),

"Random Forest": RandomForestClassifier(),

"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')

}

# Train and evaluate
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name, accuracy, auc, precision, recall, f1, mcc
    ])

    # Save model
    joblib.dump(model, f"models/{name}.pkl")

# Display results
results_df = pd.DataFrame(results, columns=[
"Model","Accuracy","AUC","Precision","Recall","F1","MCC"
])

print(results_df)

