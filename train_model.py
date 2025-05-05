import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv('data/CICIDS2017.csv')

# Encode target labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['attack_type'])

# Feature and target separation
X = df.drop(['attack_type', 'label'], axis=1)
y = df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'models/cyber_threat_model.pkl')

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
