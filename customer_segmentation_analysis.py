import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- PHASE 1: DATA GENERATION ---
def generate_data(n=1000):
    np.random.seed(42)
    data = {
        'CustomerID': range(1, n + 1),
        'Age': np.random.randint(18, 65, n),
        'Tenure_Months': np.random.randint(1, 72, n),
        'Monthly_Usage_Hrs': np.random.uniform(5, 150, n),
        'Monthly_Charge': np.random.uniform(9, 25, n),
        'Churn': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    # Injecting business logic: Low usage highly correlates with churn
    df.loc[df['Monthly_Usage_Hrs'] < 20, 'Churn'] = np.random.choice([0, 1], size=len(df[df['Monthly_Usage_Hrs'] < 20]), p=[0.1, 0.9])
    return df

# --- PHASE 2: CLUSTERING (SEGMENTATION) ---
def segment_customers(df):
    features = ['Tenure_Months', 'Monthly_Usage_Hrs', 'Monthly_Charge']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(scaled_data)
    return df

# --- PHASE 3: CHURN PREDICTION ---
def predict_churn(df):
    X = df[['Age', 'Tenure_Months', 'Monthly_Usage_Hrs', 'Monthly_Charge', 'Segment']]
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Add probabilities back to the main dataframe
    df['Churn_Probability'] = model.predict_proba(X)[:, 1]
    return df, model

# --- PHASE 4: BUSINESS INSIGHTS ---
def get_business_actions(df):
    # 1. Retention Offers: High Churn Risk + High Spenders
    retention = df[(df['Churn_Probability'] > 0.7) & (df['Monthly_Charge'] > df['Monthly_Charge'].median())]
    
    # 2. Early Access: Low Churn Risk + Power Users (Top 20% usage)
    early_access = df[(df['Churn_Probability'] < 0.2) & (df['Monthly_Usage_Hrs'] > df['Monthly_Usage_Hrs'].quantile(0.8))]
    
    # 3. Lost Causes: High Churn Risk + No Engagement
    lost_causes = df[(df['Churn_Probability'] > 0.9) & (df['Tenure_Months'] < 6)]
    
    return retention, early_access, lost_causes

# --- EXECUTION ---
if __name__ == "__main__":
    df = generate_data()
    df = segment_customers(df)
    df, churn_model = predict_churn(df)
    
    retention, early_access, lost_causes = get_business_actions(df)
    
    print("--- EXECUTIVE SUMMARY ---")
    print(f"Total Customers Analyzed: {len(df)}")
    print(f"Action 1: Send Retention Offers to {len(retention)} high-value/high-risk users.")
    print(f"Action 2: Invite {len(early_access)} power users to the 'Beta' program.")
    print(f"Action 3: Do not allocate marketing budget to {len(lost_causes)} lost causes.")
    
    # Final Visual: Churn Risk vs Value
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Monthly_Usage_Hrs', y='Churn_Probability', hue='Segment', palette='viridis')
    plt.axhline(0.7, color='red', linestyle='--', label='High Risk Threshold')
    plt.title('Customer Churn Risk vs. Engagement')
    plt.legend()
    plt.show()
