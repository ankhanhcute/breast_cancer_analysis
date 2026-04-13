import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Library loaded")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)
df = pd.read_csv('breast-cancer.csv')
print(df.shape)
print("First 5 row",df.head())
# Overview of the data
print("Dataset Info:")
print(df.info())

print("\nBasic Statistics:")
df.describe()
df = df.drop(columns=['id']) #just patient id not useful

# Check diagnosis distribution
print("Diagnosis counts:")
print(df['diagnosis'].value_counts())
print("\nPercentage:")
print(df['diagnosis'].value_counts(normalize=True) * 100)

#convert the cancer M/B into numbers: M=1 (Malignant), B=0(Benign)

df['diagnosis_encoded'] = df['diagnosis'].map({'M':1,'B':0})
print(" Diagnosis encoded!")
print(df[['diagnosis', 'diagnosis_encoded']].head())
print(df[['diagnosis', 'diagnosis_encoded']].tail())
print(df['diagnosis_encoded'].value_counts())

#how many malignant and benign

plt.figure(figsize=(8,5))
sns.countplot(x='diagnosis', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Malignant vs Benign Tumors', fontsize=16)
plt.xlabel('Diagnosis (B = Benign, M = Malignant)')
plt.ylabel('Count')
plt.show()

#Do malignant have bigger radius 

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='radius_mean', hue='diagnosis',
             bins=30, palette=['#2ecc71', '#e74c3c'])
plt.title('Tumor Radius Distribution by Diagnosis', fontsize=16)
plt.xlabel('Mean Radius')
plt.show()

#compare the key features between B/M

feature = ['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean']
fig, axes = plt.subplots(2, 2, figsize=(10,8))
axes = axes.flatten()
for i, feature in enumerate(feature):
  sns.boxplot(x='diagnosis', y=feature, data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[i])
  axes[i].set_title(f'{feature} by Diagnosis', fontsize=13)
plt.tight_layout()
plt.show()

# Which features are most correlated?
plt.figure(figsize=(16, 12))
corr = df.drop(columns=['diagnosis', 'diagnosis_encoded']).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()

# H0: There is NO significant difference in mean radius 
#     between Malignant and Benign tumors
# Ha: Malignant tumors have a significantly LARGER mean radius

malignant = df[df['diagnosis'] == 'M']['radius_mean']
benign = df[df['diagnosis'] == 'B']['radius_mean']

print(f"Malignant mean radius: {malignant.mean():.2f}")
print(f"Benign mean radius: {benign.mean():.2f}")

from scipy.stats import shapiro

stat_m, p_m = shapiro(malignant)
stat_b, p_b = shapiro(benign)

print(f"Malignant normality p-value: {p_m:.4f}")
print(f"Benign normality p-value: {p_b:.4f}")

from scipy.stats import ttest_ind, levene

# Check equal variance first
stat_lev, p_lev = levene(malignant, benign)
print(f"Levene's test p-value: {p_lev:.4f}")

# Run t-test
t_stat, p_value = ttest_ind(malignant, benign, equal_var=False)

print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.10f}")
print(f"\nAlpha = 0.05")

if p_value < 0.05:
    print(" Reject H0 — Significant difference exists!")
else:
    print(" Fail to reject H0 - No significant difference")
    
    from scipy.stats import ttest_ind, levene

# Check equal variance first
stat_lev, p_lev = levene(malignant, benign)
print(f"Levene's test p-value: {p_lev:.4f}")

# Run t-test
t_stat, p_value = ttest_ind(malignant, benign, equal_var=False)

print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.10f}")
print(f"\nAlpha = 0.05")

if p_value < 0.05:
    print(" Reject H0 — Significant difference exists!")
else:
    print(" Fail to reject H0 - No significant difference")
    
#should double check the second normality test, the shapiro test is senstive with larger samples
from scipy.stats import normaltest

stat_m, p_m = normaltest(malignant)
stat_b, p_b = normaltest(benign)

print(f"Malignant normality p-value: {p_m:.4f}")
print(f"Benign normality p-value: {p_b:.4f}")

# Also plot to visually check
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(malignant, bins=20, color='#e74c3c', edgecolor='black')
axes[0].set_title('Malignant Radius Distribution')

axes[1].hist(benign, bins=20, color='#2ecc71', edgecolor='black')
axes[1].set_title('Benign Radius Distribution')

plt.tight_layout()
plt.show()


# Prepare data
X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
y = df['diagnosis_encoded']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Model trained!")
# Make predictions
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Benign', 'Malignant']))