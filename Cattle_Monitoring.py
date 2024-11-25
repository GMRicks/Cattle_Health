#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


np.random.seed(42)  # For reproducibility
num_samples = 1000


# In[43]:


temperature = np.random.uniform(100.5, 104.5, num_samples)
pulse = np.random.uniform(40, 90, num_samples)


# In[44]:


temperature_celsius = (temperature - 32) * 5 / 9


# In[45]:


health_status = ['Fit' if (38.5 <= t <= 39.5) and (48 <= p <= 84) else 'Not Fit' 
                 for t, p in zip(temperature_celsius, pulse)]


# In[46]:


data = pd.DataFrame({
    'TemperatureF': temperature,
    'TemperatureC': temperature_celsius,
    'Pulse': pulse,
    'HealthStatus': health_status
})
print("First 5 rows of the dataset:")
print(data.head())


# In[47]:


plt.figure(figsize=(14, 7))
sns.scatterplot(data=data, x='TemperatureC', y='Pulse', hue='HealthStatus', alpha=0.8, palette='coolwarm')
plt.title('Temperature (°C) vs Pulse with Health Status', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Pulse (bpm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Health Status')
plt.show()


# In[48]:


plt.figure(figsize=(16, 8))


# In[49]:


sns.histplot(data=data, x='TemperatureC', hue='HealthStatus', kde=True, palette='coolwarm', alpha=0.7)
plt.title('Temperature Distribution (°C)', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)


# In[50]:


sns.histplot(data=data, x='Pulse', hue='HealthStatus', kde=True, palette='coolwarm', alpha=0.7)
plt.title('Pulse Distribution (bpm)', fontsize=14)
plt.xlabel('Pulse (bpm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show()


# In[51]:


sns.pairplot(data, hue='HealthStatus', vars=['TemperatureC', 'Pulse'], palette='coolwarm', diag_kind='kde')
plt.suptitle('Pairplot Analysis', y=1.02, fontsize=16)
plt.show()


# In[52]:


correlation_matrix = data[['TemperatureC', 'Pulse']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np


# In[54]:


label_encoder = LabelEncoder()
data['HealthStatusEncoded'] = label_encoder.fit_transform(data['HealthStatus'])
X = data[['TemperatureC', 'Pulse']]
y = data['HealthStatusEncoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)


# In[55]:


y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# In[56]:


conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='coolwarm')
plt.title("Confusion Matrix")
plt.show()


# In[57]:


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# In[58]:


plt.figure(figsize=(10, 6))

x_min, x_max = X_train['TemperatureC'].min() - 1, X_train['TemperatureC'].max() + 1
y_min, y_max = X_train['Pulse'].min() - 5, X_train['Pulse'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.5))


# In[59]:


Z = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


# In[60]:


plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm')

# Plot the actual test data points on top of the decision boundary
sns.scatterplot(x=X_test['TemperatureC'], y=X_test['Pulse'], hue=y_test, palette='coolwarm', alpha=0.8)

plt.title('Decision Boundary with Test Data', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Pulse Rate (bpm)', fontsize=12)
plt.show()


# In[61]:


from sklearn.ensemble import IsolationForest
import matplotlib.animation as animation


# In[62]:


isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
data_for_if = data[['TemperatureC', 'Pulse']]
isolation_forest.fit(data_for_if)


# In[64]:


data['Anomaly'] = isolation_forest.predict(data_for_if)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['TemperatureC'],
    y=data['Pulse'],
    hue=data['Anomaly'],
    palette={1: 'blue', -1: 'red'},
    alpha=0.8,
    legend='full'
)
plt.title("Anomaly Detection (Temperature vs Pulse)", fontsize=16)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Pulse (bpm)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(title="Status", labels=["Normal", "Anomalous"])
plt.show()


# In[ ]:




