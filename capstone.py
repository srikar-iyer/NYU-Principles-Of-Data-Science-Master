import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

seed = 18752360
dfmain = pd.read_csv("C:/Main/nyu/ds/spotify52kData.csv")
#dfmain[['songNumber']] = dfmain[['songNumber']] + 1
dfmain

# Question 1
dfnum = dfmain[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

fig, axs = plt.subplots(2, 5, figsize=(15, 8))
axs = axs.flatten()

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

normal_params = {
    'duration': {'mean': dfnum['duration'].mean(), 'std': dfnum['duration'].std()},
    'duration': {'mean': dfnum['duration'].mean(), 'std': dfnum['duration'].std()},
    'danceability': {'mean': dfnum['danceability'].mean(), 'std': dfnum['danceability'].std()},
    'energy': {'mean': dfnum['energy'].mean(), 'std': dfnum['energy'].std()},
    'loudness': {'mean': dfnum['loudness'].mean(), 'std': dfnum['loudness'].std()},
    'speechiness': {'mean': dfnum['speechiness'].mean(), 'std': dfnum['speechiness'].std()},
    'acousticness': {'mean': dfnum['acousticness'].mean(), 'std': dfnum['acousticness'].std()},
    'instrumentalness': {'mean': dfnum['instrumentalness'].mean(), 'std': dfnum['instrumentalness'].std()},
    'liveness': {'mean': dfnum['liveness'].mean(), 'std': dfnum['liveness'].std()},
    'valence': {'mean': dfnum['valence'].mean(), 'std': dfnum['valence'].std()},
    'tempo': {'mean': dfnum['tempo'].mean(), 'std': dfnum['tempo'].std()}
}



for i, feature in enumerate(features): 
    # Plot histogram
    plt.hist(dfnum[feature], bins=30, density=True, alpha=0.9, color='blue', label='Sample Data')
    
    # Calculate normal distribution based on mean and standard deviation
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, normal_params[feature]['mean'], normal_params[feature]['std'])
    
    # Plot normal distribution
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    
    plt.xlabel(feature.capitalize())
    plt.ylabel('Density')
    plt.title(f'Histogram with Normal Distribution for {feature.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_pval = kstest(norm.pdf(x, normal_params[feature]['mean'], normal_params[feature]['std']), dfnum[feature],  method='exact')
    print(f"Kolmogorov-Smirnov test for {feature}: KS Statistic = {ks_stat}, p-value = {ks_pval}")

filtered_df = dfmain[dfmain['duration'] / (1000. * 60) < 20.0]


plt.figure(figsize=(12, 10))

plt.scatter(dfmain['popularity'], dfmain['duration'] / (1000. * 60), alpha=0.8)
plt.title('Relationship between Popularity and Song Length')
          
#plt.scatter(filtered_df['popularity'], filtered_df['duration'] / (1000. * 60), alpha=0.8)
#plt.title('Relationship between Popularity and Song Length (Duration < 20 min)')          
          
plt.ylabel('Duration (min)')
plt.xlabel('Popularity')
plt.grid(True)
plt.show()

print(dfmain[['popularity', 'duration']].describe())
#print(filtered_df[['popularity', 'duration']].describe())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# Assuming dfmain is your DataFrame
# Filter the DataFrame to include only rows where duration is under 20 minutes
#filtered_df = dfmain
filtered_df = dfmain[(dfmain['duration'] / (1000. * 60) >= 12.00) & (dfmain['duration'] / (1000. * 60) <= 100000.00)]

# Extracting the features and target variable
X = filtered_df[['popularity']]
y = filtered_df['duration'] / (1000. * 60)  # Converting duration to minutes

#X = dfmain[['popularity']]
#y = dfmain['duration'] / (1000. * 60)  # Converting duration to minutes


# Ordinary Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X, y)
linear_pred = linear_reg.predict(X)
linear_r2 = r2_score(y, linear_pred)

# Plot the relationship between popularity and song length for the filtered data
plt.figure(figsize=(12, 10))
plt.scatter(filtered_df['popularity'], y, alpha=0.8, label='Actual Data')
#plt.scatter(dfmain['popularity'], y, alpha=0.8, label='Actual Data')
plt.plot(X, linear_pred, color='red', label=f'Linear Regression (R-squared={linear_r2:.2f})')
plt.title('Relationship between Popularity and Song Length (Duration > 12 min  )')
plt.ylabel('Duration (min)')
plt.xlabel('Popularity')
plt.legend()
plt.grid(True)
plt.show()

# Display linear regression statistics
linear_reg_stats = {
    'Intercept': linear_reg.intercept_,
    'Coefficient': linear_reg.coef_[0],
    'R-squared': linear_r2,
}
print("Linear Regression Statistics:")
for key, value in linear_reg_stats.items():
    print(f"{key}: {value}")

from scipy.stats import mannwhitneyu

# Assuming dfmain is your DataFrame
explicit_songs = dfmain[dfmain['explicit'] == 1]['popularity']
non_explicit_songs = dfmain[dfmain['explicit'] == 0]['popularity']

# Perform Mann-Whitney U test
mwu_stat, mwu_pval = mannwhitneyu(explicit_songs, non_explicit_songs, alternative='two-sided')

print(f"Mann-Whitney U Test: U Statistic = {mwu_stat}, p-value = {mwu_pval}")

explicit_median = np.median(explicit_songs)
non_explicit_median = np.median(non_explicit_songs)
explicit_percentiles = np.percentile(explicit_songs, [25, 75])  # 25th and 75th percentiles
non_explicit_percentiles = np.percentile(non_explicit_songs, [25, 75])  # 25th and 75th percentiles


# Create a boxplot with median and percentile annotations
plt.figure(figsize=(8, 6))
plt.boxplot([explicit_songs, non_explicit_songs], labels=['Explicit', 'Non-Explicit'])
plt.title('Popularity Comparison: Explicit vs Non-Explicit Songs')
plt.xlabel('Explicitness')
plt.ylabel('Popularity')
plt.grid(True)

# Annotate medians and percentiles
plt.text(1, explicit_median, f'Median: {explicit_median}', ha='center', va='bottom', color='blue', fontweight='bold')
plt.text(2, non_explicit_median, f'Median: {non_explicit_median}', ha='center', va='bottom', color='blue', fontweight='bold')
plt.text(1, explicit_percentiles[0], f'25th Percentile: {explicit_percentiles[0]}', ha='center', va='top', color='red')
plt.text(1, explicit_percentiles[1], f'75th Percentile: {explicit_percentiles[1]}', ha='center', va='bottom', color='red')
plt.text(2, non_explicit_percentiles[0], f'25th Percentile: {non_explicit_percentiles[0]}', ha='center', va='top', color='red')
plt.text(2, non_explicit_percentiles[1], f'75th Percentile: {non_explicit_percentiles[1]}', ha='center', va='bottom', color='red')

plt.show()

from scipy.stats import mannwhitneyu

# Assuming dfmain is your DataFrame
majorkey_songs = dfmain[dfmain['mode'] == 1]['popularity']
minorkey_songs = dfmain[dfmain['mode'] == 0]['popularity']

# Perform Mann-Whitney U test
mwus_stat, mwus_pval = mannwhitneyu(majorkey_songs, minorkey_songs, alternative='two-sided')

print(f"Mann-Whitney U Test: U Statistic = {mwus_stat}, p-value = {mwus_pval}")

makey_median = np.median(majorkey_songs)
mikey_median = np.median(minorkey_songs)
makey_percentiles = np.percentile(majorkey_songs, [25, 75])  # 25th and 75th percentiles
mikey_percentiles = np.percentile(minorkey_songs, [25, 75])  # 25th and 75th percentiles

plt.figure(figsize=(8, 6))
plt.boxplot([majorkey_songs, minorkey_songs], labels=['Major Key', 'Minor Key'])
plt.title('Popularity Comparison: Major Key vs Minor Key Songs')
plt.xlabel('Type of Key')
plt.ylabel('Popularity')
plt.grid(True)

plt.text(1, makey_median, f'Median: {makey_median}', ha='center', va='bottom', color='blue', fontweight='bold')
plt.text(2, makey_median, f'Median: {mikey_median}', ha='center', va='bottom', color='blue', fontweight='bold')
plt.text(1, makey_percentiles[0], f'25th Percentile: {makey_percentiles[0]}', ha='center', va='top', color='red')
plt.text(1, makey_percentiles[1], f'75th Percentile: {makey_percentiles[1]}', ha='center', va='bottom', color='red')
plt.text(2, mikey_percentiles[0], f'25th Percentile: {mikey_percentiles[0]}', ha='center', va='top', color='red')
plt.text(2, mikey_percentiles[1], f'75th Percentile: {mikey_percentiles[1]}', ha='center', va='bottom', color='red')
 
plt.show()

'''import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Assuming dfmain is your DataFrame
energy = dfmain['energy']
loudness = dfmain['loudness']

#energy = (dfmain[2 ** (dfmain['loudness'])<5]['energy'])

#loudness = 2 ** (dfmain[2 **(dfmain['loudness'])<5]['loudness'])

# Define linear regression function
def linear_regression(x, a, b):
    return a * x + b

# Perform linear regression
popt_lin, _ = curve_fit(linear_regression, energy, loudness)
y_lin_pred = linear_regression(energy, *popt_lin)
r2_lin = r2_score(loudness, y_lin_pred)

# Get the coefficients
a = popt_lin[0]
b = popt_lin[1]

# Create scatter plot
plt.figure(figsize=(12, 10))
plt.scatter(energy, loudness, alpha=0.8, label='Data')
plt.title('Relationship between Energy and Loudness')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.grid(True)

# Plot linear regression line
plt.plot(energy, y_lin_pred, color='red', label=f'Line of Best Fit (R-squared = {r2_lin:.2f}): y = {a:.2f}x + {b:.2f}')

plt.legend()
plt.show()'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Assuming dfmain is your DataFrame
energy = dfmain['energy']
loudness = dfmain['loudness']

# Define linear regression function
def linear_regression(x, a, b):
    return a * x + b

# Define logarithmic regression function
def logarithmic_regression(x, a, b):
    return a * np.log(x) + b

# Perform linear regression
popt_lin, _ = curve_fit(linear_regression, energy, loudness)
y_lin_pred = linear_regression(energy, *popt_lin)
r2_lin = r2_score(loudness, y_lin_pred)

# Perform logarithmic regression
popt_log, _ = curve_fit(logarithmic_regression, energy, loudness)
y_log_pred = logarithmic_regression(energy, *popt_log)
r2_log = r2_score(loudness, y_log_pred)

# Get the coefficients for linear regression
a_lin = popt_lin[0]
b_lin = popt_lin[1]

# Get the coefficients for logarithmic regression
a_log = popt_log[0]
b_log = popt_log[1]

# Create scatter plot
plt.figure(figsize=(12, 10))
plt.scatter(energy, loudness, alpha=0.8, label='Data')
plt.title('Relationship between Energy and Loudness')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.grid(True)

# Plot linear regression line
plt.plot(energy, y_lin_pred, color='red', label=f'Linear Fit (R-squared = {r2_lin:.2f}): y = {a_lin:.2f}x + {b_lin:.2f}')

# Plot logarithmic regression line
plt.plot(sorted(energy), logarithmic_regression(sorted(energy), *popt_log), color='white', label=f'Logarithmic Fit (R-squared = {r2_log:.2f}): y = {a_log:.2f}ln(x) + {b_log:.2f}')

plt.legend()
plt.show()

X = dfmain[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = dfmain['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print("Coefficients:", model.coef_)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target
X = dfmain[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = dfmain['popularity']

# Initialize lists to store results
mse_values = []
r2_values = []

# Loop through each feature and perform regression
for feature_name in X.columns:
    # Extract feature as a single-column DataFrame
    X_feature = X[[feature_name]]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=seed)

    # Fit linear regression model
    model_feature = LinearRegression()
    model_feature.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_feature = model_feature.predict(X_test)

    # Evaluate the model
    mse_feature = mean_squared_error(y_test, y_pred_feature)
    r2_feature = r2_score(y_test, y_pred_feature)
    
    # Append MSE and R-squared values to lists
    mse_values.append(mse_feature)
    r2_values.append(r2_feature)
     
    # Plot scatter plot and line of best fit for each feature
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred_feature, color='red', label='Line of Best Fit')
    plt.title(f'Regression of {feature_name.capitalize()} with Popularity')
    plt.xlabel(feature_name.capitalize())
    plt.ylabel('Popularity')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print MSE and R-squared values for each feature
    print(f"Mean Squared Error ({feature_name.capitalize()}): {mse_feature:.2f}")
    print(f"R-squared Score ({feature_name.capitalize()}): {r2_feature:.2f}\n")

# Print overall MSE and R-squared values
print("Overall Mean Squared Error:", np.mean(mse_values))
print("Overall R-squared Score:", np.mean(r2_values))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# Define features and target
X = dfmain[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = dfmain['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Fit linear regression model
model_all_features = LinearRegression()
model_all_features.fit(X_train, y_train)

# Make predictions on the test set
y_pred_all_features = model_all_features.predict(X_test)

# Evaluate the linear regression model
mse_all_features = mean_squared_error(y_test, y_pred_all_features)
r2_all_features = r2_score(y_test, y_pred_all_features)

print("Linear Regression - Mean Squared Error (All Features):", mse_all_features)
print("Linear Regression - R-squared Score (All Features):", r2_all_features)

# Fit Ridge regression model
model_ridge = Ridge(alpha=0.5)  # You can adjust the alpha value
model_ridge.fit(X_train, y_train)

# Make predictions using Ridge regression
y_pred_ridge = model_ridge.predict(X_test)

# Evaluate the Ridge regression model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression - Mean Squared Error:", mse_ridge)
print("Ridge Regression - R-squared Score:", r2_ridge)

# Fit Lasso regression model
model_lasso = Lasso(alpha=0.1)  # You can adjust the alpha value
model_lasso.fit(X_train, y_train)

# Make predictions using Lasso regression
y_pred_lasso = model_lasso.predict(X_test)

# Evaluate the Lasso regression model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression - Mean Squared Error:", mse_lasso)
print("Lasso Regression - R-squared Score:", r2_lasso)

'''# Fit TensorFlow linear model (single-layer neural network)
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[X_train.shape[1]])
])
model_tf.compile(optimizer='sgd', loss='mean_squared_error')
model_tf.fit(X_train, y_train, epochs=100, verbose=0)  # You can adjust the number of epochs

# Make predictions using TensorFlow model
y_pred_tf = model_tf.predict(X_test).flatten()

# Evaluate the TensorFlow model
mse_tf = mean_squared_error(y_test, y_pred_tf)
r2_tf = r2_score(y_test, y_pred_tf)

print("\nTensorFlow Linear Model - Mean Squared Error:", mse_tf)
print("TensorFlow Linear Model - R-squared Score:", r2_tf)'''

pca = PCA()
pca.fit(dfnum)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
total_variance_explained = explained_variance_ratio.sum()

print("Explained Variance Ratio for Each Principal Component:", explained_variance_ratio)
print("Total Variance Explained by Important Principal Components:", explained_variance_ratio[0])


from numpy.linalg import eig

# Calculate covariance matrix
covariance_matrix = np.cov(dfnum.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(covariance_matrix)

# Apply Kaiser criterion
important_eigenvectors = eigenvectors[:, eigenvalues > 1]
important_eigenvalues = eigenvalues[eigenvalues > 1]


# Compute the total variance explained by important principal components
total_variance_explained = np.sum(important_eigenvalues) / np.sum(eigenvalues)

print("Eigenvalues:")
print(important_eigenvalues)
print("\nEigenvectors:")
print(important_eigenvectors)
print("\nTotal Variance Explained by Important Principal Components:", total_variance_explained)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the data
X = dfmain[['valence']]  # Features
y = dfmain['mode']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test.tolist(), y_pred.tolist())
conf_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
class_report = classification_report(y_test.tolist(), y_pred.tolist())

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plot the logistic regression curve
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X['valence'], y=y, hue=y, palette='viridis')
plt.xlabel('Valence')
plt.ylabel('Mode (0: Minor, 1: Major)')
plt.title('Logistic Regression for Key Prediction')
plt.legend(['Minor Key', 'Major Key'], loc='upper right')

# Plot the logistic regression curve
x_values = np.linspace(X['valence'].min(), X['valence'].max(), 100)
y_prob = model.predict_proba(x_values.reshape(-1, 1))[:, 1]  # Probability of being in major key
plt.plot(x_values, y_prob, color='red')

plt.show()
print(y_test.tolist())
print(y_pred.tolist())

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract TP, FP, TN, FN from the confusion matrix
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)


# Compute ROC curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of being in major key
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
print("AUC:", roc_auc)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming dfnum contains all numerical features and 'mode' column for target variable

# Prepare the data
X = dfnum  # Features
y = dfmain['mode']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

# Initialize an empty dictionary to store results for each feature
results = {}

# Loop through each feature and build logistic regression models
for col in X.columns:
    # Extract the feature
    X_feature = X[[col]]
    
    # Split the feature data into training and testing sets
    X_train_feature, X_test_feature = train_test_split(X_feature, test_size=0.5, random_state=seed)
    
    # Build and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_feature, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_feature)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Store results in the dictionary
    results[col] = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'model': model
    }

    # Plot the logistic regression curve
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_feature[col], y=y, hue=y, palette='viridis')
    plt.xlabel(col.capitalize())
    plt.ylabel('Mode (0: Minor, 1: Major)')
    plt.title(f'Logistic Regression for Key Prediction using {col.capitalize()}')
    plt.legend(['Minor Key', 'Major Key'], loc='upper right')

    # Plot the logistic regression curve
    x_values = np.linspace(X_feature[col].min(), X_feature[col].max(), 100)
    y_prob = model.predict_proba(x_values.reshape(-1, 1))[:, 1]  # Probability of being in major key
    plt.plot(x_values, y_prob, color='red')

    plt.show()

# Print results
for feature, result in results.items():
    print(f"Feature: {feature.capitalize()}")
    print("Accuracy:", result['accuracy'])
    print("Confusion Matrix:\n", result['confusion_matrix'])
    print("Classification Report:\n", result['classification_report'])

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming dfnum contains all numerical features and 'mode' column for target variable

# Prepare the data
X = dfnum  # Features
y = dfmain['mode']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize an empty dictionary to store results for each feature
results = {}

# Loop through each feature and build logistic regression models
for col in X.columns:
    # Extract the feature
    X_feature = X[[col]]
    
    # Split the feature data into training and testing sets
    X_train_feature, X_test_feature, y_train, y_test = train_test_split(X_feature, y, test_size=0.5, random_state=seed)
    
    # Build and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_feature, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_feature)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Calculate ROC curve and AUROC
    y_prob = model.predict_proba(X_test_feature)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Store results in the dictionary
    results[col] = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_curve': (fpr, tpr),
        'roc_auc': roc_auc,
        'model': model
    }

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUROC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {col.capitalize()}')
    plt.legend(loc='lower right')
    plt.show()

# Print results
for feature, result in results.items():
    print(f"Feature: {feature.capitalize()}")
    print("Accuracy:", result['accuracy'])
    print("Confusion Matrix:\n", result['confusion_matrix'])
    print("Classification Report:\n", result['classification_report'])
    print("AUROC:", result['roc_auc'])
    print("ROC Curve Coordinates:", result['roc_curve'])
    print()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Assuming dfmain is your DataFrame containing the song data

# Convert genre label to binary numerical label (classical or not)
dfmain['is_classical'] = dfmain['track_genre'].apply(lambda x: 1 if x == 'classical' else 0)

# Prepare the data
X_duration = dfmain[['duration']]  # Features
y_duration = dfmain['is_classical']  # Target variable

# Split the data into training and testing sets
X_train_duration, X_test_duration, y_train_duration, y_test_duration = train_test_split(X_duration, y_duration, test_size=0.2, random_state=seed)

# Build and train the logistic regression model based on duration
model_duration = LogisticRegression()
model_duration.fit(X_train_duration, y_train_duration)

# Make predictions
y_pred_duration = model_duration.predict(X_test_duration)

# Evaluate the model
accuracy_duration = accuracy_score(y_test_duration, y_pred_duration)
print("Accuracy for Duration Predictor:", accuracy_duration)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_duration, model_duration.predict_proba(X_test_duration)[:,1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plot logistic regression curve
plt.figure(figsize=(8, 6))
plt.scatter(X_duration, y_duration, color='blue', alpha=0.5)
plt.xlabel('Duration')
plt.ylabel('Is Classical (1: Yes, 0: No)')
plt.title('Logistic Regression Curve for Classical Music Prediction based on Duration')
x_values = np.linspace(X_duration.min(), X_duration.max(), 100)
y_prob = model_duration.predict_proba(x_values.reshape(-1, 1))[:, 1]
plt.plot(x_values, y_prob, color='red', label='Logistic Regression Curve')
plt.legend()
plt.show()


# Confusion matrix
conf_matrix = confusion_matrix(y_test_duration, y_pred_duration)
print("Confusion Matrix:\n", conf_matrix)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target
X = np.matmul((dfnum.values), important_eigenvectors)
y = dfmain['is_classical'] 
print(important_eigenvectors.shape, (dfnum.values).shape, x.shape, y.shape)
# Assuming dfnum_pca is your DataFrame containing PCA components
# X_pca should contain the PCA components (3 principal components) as features
# Assuming dfnum is your DataFrame with numerical data

# Split the data into training and testing sets
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Build and train the logistic regression model based on PCA components
model_pca = LogisticRegression()
model_pca.fit(X_train_pca, y_train)

# Make predictions
y_pred_pca = model_pca.predict(X_test_pca)

# Evaluate the model
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Accuracy for PCA Predictor:", accuracy_pca)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define features and target
X = np.matmul(dfnum.values, important_eigenvectors)
y = dfmain['is_classical']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Build and train the logistic regression model based on PCA components
model_pca = LogisticRegression()
model_pca.fit(X_train, y_train)

# Make predictions
y_pred = model_pca.predict(X_test)

# Evaluate the model
accuracy_pca = accuracy_score(y_test, y_pred)
print("Accuracy for PCA Predictor:", accuracy_pca)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Logistic Regression for Classification')
plt.legend(['Not Classical', 'Classical'], loc='upper right')

'''# Plot the logistic regression decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')'''
# Plot the ROC curve
y_prob = model_pca.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Assuming dfmain is your DataFrame containing the song data

# Convert genre label to binary numerical label (classical or not)or
dfmain['is_electronic'] = dfmain['track_genre'].apply(lambda x: 1 if x == 'electronic' or  x == 'anime' or x == 'chill' or x == 'club' or x == 'detroit-techno' or x == 'dubstep' or x == 'edm' else 0)
                                                      

# Prepare the data
X_duration = dfmain[['acousticness']]  # Features
y_duration = dfmain['is_electronic']  # Target variable

# Split the data into training and testing sets
X_train_duration, X_test_duration, y_train_duration, y_test_duration = train_test_split(X_duration, y_duration, test_size=0.2, random_state=seed)

# Build and train the logistic regression model based on duration
model_duration = LogisticRegression()
model_duration.fit(X_train_duration, y_train_duration)

# Make predictions
y_pred_duration = model_duration.predict(X_test_duration)

# Evaluate the model
accuracy_duration = accuracy_score(y_test_duration, y_pred_duration)
print("Accuracy for Duration Predictor:", accuracy_duration)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_duration, model_duration.predict_proba(X_test_duration)[:,1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plot logistic regression curve
plt.figure(figsize=(8, 6))
plt.scatter(X_duration, y_duration, color='blue', alpha=0.5)
plt.xlabel('Acousticness')
plt.ylabel('Is Classical (1: Yes, 0: No)')
plt.title('Logistic Regression Curve for Electronic-Sounding Music Prediction based on Acousticness')
x_values = np.linspace(X_duration.min(), X_duration.max(), 100)
y_prob = model_duration.predict_proba(x_values.reshape(-1, 1))[:, 1]
plt.plot(x_values, y_prob, color='red', label='Logistic Regression Curve')
plt.legend()
plt.show()


# Confusion matrix
conf_matrix = confusion_matrix(y_test_duration, y_pred_duration)
print("Confusion Matrix:\n", conf_matrix)
