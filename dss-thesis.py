
# Importing the needed libraries
# Data Handling 
import numpy as np
import pandas as pd

# Preprocessing 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


# Model Selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# ===== Models =====
from sklearn.linear_model import LogisticRegression        # Logistic Regression
from sklearn.svm import LinearSVC                          # Support Vector Machine
from xgboost import XGBClassifier                          # XGBoost
from xgboost import plot_importance
from sklearn.pipeline import Pipeline

# Evaluation Metrics (Classification)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Visualization 
import matplotlib.pyplot as plt

# Utility
import warnings
warnings.filterwarnings('ignore')
import logging




##########################################################
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Loading the main (raw dataset) --> a CSV file
file_path = "/Users/ivan/Documents/thesis/accepted_2007_to_2018Q4.csv"
df = pd.read_csv(file_path)

#Brief initial familiarization with the dataset
# Print first 5 rows to verify
print(df.head())

# Check basic info
print(df.info())
df.dtypes
df.describe()

# Check if there are any duplicates in the entire dataset
duplicates = df.duplicated()
# Show the number of duplicate rows
print(f"Number of duplicate rows: {duplicates.sum()}")

# Check for missing values
na_counts = df.isna().sum()

# Print the result
print("Number of NAs per column:")
print(na_counts)

#### Exploring the target ####

#Loan status is the outcome variable, it is a categorical one
#Distribution of the loan status variable
# Show distinct values in the loan_status column
print("Unique values in loan_status:")
print(df['loan_status'].unique())

# Number of rows per loan_status value
loan_status_counts = df['loan_status'].value_counts()
print("\nLoan Status Distribution:")
print(loan_status_counts)

# Bar chart of loan_status distribution
plt.figure(figsize=(12, 8))
plt.bar(loan_status_counts.index, loan_status_counts.values, color = 'darkblue')
plt.title("Initial Distribution of Loan Status", fontsize = 16)
plt.xlabel("Loan Status", fontsize = 12)
plt.ylabel("Number of Records", fontsize =12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.ticklabel_format(style='plain', axis='y')  
plt.show()

# Remove rows where loan_status is NaN
df = df[df['loan_status'].notna()]

# Remove rows where loan_status is "Current" or "Does not meet the credit policy"
# The current status is removed because these loans do not have an outcome yet,
# while the goal of this thesis is to predict the outcome of a loan
# List of statuses to remove
statuses_to_remove = [
    "Current",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off"
]

# Filter the DataFrame for the rest of the values in the loan_status column
df = df[~df['loan_status'].isin(statuses_to_remove)]


# Same check as before of the loan_status distribution
loan_status_counts = df['loan_status'].value_counts()
print("\nLoan Status Distribution:")
print(loan_status_counts)


# New distribution (bar chart)
plt.figure(figsize=(12, 8))
plt.bar(loan_status_counts.index, loan_status_counts.values, color = 'darkblue')
plt.title("Distribution of Loan Status", fontsize = 16)
plt.xlabel("Loan Status", fontsize = 12)
plt.ylabel("Number of Records", fontsize =12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.ticklabel_format(style='plain', axis='y')  
plt.show()

# Nas in the left 1 382 351 rows by column
na_counts = df.isna().sum()
print("Number of NAs per column:")
print(na_counts)

# Check for missing values in 'emp_length' and 'emp_title' columns
# as I noticed that these two are having quite some missing values 
# while they can be useful inputs of the models
missing_emp_length = df['emp_length'].isna().sum()
missing_emp_title = df['emp_title'].isna().sum()

print(f"Missing values in 'emp_length': {missing_emp_length}")
print(f"Missing values in 'emp_title': {missing_emp_title}")

# I will only keep columns with no more than 85 000 missing values

print(na_counts[na_counts <= 85000])
print(na_counts[na_counts > 85000])


# 62 columns have more than 85 000 missing values
# 89 columns have less than 85 000 missing values (so I continue with these 89)

# Filter columns with <= 85,000 missing values
columns_to_keep = na_counts[na_counts <= 85000].index

# Subset the df
df_filtered = df[columns_to_keep]

df = df_filtered


# ################################################################################

#### Feature Selection ####

# List of selected columns (with loan status being the outcome variable)
# This list is based on the litrature reviewed

selected_columns = [
    'loan_status', 'loan_amnt', 'term', 'installment', 'emp_length', 'dti', 
    'int_rate', 'home_ownership', 'annual_inc', 'purpose', 'funded_amnt', 'grade',
    'zip_code', 'addr_state', 'fico_range_low', 'fico_range_high', 'open_acc', 
    'out_prncp', 'acc_now_delinq', 'hardship_flag', 'num_accts_ever_120_pd', 
    'verification_status', 'pub_rec', 'num_sats','total_acc', 
    'num_tl_30dpd', 'delinq_amnt', 'application_type', 
    'chargeoff_within_12_mths', 'pub_rec_bankruptcies', 'issue_d', 'revol_bal'
]


# The new df contains 32 columns and 1 379 602 rows

# Create a new df with the selected columns
df_selected = df[selected_columns]

# Print the number of columns
print(f"Number of columns: {len(df_selected.columns)}")
print(df_selected.head())

# Count NAs at the column level and sort by the number of missing values in descending order
na_column_level_sorted = df_selected.isna().sum().sort_values(ascending=False)

# Print the sorted result
print("Columns ordered by the number of missing values:")
print(na_column_level_sorted)

# only 8 of the columns have missing values. 
#Zip code is the only categorical variable with missing values and it has only 1 missing

# Columns ordered by the number of missing values:
# emp_length                  81403
# num_tl_30dpd                67527
# num_accts_ever_120_pd       67527
# num_sats                    55841
# pub_rec_bankruptcies          697
# dti                           412
# chargeoff_within_12_mths       56
# zip_code                        1

# Count NAs at the row level
na_row_level = df_selected.isna().sum(axis=1)

# Sort by the number of missing values in descending order
na_row_level_sorted = na_row_level.sort_values(ascending=False)

# Print the sorted result (optional: display the first few rows)
print("Rows ordered by the number of missing values:")
print(na_row_level_sorted)

#In this new df_selected the highest number of missing values is 5
#Imputation will be applied on those in a pipeline


#Numeric vs Categorical columns splitted before Imputation is applied
# Identify columns with missing values
missing_columns = df_selected.columns[df_selected.isna().sum() > 0]
print(missing_columns)

df_selected[missing_columns].dtypes


#The emp_length column is not in a numeric format
print("Unique values in emp_length column:")
print(df_selected['emp_length'].value_counts(dropna=False))

# Mapping dictionary for emp_length
emp_length_mapping = {
    '10+ years': 10,  
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0.5,
    'n/a': np.nan 
}

# Applying the mapping to the 'emp_length' column
df_selected['emp_length'] = df_selected['emp_length'].map(emp_length_mapping)

# Check the unique values in the 'emp_length' column after mapping
print("Unique values in 'emp_length' after mapping:")
print(df_selected['emp_length'].value_counts(dropna=False))

# The term column is in the format '36 months'  --> It will be converted to numerical keeping only the numbers and not the word 'months'
df_selected['term'] = df_selected['term'].str.extract('(\d+)').astype(float)

# Now all the columns with missing values except the zip_code column are float64 data type

#Zip code will be treated differently with regards to imputation (compared to numeric columns)
#The most frequent zip_code within the specific state will be chosen

# Group by 'state' and get the most frequent zip code for each state
most_frequent_zip_by_state = df_selected.groupby('addr_state')['zip_code'].agg(lambda x: x.mode()[0])

# Function to impute missing zip codes based on state
def impute_zip_code(row, most_frequent_zip_by_state):
    if pd.isna(row['zip_code']):
        return most_frequent_zip_by_state[row['addr_state']]  
    return row['zip_code']

# Apply the imputation function
df_selected['zip_code'] = df_selected.apply(impute_zip_code, axis=1, most_frequent_zip_by_state=most_frequent_zip_by_state)

# Correlation Matrix

df_numeric = df_selected.select_dtypes(include=['number'])
df_numeric.head()
# Compute correlation matrix
corr_matrix = df_numeric.corr().round(2)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 12))

# Create the heatmap using imshow
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add colorbar
fig.colorbar(cax)

# Set ticks and labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

# Add the correlation values inside the heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, corr_matrix.iloc[i, j],
                       ha="center", va="center", color="black", fontsize=8)

# Set title and layout
plt.title('Correlation Matrix (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()


### FEATURE ENGINEERING ###
#Extractig only the year from the issue_d column which will be later used for creating an interest rate column 

# Convert the string to datetime object (replace 'your_date_column' with the actual column name)
df_selected['issue_d'] = pd.to_datetime(df_selected['issue_d'])

# Extract the year from the datetime column
df_selected['issue_year'] = df_selected['issue_d'].dt.year

# Print the result
print(df_selected[['issue_d', 'issue_year']].head())

df_selected['issue_year'].dtypes

### EDA - distributions and outliers ###

# Plot for home_ownership_modified
plt.figure(figsize=(8, 6))
home_ownership_counts = df_selected['home_ownership_modified'].value_counts()
plt.bar(home_ownership_counts.index, home_ownership_counts.values)
plt.title('Distribution of home_ownership_modified')
plt.xlabel('home_ownership_modified')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot for 'purpose'
plt.figure(figsize=(8, 6))
purpose_counts = df_selected['purpose'].value_counts()
plt.bar(purpose_counts.index, purpose_counts.values)
plt.title('Distribution of purpose')
plt.xlabel('purpose')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot for 'grade'
plt.figure(figsize=(8, 6))
verification_status_counts = df_selected['grade'].value_counts()
plt.bar(verification_status_counts.index, verification_status_counts.values)
plt.title('Distribution of grade_status')
plt.xlabel('grade')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# Many different zips --> think what to do with this column, maybe just have the analysis by state
# Plot for 'zip_modified'
plt.figure(figsize=(8, 6))
verification_status_counts = df_selected['zip_modified'].value_counts()
plt.bar(verification_status_counts.index, verification_status_counts.values)
plt.title('Distribution of zip_modified')
plt.xlabel('zip_modified')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot for 'verification_status'
plt.figure(figsize=(8, 6))
verification_status_counts = df_selected['verification_status'].value_counts()
plt.bar(verification_status_counts.index, verification_status_counts.values)
plt.title('Distribution of verification_status')
plt.xlabel('verification_status')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot for 'application_type'
plt.figure(figsize=(8, 6))
application_type_counts = df_selected['application_type'].value_counts()
plt.bar(application_type_counts.index, application_type_counts.values)
plt.title('Distribution of application_type')
plt.xlabel('application_type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# Plot for 'addr_state'
plt.figure(figsize=(8, 6))
addr_state_counts = df_selected['addr_state'].value_counts()
plt.bar(addr_state_counts.index, addr_state_counts.values)
plt.title('Distribution of addr_state')
plt.xlabel('addr_state')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot for 'hardship_flag'
plt.figure(figsize=(8, 6))
hardship_flag_counts = df_selected['hardship_flag'].value_counts()
plt.bar(hardship_flag_counts.index, hardship_flag_counts.values)
plt.title('Distribution of hardship_flag')
plt.xlabel('hardship_flag')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# 
hardship_counts = df_selected['hardship_flag'].value_counts()
print(hardship_counts)
# hardship_flag
# N    1378774
# Y        828
# Name: count, dtype: int64


# Check distinct values in the 'purpose' column
distinct_purpose_values = df_selected['purpose'].unique()
print(distinct_purpose_values)


# Count the occurrences of each distinct value in the 'purpose' column
purpose_counts = df_selected['purpose'].value_counts()

print(purpose_counts)

# debt_consolidation    800337
# credit_card           301791
# home_improvement       89831
# other                  80361
# major_purchase         30290
# medical                16005
# small_business         15937
# car                    14885
# moving                  9760
# vacation                9289
# house                   7544
# wedding                 2294
# renewable_energy         952
# educational              326

# Boxplots to check for outliers

# Plot for 'loan_amnt'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['loan_amnt'].dropna())
plt.title('Boxplot of loan_amnt')
plt.xlabel('loan_amnt')
plt.ylabel('Value')
plt.show()

# Plot for 'installment'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['installment'].dropna())
plt.title('Boxplot of installment')
plt.xlabel('installment')
plt.ylabel('Value')
plt.show()

# Plot for 'emp_length'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['emp_length'].dropna())
plt.title('Boxplot of emp_length')
plt.xlabel('emp_length')
plt.ylabel('Value')
plt.show()

# Plot for 'dti'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['dti'].dropna())
plt.title('Boxplot of dti')
plt.xlabel('dti')
plt.ylabel('Value')
plt.show()

# Plot for 'int_rate'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['int_rate'].dropna())
plt.title('Boxplot of int_rate')
plt.xlabel('int_rate')
plt.ylabel('Value')
plt.show()

# Plot for 'annual_inc'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['annual_inc'].dropna())
plt.title('Boxplot of annual_inc')
plt.xlabel('annual_inc')
plt.ylabel('Value')
plt.show()

# Plot for 'open_acc'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['open_acc'].dropna())
plt.title('Boxplot of open_acc')
plt.xlabel('open_acc')
plt.ylabel('Value')
plt.show()

# Plot for 'acc_now_delinq'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['acc_now_delinq'].dropna())
plt.title('Boxplot of acc_now_delinq')
plt.xlabel('acc_now_delinq')
plt.ylabel('Value')
plt.show()

# Plot for 'num_accts_ever_120_pd'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['num_accts_ever_120_pd'].dropna())
plt.title('Boxplot of num_accts_ever_120_pd')
plt.xlabel('num_accts_ever_120_pd')
plt.ylabel('Value')
plt.show()

# Plot for 'pub_rec'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['pub_rec'].dropna())
plt.title('Boxplot of pub_rec')
plt.xlabel('pub_rec')
plt.ylabel('Value')
plt.show()

# Plot for 'total_acc'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['total_acc'].dropna())
plt.title('Boxplot of total_acc')
plt.xlabel('total_acc')
plt.ylabel('Value')
plt.show()

# Plot for 'num_tl_30dpd'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['num_tl_30dpd'].dropna())
plt.title('Boxplot of num_tl_30dpd')
plt.xlabel('num_tl_30dpd')
plt.ylabel('Value')
plt.show()

# Plot for 'delinq_amnt'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['delinq_amnt'].dropna())
plt.title('Boxplot of delinq_amnt')
plt.xlabel('delinq_amnt')
plt.ylabel('Value')
plt.show()

# Plot for 'chargeoff_within_12_mths'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['chargeoff_within_12_mths'].dropna())
plt.title('Boxplot of chargeoff_within_12_mths')
plt.xlabel('chargeoff_within_12_mths')
plt.ylabel('Value')
plt.show()

# Plot for 'pub_rec_bankruptcies'
plt.figure(figsize=(8, 6))
plt.boxplot(df_selected['pub_rec_bankruptcies'].dropna())
plt.title('Boxplot of pub_rec_bankruptcies')
plt.xlabel('pub_rec_bankruptcies')
plt.ylabel('Value')
plt.show()

#Violin plots

import matplotlib.pyplot as plt

# Helper function to plot a violin chart in light purple
def plot_violin(data, title, xlabel):
    plt.figure(figsize=(8, 6))
    vp = plt.violinplot(data.dropna(), showmeans=True, showextrema=True)
    
    for body in vp['bodies']:
        body.set_facecolor('#D8BFD8')  # Light purple hex color
        body.set_edgecolor('black')
        body.set_alpha(0.9)

    plt.title(f'Violin Plot of {title}')
    plt.xlabel(xlabel)
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Plot for each feature
plot_violin(df_selected['loan_amnt'], 'loan_amnt', 'loan_amnt')
plot_violin(df_selected['installment'], 'installment', 'installment')
plot_violin(df_selected['emp_length'], 'emp_length', 'emp_length')
plot_violin(df_selected['dti'], 'dti', 'dti')
plot_violin(df_selected['int_rate'], 'int_rate', 'int_rate')
plot_violin(df_selected['annual_inc'], 'annual_inc', 'annual_inc')
plot_violin(df_selected['open_acc'], 'open_acc', 'open_acc')
plot_violin(df_selected['acc_now_delinq'], 'acc_now_delinq', 'acc_now_delinq')
plot_violin(df_selected['num_accts_ever_120_pd'], 'num_accts_ever_120_pd', 'num_accts_ever_120_pd')
plot_violin(df_selected['pub_rec'], 'pub_rec', 'pub_rec')
plot_violin(df_selected['total_acc'], 'total_acc', 'total_acc')
plot_violin(df_selected['num_tl_30dpd'], 'num_tl_30dpd', 'num_tl_30dpd')
plot_violin(df_selected['delinq_amnt'], 'delinq_amnt', 'delinq_amnt')
plot_violin(df_selected['chargeoff_within_12_mths'], 'chargeoff_within_12_mths', 'chargeoff_within_12_mths')
plot_violin(df_selected['pub_rec_bankruptcies'], 'pub_rec_bankruptcies', 'pub_rec_bankruptcies')


# Count loans per year
loans_per_year = df_selected['issue_year'].value_counts().sort_index()
print(loans_per_year)

# 2007       251
# 2008      1562
# 2009      4716
# 2010     11536
# 2011     21721
# 2012     53367
# 2013    134808
# 2014    223710
# 2015    377796
# 2016    300346
# 2017    181728
# 2018     68061


# Plot
plt.figure(figsize=(10, 6))
plt.bar(loans_per_year.index.astype(str), loans_per_year.values)
plt.title('Number of Loans Issued per Year')
plt.xlabel('Year')
plt.ylabel('Number of Loans')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Years vary from 2007 to 2018 in this current dataframe
#This will be used for the mapping with the microeconomic data

### FEATURE ENGINEERING Ctnd. ###

#Zip code has only the first three digits visible for anonymization purposes
df_selected['zip_modified'] = df_selected['zip_code'].str[:3]

#Because of high correlation between fico_range_low and fico_range_high, I will take the average of these two
df_selected['fico_average'] = (df_selected['fico_range_low'] + df_selected['fico_range_high'])/2
df_selected = df_selected.drop(['fico_range_low', 'fico_range_high', 'zip_code', 'issue_d'], axis = 1)


# Pregrrouping of the home_ownership columns

conditions = [
    df_selected['home_ownership'] == 'MORTGAGE',
    df_selected['home_ownership'] == 'OWN',
    df_selected['home_ownership'] == 'RENT',
    df_selected['home_ownership'].isin(['OTHER', 'NONE', 'ANY'])
]

choices = ['MORTGAGE', 'OWN', 'RENT', 'OTHER/NONE']

df_selected['home_ownership_modified'] = np.select(conditions, choices, default='UNKNOWN')
print(df_selected['home_ownership_modified'].value_counts())

df_selected = df_selected.drop(['home_ownership'], axis = 1)

#Exploratory Data Analysis

df_numeric = df_selected.select_dtypes(include=['number'])
df_numeric.head()

df_categorical = df_selected.select_dtypes(exclude=['number'])
df_categorical.head()


df_numeric.hist(bins=30, figsize=(20,10))
plt.tight_layout()
plt.show()

df_categorical = df_selected.select_dtypes(exclude=['number'])
df_categorical.head()

#####

# I want to check if this record that is such an outlier in the pub_rec column is also an outlier for the other variables
# If so, it will be removed from the data
df_selected[df_selected['pub_rec'] > 80]


#EDA shows a high correlation between these two, therefore I want to check the matching
df_selected['Equality'] = df_selected['loan_amnt'] == df_selected['funded_amnt']
df_selected['Equality'].value_counts()
# True     1380289
# False       2062
#In about 99.9% of the cases the funded amount = loan amount, 
# so only one of these columns will be kept

df_selected = df_selected.drop(['Equality', 'loan_amnt'], axis = 1)

# Rename the 'funded_amnt' column to 'funded_amnt(loan_amnt)'
df_selected = df_selected.rename(columns={'funded_amnt': 'funded_amnt(loan_amnt)'})

# Display the DataFrame to confirm the change
df_selected.head()
# loan amnt is the requested amount, while funded amount is what is actually received
#therefore only the funded amount will be kept

#num_sats column will be remove as it can contain current loans (i.e., loans without yet an outcome)
#it also creates a correlation of 1.0 with open_acc which might be more valuable for the model
df_selected = df_selected.drop(['num_sats'], axis = 1)


df_selected.info()
#This current dataframe contains 1379602 rows and 30 columns


### Rerunning correlation matrix ###
df_numeric = df_selected.select_dtypes(include=['number'])
df_numeric.head()

# Compute correlation matrix
corr_matrix = df_numeric.corr().round(2)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 12))

# Create the heatmap using imshow
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add colorbar
fig.colorbar(cax)

# Set ticks and labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

# Add the correlation values inside the heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, corr_matrix.iloc[i, j],
                       ha="center", va="center", color="black", fontsize=8)

# Set title and layout
plt.title('Correlation Matrix (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()


# 1. Select only the features (drop target and non-numeric if necessary)
X = df_selected.drop(['loan_status', 'id', 'issue_d', 'zip_code', 'addr_state'], axis=1)

# If you still have categorical variables, encode them first:
X = pd.get_dummies(X, drop_first=True)

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4. Explained variance plot

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

#The graph is pretty linear so pca will not be used for dimensionality reduction

#### FEATURE ENGINEERING Cntd.

#Categorical variables encoded
df_selected['is_verified'] = df_selected['verification_status'].isin(['Verified', 'Source Verified'])
df_selected = df_selected.drop(['verification_status'], axis = 1)
# In this way will be as a boolean column, will make more sense, and will be easier for encoding


# print("Unique values in 'hardship_flag':", df_selected['hardship_flag'].unique())
# print("Unique values in 'is_verified':", df_selected['is_verified'].unique())
# print("Unique values in 'application_type':", df_selected['application_type'].unique())
df_selected['is_individual'] = df_selected['application_type'].isin(['Individual'])
df_selected = df_selected.drop(['application_type'], axis = 1)


# Unique values in 'hardship_flag': ['N' 'Y']
# Unique values in 'is_verified': [False  True]
# Unique values in 'is_individual': [ True False]

# Convert 'hardship_flag' where 'Y' → 1 and 'N' → 0
df_selected['hardship_flag'] = (df_selected['hardship_flag'] == 'Y').astype(int)

# Convert 'is_verified' from boolean to int (True → 1, False → 0)
df_selected['is_verified'] = df_selected['is_verified'].astype(int)

# Convert 'is_individual' from boolean to int (True → 1, False → 0)
df_selected['is_individual'] = df_selected['is_individual'].astype(int)

#REMOVING THE ZIP FOR NOW
df_selected = df_selected.drop(['zip_modified'], axis = 1)



# Handling of categorical variables (individual approach)
df_selected = pd.get_dummies(df_selected, columns=['home_ownership_modified'], drop_first=True, dtype = int)

le = LabelEncoder()
df_selected['loan_status'] = le.fit_transform(df_selected['loan_status'])

for col in ['purpose', 'addr_state']:
    freq_encoding = df_selected[col].value_counts(normalize=True)
    df_selected[col + '_freq'] = df_selected[col].map(freq_encoding)

# Drop the original columns
df_selected.drop(['purpose', 'addr_state'], axis=1, inplace=True)

grade_order = [['G', 'F', 'E', 'D', 'C', 'B', 'A']]
ordinal_enc = OrdinalEncoder(categories=grade_order)

df_selected['grade'] = ordinal_enc.fit_transform(df_selected[['grade']]).astype(int)

# Importing macroeconomic data
imf_df = pd.read_csv("/Users/ivan/Documents/thesis/macroeconomic_variables.csv")
imf_df.head()

imf_df.dtypes

#Merging with the main dataset
df_selected = df_selected.merge(imf_df, how='left', left_on='issue_year', right_on='matching_year')
df_selected = df_selected.drop(['Year', 'matching_year'], axis =1)

# 34 columns, 1379602 rows


#Rerunning the correlation matrix

df_numeric = df_selected.select_dtypes(include=['number'])
df_numeric.head()
# Compute correlation matrix
corr_matrix = df_numeric.corr().round(2)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 12))

# Create the heatmap using imshow
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add colorbar
fig.colorbar(cax)

# Set ticks and labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

# Add the correlation values inside the heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, corr_matrix.iloc[i, j],
                       ha="center", va="center", color="black", fontsize=8)

# Set title and layout
plt.title('Correlation Matrix (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()


########### MODELLING #########
# Separate features and target
X = df_selected.drop(['loan_status'], axis=1)
y = df_selected['loan_status']

# Stratified splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
logging.info("Data split completed.")

# Baseline Model (predicting the majority class)
logging.info("Running baseline model...")

majority_class = y_train.mode()[0]
baseline_preds = [majority_class] * len(y_test)

logging.info(f"Baseline Accuracy: {accuracy_score(y_test, baseline_preds):.4f}")
logging.info(f"Baseline Precision: {precision_score(y_test, baseline_preds, average='macro', zero_division=0):.4f}")
logging.info(f"Baseline Recall: {recall_score(y_test, baseline_preds, average='macro', zero_division=0):.4f}")
logging.info(f"Baseline F1 Score: {f1_score(y_test, baseline_preds, average='macro', zero_division=0):.4f}")

# 2025-04-27 14:19:17,192 - INFO - Running baseline model...
# 2025-04-27 14:19:17,334 - INFO - Baseline Accuracy: 0.7805
# 2025-04-27 14:19:17,481 - INFO - Baseline Precision: 0.1301
# 2025-04-27 14:19:17,599 - INFO - Baseline Recall: 0.1667
# 2025-04-27 14:19:17,713 - INFO - Baseline F1 Score: 0.1461
#This is indeed the absolute minimum the models should perform

# Setup logging for detailed progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Define imputers and scalers

# Simple vs Iterative Imputation, Standard vs Robust Scaler vs None
imputers = [SimpleImputer(), IterativeImputer(max_iter=5, random_state=42)]
scalers = [StandardScaler(), RobustScaler(), None]  # None means no scalin

## Logistic Regression model ##

# Baseline LR
log_reg = LogisticRegression(random_state=42, 
                             solver = 'saga',
                             class_weight='balanced', 
                             max_iter=10000)

# Define Pipeline and Parameter Grid 

# Create a pipeline
pipeline_lr = Pipeline([
    ('imputer', SimpleImputer()),  
    ('scaler', StandardScaler()),
    ('logreg', log_reg)             
])

# Model evaluation (priorr tuning)
pipeline_lr.fit(X_train, y_train)
test_initial_lr = pipeline_lr.predict(X_test)

logger.info("Before Hyperparameter Tuning")
logger.info(f"Validation Accuracy: {accuracy_score(y_test, test_initial_lr):.4f}")
logger.info(f"Validation Precision (Macro): {precision_score(y_test, test_initial_lr, average='macro', zero_division=0):.4f}")
logger.info(f"Validation Recall (Macro): {recall_score(y_test, test_initial_lr, average='macro', zero_division=0):.4f}")
logger.info(f"Validation F1 Score (Macro): {f1_score(y_test, test_initial_lr, average='macro', zero_division=0):.4f}")

# INFO - Before Hyperparameter Tuning
# INFO - Validation Accuracy: 0.6516
# INFO - Validation Precision (Macro): 0.3921
# INFO - Validation Recall (Macro): 0.3936
# INFO - Validation F1 Score (Macro): 0.3457

# Hyperparameter tuning
# GridSearchCV parameter grid
param_grid = {
    'imputer': imputers,
    'scaler': scalers,
    'logreg__C': [0.01, 0.1, 1.0, 10.0],  
    'logreg__penalty': ['l2'],  
}

# 3-fold cross-validation (stratifiied)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline_lr, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=3)

# Fit the Grid Search Model

# Logging the start of grid search
logger.info("Starting grid search with 3-fold cross-validation for Logistic Regression...")

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best model
best_model_lr = grid_search.best_estimator_

# Log the best parameters
logger.info(f"Best parameters found: {grid_search.best_params_}")
# Best parameters found: {'imputer': IterativeImputer(max_iter=5, random_state=42), 
# 'logreg__C': 1.0, 'logreg__penalty': 'l2', 'scaler': StandardScaler()}


# Predict on the test set
test_preds_lr = best_model_lr.predict(X_test)

# Evaluate test set
test_accuracy = accuracy_score(y_test, test_preds_lr)
test_precision = precision_score(y_test, test_preds_lr, average='macro', zero_division=0)
test_recall = recall_score(y_test, test_preds_lr, average='macro', zero_division=0)
test_f1 = f1_score(y_test, test_preds_lr, average='macro', zero_division=0)

logger.info("After Hyperparameter Tuning")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")
logger.info(f"Test Precision (Macro): {test_precision:.4f}")
logger.info(f"Test Recall (Macro): {test_recall:.4f}")
logger.info(f"Test F1 Score (Macro): {test_f1:.4f}")
# INFO - After Hyperparameter Tuning
# INFO - Test Accuracy: 0.6522
# INFO - Test Precision (Macro): 0.3950
# INFO - Test Recall (Macro): 0.4021
# INFO - Test F1 Score (Macro): 0.3536

# -----------------------------------------------------------------------------------
#SVC model

# SVC model (Multiclass) - baseline
# Due to limitations LinearSVC is used
svc = LinearSVC(random_state=42, class_weight = 'balanced', max_iter =1000)

# Define Pipeline and Parameter Grid

pipeline_svc = Pipeline([
    ('imputer', SimpleImputer()), 
    ('scaler', StandardScaler()),
    ('svc', svc)
])

# Traiin and model evaluation before tuning
pipeline_svc.fit(X_train, y_train)
test_initial = pipeline_svc.predict(X_test)

logger.info("Before Hyperparameter Tuning")
logger.info(f"Validation Accuracy: {accuracy_score(y_test, test_initial):.4f}")
logger.info(f"Validation Precision (Macro): {precision_score(y_test, test_initial, average='macro', zero_division=0):.4f}")
logger.info(f"Validation Recall (Macro): {recall_score(y_test, test_initial, average='macro', zero_division=0):.4f}")
logger.info(f"Validation F1 Score (Macro): {f1_score(y_test, test_initial, average='macro', zero_division=0):.4f}")
# INFO - Before Hyperparameter Tuning
# INFO - Validation Accuracy: 0.7681
# INFO - Validation Precision (Macro): 0.3993
# INFO - Validation Recall (Macro): 0.3753
# INFO - Validation F1 Score (Macro): 0.3324


# GridSearchCV parameter grid
param_grid = {
    'imputer': imputers,
    'scaler': scalers,
    'svc__C': [0.01, 0.1, 1, 10]
}

# 3-fold cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline_svc, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=3)
# Fit the Grid Search Model

# Logging the start of grid search
logger.info("Starting grid search with 3-fold cross-validation...")

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model_svc = grid_search.best_estimator_

# Log the best parameters
logger.info(f"Best parameters found: {grid_search.best_params_}")
#Best parameters found: {'imputer': SimpleImputer(), 
# 'scaler': StandardScaler(), svc__C': 10}

# Evaluation on test set
test_preds = best_model_svc.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='macro', zero_division=0)
test_recall = recall_score(y_test, test_preds, average='macro', zero_division=0)
test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)

logger.info("After Hyperparameter Tuning")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")
logger.info(f"Test Precision (Macro): {test_precision:.4f}")
logger.info(f"Test Recall (Macro): {test_recall:.4f}")
logger.info(f"Test F1 Score (Macro): {test_f1:.4f}")
# INFO - After Hyperparameter Tuning
# INFO - Test Accuracy: 0.7681
# INFO - Test Precision (Macro): 0.4003
# INFO - Test Recall (Macro): 0.3756
# INFO - Test F1 Score (Macro): 0.3328

#XGBoost
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(y_train.unique()), 
    eval_metric='mlogloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=42
)


pipeline_xbg = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('xgb', xgb)
])


# Train and Evaluate Initial Model
pipeline_xbg.fit(X_train, y_train)
test_initial_xbg= pipeline_xbg.predict(X_test)

logger.info("Before Hyperparameter Tuning")
logger.info(f"Validation Accuracy: {accuracy_score(y_test, test_initial_xbg):.4f}")
logger.info(f"Validation Precision (Macro): {precision_score(y_test, test_initial_xbg, average='macro', zero_division=0):.4f}")
logger.info(f"Validation Recall (Macro): {recall_score(y_test, test_initial_xbg, average='macro', zero_division=0):.4f}")
logger.info(f"Validation F1 Score (Macro): {f1_score(y_test, test_initial_xbg, average='macro', zero_division=0):.4f}")
# INFO - Before Hyperparameter Tuning
# INFO - Validation Accuracy: 0.8012
# INFO - Validation Precision (Macro): 0.4424
# INFO - Validation Recall (Macro): 0.3545
# INFO - Validation F1 Score (Macro): 0.3299

# Grid Search with Cross-Validation
param_grid = {
    'imputer': imputers,
    'scaler': scalers,
    'xgb__max_depth': [3, 6],
    'xgb__learning_rate': [0.01, 0.1, 0.3],
    'xgb__n_estimators': [50, 100, 200]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search_xbg = GridSearchCV(pipeline_xbg, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=3)

logger.info("Starting Grid Search...")
grid_search_xbg.fit(X_train, y_train)

logger.info(f"Best Parameters: {grid_search_xbg.best_params_}")
best_model_xbg = grid_search_xbg.best_estimator_
# Best Parameters: {'imputer': SimpleImputer(), 'scaler': StandardScaler(), 
# 'xgb__learning_rate': 0.3, 'xgb__max_depth': 6, 'xgb__n_estimators': 200}

test_tuned = best_model_xbg.predict(X_test)

logger.info("After Hyperparameter Tuning")
logger.info(f"Validation Accuracy: {accuracy_score(y_test, test_tuned):.4f}")
logger.info(f"Validation Precision (Macro): {precision_score(y_test, test_tuned, average='macro', zero_division=0):.4f}")
logger.info(f"Validation Recall (Macro): {recall_score(y_test, test_tuned, average='macro', zero_division=0):.4f}")
logger.info(f"Validation F1 Score (Macro): {f1_score(y_test, test_tuned, average='macro', zero_division=0):.4f}")
# INFO - After Hyperparameter Tuning
# INFO - Validation Accuracy: 0.8006
# INFO - Validation Precision (Macro): 0.4238
# INFO - Validation Recall (Macro): 0.3579
# INFO - Validation F1 Score (Macro): 0.3404


###### Per class evaluation (LR + XGBoost) ######
### Logistic regression:

# Print detailed classification report
report_lr = classification_report(y_test, test_preds_lr, target_names=le.classes_, zero_division=0)
print(report_lr)
# precision    recall  f1-score   support

#        Charged Off       0.32      0.64      0.43     80568
#            Default       0.00      0.25      0.00        12
#         Fully Paid       0.88      0.67      0.76    323025
#    In Grace Period       0.32      0.32      0.32      2531
#  Late (16-30 days)       0.14      0.21      0.17      1305
# Late (31-120 days)       0.70      0.32      0.44      6440

#           accuracy                           0.65    413881
#          macro avg       0.40      0.40      0.35    413881
#       weighted avg       0.76      0.65      0.69    413881

### XGBoost
# Print detailed classification report
report_xgb = classification_report(y_test, test_tuned, target_names=le.classes_, zero_division=0)
print(report_xgb)
# precision    recall  f1-score   support

#        Charged Off       0.57      0.11      0.18     80568
#            Default       0.00      0.00      0.00        12
#         Fully Paid       0.81      0.98      0.89    323025
#    In Grace Period       0.36      0.13      0.19      2531
#  Late (16-30 days)       0.16      0.01      0.03      1305
# Late (31-120 days)       0.65      0.92      0.76      6440

#           accuracy                           0.80    413881
#          macro avg       0.42      0.36      0.34    413881
#       weighted avg       0.76      0.80      0.74    413881

###### Feature importance (XGBoost) #######
# using the built in function

xgb_model_v2 = best_model_xbg.named_steps['xgb']
importances = xgb_model_v2.feature_importances_
feature_names = X_train.columns

# Relative importance
importances_percent = importances / importances.sum()

# df for easy sorting and plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances_percent
})

# Top 20 features selected
top_20 = importance_df.sort_values(by='Importance', ascending=False).head(20)

# Generate a color map in blue-purple range
colors = plt.cm.magma(np.linspace(0.3, 0.9, 20))

# Plot
plt.figure(figsize=(10, 8))
plt.barh(top_20['Feature'], top_20['Importance'], color=colors)
plt.xlabel('Relative Importance')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.gca().invert_yaxis() 
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#List of top 20 features and the % imporrtance
importances = xgb_model_v2.feature_importances_
importances_percent = 100 * importances / importances.sum()
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance (%)': importances_percent
}).sort_values(by='Importance (%)', ascending=False)
print(importance_df.head(20).to_string(index=False))

# Feature  Importance (%)
#                        grade       48.388149
#                    out_prncp       21.219131
#                         term        6.253744
# home_ownership_modified_RENT        3.130203
#                   issue_year        2.113601
#                hardship_flag        2.018114
#            Unemployment Rate        1.700977
#                     int_rate        1.696792
#                    Inflation        1.219060
#                  is_verified        0.800905
#                 num_tl_30dpd        0.708813
#                   emp_length        0.698229
#                 fico_average        0.669055
#       funded_amnt(loan_amnt)        0.651107
#                          dti        0.631146
#                is_individual        0.625122
#               GDP per Capita        0.620735
#  home_ownership_modified_OWN        0.601826
#                   annual_inc        0.597923
#              addr_state_freq        0.546447
