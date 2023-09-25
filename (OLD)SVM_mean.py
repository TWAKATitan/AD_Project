import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

matching_features = []

# Read Excel file, treating '--' as missing values
data = pd.read_excel('ExampleDATA_FDG_SUVR.xlsx', na_values='--')

# 提取特徵和目標
X = data.iloc[259:708, 13:]  
y = data['Group'].iloc[259:708]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Fill missing values with mean
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature selection - Use SelectKBest to select the top k most important features
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Create an SVM classifier
classifier = svm.SVC()  # Default parameters

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Use grid search to find the best hyperparameter combination
grid_search = GridSearchCV(classifier, param_grid, cv=10)
grid_search.fit(X_train_scaled, y_train)

# Output the best parameters
print("Best Parameters: ", grid_search.best_params_)

# Make predictions using the model with the best parameters
best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test_scaled)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy * 100))
print('F1 Score: ', "%.2f" % (f1 * 100))

# Get the indices of the features selected by the best model
selected_feature_indices = selector.get_support(indices=True)

# Get the column names of the selected features
selected_feature_names = data.columns[13:][selected_feature_indices]

# Output the column names of the selected features
print("Selected Features:")
for feature in selected_feature_names:
    print(feature)

# Create a new DataFrame for merged features
merged_features_df = data.copy()  # Make a copy of the original DataFrame
columns_to_copy = ['Group'] + list(selected_feature_names)

# Create a new DataFrame by copying only the specified columns
selected_features_df = data[columns_to_copy].copy()




# 初始化一個空的匹配特徵列表
matching_features = []
matching_feature_names = []
prefix = []
suffix = []

# 遍歷所選特徵名稱

for i, feature_name in enumerate(selected_feature_names):

    # 提取特徵名稱的前三個字母
    prefix.append(feature_name[:-5])
    
    
    # 提取特徵名稱的後五個字母
    suffix.append(feature_name[-5:])
    
    

    # 檢查後綴是否符合所需模式
    if (prefix[i] == prefix[i - 1]) and (suffix[i] != suffix[i-1]) and suffix[i] in {'R_o_o', 'L_o_o'}:

        matching_feature_names.append(feature_name)
        matching_feature_names.append(selected_feature_names[i-1])
        
        # 計算匹配特徵的平均值
        merged_feature = data[matching_feature_names].mean(axis=1)

        # 將匹配特徵添加到具有新名稱的 DataFrame 中
        new_feature_name = feature_name[:-5] + "_avg"
        merged_features_df[new_feature_name] = merged_feature
        #selected_features_df[new_feature_name] = merged_feature

        # 將匹配特徵名稱添加到列表以供稍後刪除
        
        matching_features.extend(matching_feature_names)
    
        
        matching_feature_names = []
    
# 刪除與匹配特徵相對應的列
merged_features_df.drop(matching_features, axis=1, inplace=True)

#selected_features_df.drop(matching_features, axis=1, inplace=True)

# 將合併的特徵 DataFrame 匯出到新的 Excel 文件
merged_features_df.to_excel('Merged_Features.xlsx', index=False)
selected_features_df.to_excel('selected_Features.xlsx', index=False)
