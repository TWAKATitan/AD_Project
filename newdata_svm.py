
#newdata testing 
#Accuracy:  65.25
#ANOVA分析
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
import shap  # 导入SHAP库

# 读取Excel文件，将'--'视为缺失值
data = pd.read_excel('newdata.xlsx', na_values='--')

# 提取特徵和目標
X = data.iloc[:, 14:]  # Starting from the 6th column as feature variables
y = data['Group']


# 处理缺失值使用SimpleImputer
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X = imputer.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM分类器并使用最佳参数
best_params = {'C': 10, 'gamma':'scale', 'kernel':'poly'}
classifier = svm.SVC(**best_params)

# 训练模型
classifier.fit(X_train_scaled, y_train)

# 使用最佳参数的模型进行预测
y_pred = classifier.predict(X_test_scaled)

# 计算准确率和F1分数
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy * 100))
print('F1 Score: ', "%.2f" % (f1 * 100))

