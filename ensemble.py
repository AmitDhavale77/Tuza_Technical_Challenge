from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Best parameters from GridSearchCV for each model (replace these with your actual results)
best_knn_params = {'metric': 'euclidean', 'n_neighbors': 4, 'weights': 'distance'}
best_rf_params = {'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 32}
best_svm_params = {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}

file_path = 'updated_transaction_data_withlabels.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

print(data.head())

print(data.columns)

label_column = 'Current pay'  
X = data.drop(columns=[label_column, 'Transaction_per_Unit_Turnover_RobustScaled'])  # Features
y = data[label_column]  # Target labels (already encoded)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], 
                           weights=best_knn_params['weights'], 
                           metric=best_knn_params['metric'])

rf = RandomForestClassifier(max_depth=best_rf_params['max_depth'],
                            max_features=best_rf_params['max_features'],
                            min_samples_leaf=best_rf_params['min_samples_leaf'],
                            min_samples_split=best_rf_params['min_samples_split'],
                            n_estimators=best_rf_params['n_estimators'])

svm = SVC(probability=True, C=best_svm_params['C'], kernel=best_svm_params['kernel'], gamma=best_svm_params['gamma'])

# Create a VotingClassifier with hard voting (majority class)
voting_clf = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm)], voting='soft')

# Stratified K-Folds Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the ensemble model
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')

# Print Cross-validation results
print("Cross-validation scores for the ensemble model: ", cv_scores)
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
print(f"Standard deviation of cross-validation score: {cv_scores.std():.4f}")

# Optionally, fit the ensemble model and evaluate its performance on the test set
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

# Evaluation on test set
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Perform cross-validation on individual models
models = {'KNN': knn, 'Random Forest': rf, 'SVM': svm}
for model_name, model in models.items():
    cv_scores_individual = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{model_name} - Cross-validation mean score: {cv_scores_individual.mean():.4f}")
    print(f"{model_name} - Standard deviation: {cv_scores_individual.std():.4f}")

import matplotlib.pyplot as plt

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()





# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Current pay']), data['Current pay'], test_size=0.2, random_state=42)

# Instantiate the models with the best parameters
knn = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], 
                           weights=best_knn_params['weights'], 
                           metric=best_knn_params['metric'])

rf = RandomForestClassifier(max_depth=best_rf_params['max_depth'],
                            max_features=best_rf_params['max_features'],
                            min_samples_leaf=best_rf_params['min_samples_leaf'],
                            min_samples_split=best_rf_params['min_samples_split'],
                            n_estimators=best_rf_params['n_estimators'])

svm = SVC(C=best_svm_params['C'], kernel=best_svm_params['kernel'], gamma=best_svm_params['gamma'])

# Create a VotingClassifier with hard voting (majority class)
voting_clf = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm)], voting='hard')

# Stratified K-Folds Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the ensemble model
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')

# Print Cross-validation results
print("Cross-validation scores for the ensemble model: ", cv_scores)
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
print(f"Standard deviation of cross-validation score: {cv_scores.std():.4f}")

# Optionally, fit the ensemble model and evaluate its performance on the test set
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

# Evaluation on test set
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Perform cross-validation on individual models
models = {'KNN': knn, 'Random Forest': rf, 'SVM': svm}
for model_name, model in models.items():
    cv_scores_individual = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{model_name} - Cross-validation mean score: {cv_scores_individual.mean():.4f}")
    print(f"{model_name} - Standard deviation: {cv_scores_individual.std():.4f}")