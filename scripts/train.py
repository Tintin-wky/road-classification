import pprint
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from signal_dataset import SIGNAL_DATASET

scaler = StandardScaler()
def preprocess(data, scale=True, pca=False, save=False):
    if scale is True:
        scaler.fit(data)
        if save:
            joblib.dump(scaler, '../models/scaler.joblib')
        data_scaled = scaler.transform(data)
        if pca is True:
            pca = PCA(n_components=0.99)
            data_pca = pca.fit_transform(data_scaled)
            return data_pca
        else:
            return data_scaled


def classify(algorithm, features, labels, detail=False, save=False):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=31)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    if detail:
        print(f'Algorithm: {algorithm}')
    if algorithm == 'knn':
        knn = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': range(3, 16),
            'weights': ['distance', None]
        }
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        if detail:
            print("Best parameters found: ", best_params)
            print("Best score: ", best_score)
        best_knn = KNeighborsClassifier(**best_params)
        best_knn.fit(x_train, y_train)
        if save:
            joblib.dump(best_knn, '../models/knn.joblib')
        y_pred = best_knn.predict(x_test)
        if detail:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, zero_division=0))
        return {'model':best_knn, 'precision':classification_report(y_test, y_pred, output_dict=True, zero_division=0)['weighted avg']['precision']}
    elif algorithm == 'svm':
        svm = SVC()
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        if detail:
            print("Best parameters found: ", best_params)
            print("Best score: ", best_score)
        best_svm = SVC(**best_params)
        best_svm.fit(x_train, y_train)
        if save:
            joblib.dump(best_svm, '../models/svm.joblib')
        y_pred = best_svm.predict(x_test)
        if detail:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, zero_division=0))
        return {'model':best_svm, 'precision':classification_report(y_test, y_pred, output_dict=True, zero_division=0)['weighted avg']['precision']}
    elif algorithm == 'rf':
        rf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],  # 树的数量
            'max_depth': [None, 10, 20],  # 树的最大深度
            'min_samples_split': [2, 5, 10]  # 分割内部节点所需的最小样本数
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        if detail:
            print("Best parameters found: ", best_params)
            print("Best score: ", best_score)
        best_rf = RandomForestClassifier(**best_params, random_state=42)
        best_rf.fit(x_train, y_train)
        if save:
            joblib.dump(rf, '../models/rf.joblib')
        y_pred = best_rf.predict(x_test)
        if detail:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, zero_division=0))
        return {'model':best_rf, 'precision':classification_report(y_test, y_pred, output_dict=True, zero_division=0)['weighted avg']['precision']}
    elif algorithm == 'lr':
        lr = LogisticRegression(multi_class='multinomial')
        param_grid = {
            'max_iter': [1000],
            'C': [0.1, 1, 10],  # 正则化强度的倒数
            'solver': ['lbfgs']  # 优化算法
        }
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        if detail:
            print("Best parameters found: ", best_params)
            print("Best score: ", best_score)
        best_lr = LogisticRegression(**best_params)
        best_lr.fit(x_train, y_train)
        if save:
            joblib.dump(best_lr, '../models/lr.joblib')
        y_pred = best_lr.predict(x_test)
        if detail:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, zero_division=0))
        return {'model':best_lr, 'precision':classification_report(y_test, y_pred, output_dict=True, zero_division=0)['weighted avg']['precision']}


if __name__ == "__main__":
    dataset_dir = '../dataset'
    chosen_labels = ['Stop', 'BrickRoad1', 'BrickRoad2', 'BrickRoad3', 'CarRoad']
    dataset = SIGNAL_DATASET(chosen_labels=chosen_labels, dataset_dir=dataset_dir)
    algorithms = ['knn', 'svm', 'rf', 'lr']
    results = {algorithm:{} for algorithm in algorithms}
    results['knn'] = classify('knn', features=preprocess(dataset.features, pca=True), labels=dataset.labels)
    results['svm'] = classify('svm', features=preprocess(dataset.features), labels=dataset.labels)
    results['rf'] = classify('rf', features=preprocess(dataset.features), labels=dataset.labels)
    results['lr'] = classify('lr', features=preprocess(dataset.features), labels=dataset.labels)
    pprint.pprint(results)
