import pprint
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from signal_dataset import SIGNAL_DATASET

def confusion_matrix_visualization(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def classify(algorithm, features, labels, detail=True, save=True, scoring='accuracy'):
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
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring=scoring)
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
            confusion_matrix_visualization(y_test, y_pred)
            print(classification_report(y_test, y_pred, zero_division=0))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return {'model': best_knn,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision']
                }
    elif algorithm == 'svm':
        svm = SVC()
        # param_grid = {
        #     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #     'gamma': ['scale', 'auto']
        # }
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]
        }
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=kf, scoring=scoring)
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
            confusion_matrix_visualization(y_test, y_pred)
            print(classification_report(y_test, y_pred, zero_division=0))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return {'model': best_svm,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision']
                }
    elif algorithm == 'rf':
        rf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],  # 树的数量
            'max_depth': [None, 10, 20],  # 树的最大深度
            'min_samples_split': [2, 5, 10]  # 分割内部节点所需的最小样本数
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring=scoring)
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
            confusion_matrix_visualization(y_test, y_pred)
            print(classification_report(y_test, y_pred, zero_division=0))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return {'model': best_rf,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision']
                }
    elif algorithm == 'lr':
        lr = LogisticRegression(multi_class='multinomial')
        param_grid = {
            'max_iter': [3000],
            'C': [0.1, 1, 10, 100],  # 正则化强度的倒数
            'solver': ['lbfgs']  # 优化算法
        }
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=kf, scoring=scoring)
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
            confusion_matrix_visualization(y_test, y_pred)
            print(classification_report(y_test, y_pred, zero_division=0))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return {'model': best_lr,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision']
                }


if __name__ == "__main__":
    # chosen_labels = ['board1', 'board2', 'board3']
    # chosen_labels = ['board2', 'board3']
    # chosen_labels = ['carroad1', 'carroad2', 'carroad3']
    # chosen_labels = ['flat1', 'flat2', 'flat3', 'flat4', 'flat5', 'flat6']
    # chosen_labels = ['flat5', 'flat6']
    # chosen_labels = ['flat1', 'flat2', 'flat3', 'flat4', 'flat56']
    # chosen_labels = ['flat1', 'flat4']
    # chosen_labels = ['flat14', 'flat2', 'flat3', 'flat56']
    # chosen_labels = ['flat14', 'flat2', 'flat3']
    # chosen_labels = ['brick1-', 'brick2', 'brick3', 'brick4', 'brick5', 'brick6', 'brick7']
    # chosen_labels = ['brick2', 'brick7']
    # chosen_labels = ['brick1-', 'brick2', 'brick3', 'brick4', 'brick5', 'brick6']
    # chosen_labels = ['brick1-', 'brick23', 'brick4', 'brick5', 'brick6']
    # chosen_labels = ['brick1-', 'brick2']
    # chosen_labels = ['brick2', 'brick3']
    # chosen_labels = ['brick12', 'brick3', 'brick4', 'brick5', 'brick6']

    chosen_labels = ['stop', 'grass', 'dirt', 'floor-', 'playground', 'rideroad', 'runway']
    chosen_labels += ['carroad3']
    chosen_labels += ['board2', 'board3']
    chosen_labels += ['flat14', 'flat2', 'flat3']
    chosen_labels += ['brick12', 'brick3', 'brick4', 'brick5', 'brick6']
    # chosen_labels += ['flat56']

    # chosen_labels = ['rideroad', 'flat2']
    # chosen_labels = ['playground', 'board2']

    dataset = SIGNAL_DATASET(chosen_labels)
    algorithms = ['knn', 'svm', 'rf', 'lr']
    results = {algorithm: {} for algorithm in algorithms}
    # results['knn'] = classify('knn', features=dataset.features_scaled_pca, labels=dataset.labels)
    results['svm'] = classify('svm', features=dataset.features_scaled, labels=dataset.labels)
    # results['rf'] = classify('rf', features=dataset.features_scaled, labels=dataset.labels)
    # results['lr'] = classify('lr', features=dataset.features_scaled, labels=dataset.labels)
    pprint.pprint(results['svm'])
