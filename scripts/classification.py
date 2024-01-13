import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scripts.signal_dataset import SIGNAL_DATASET

def preprocess(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=0.99)
    data_pca = pca.fit_transform(data_scaled)
    return data_pca

def classify(algorithm, detail=True, train_state=42, evaluate_state=31):
    if algorithm == 'knn':
        kf = KFold(n_splits=10, shuffle=True, random_state=train_state)
        best_k = 1
        best_score = 0
        for k in range(1, 16):
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, features, labels, cv=kf, scoring='accuracy')
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_k = k
                best_score = avg_score
            if detail:
                print(f'k={k}, Average Accuracy: {avg_score:.4f}')
        if detail:
            print(f'Best k: {best_k} with Average Accuracy: {best_score:.4f}')

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=evaluate_state)
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))
    elif algorithm == 'wknn':
        kf = KFold(n_splits=10, shuffle=True, random_state=train_state)
        best_k = 1
        best_score = 0
        for k in range(1, 16):
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            scores = cross_val_score(knn, features, labels, cv=kf, scoring='accuracy')
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_k = k
                best_score = avg_score
            if detail:
                print(f'k={k}, Average Accuracy: {avg_score:.4f}')
        if detail:
            print(f'Best k: {best_k} with Average Accuracy: {best_score:.4f}')

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=evaluate_state)
        knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))
    elif algorithm == 'svm':
        kf = KFold(n_splits=10, shuffle=True, random_state=train_state)
        best_C = 1.0
        best_score = 0
        best_gamma = 'scale'
        for C in np.logspace(-3, 3, 7):  # 对C值进行对数尺度的搜索
            for gamma in ['scale', 'auto']:
                svm = SVC(C=C, gamma=gamma,kernel='rbf')
                scores = cross_val_score(svm, features, labels, cv=kf, scoring='accuracy')
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_C = C
                    best_score = avg_score
                    best_gamma = gamma
                if detail:
                    print(f'C={C}, gamma={gamma}, Average Accuracy: {avg_score:.4f}')
        if detail:
            print(f'Best C: {best_C}, Best gamma: {best_gamma} with Average Accuracy: {best_score:.4f}')

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=evaluate_state)
        best_svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
        best_svm.fit(x_train, y_train)
        y_pred = best_svm.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    dataset_dir = '../dataset'
    chosen_labels = ['Stop', 'BrickRoad1', 'BrickRoad2', 'BrickRoad3', 'CarRoad']
    dataset = SIGNAL_DATASET(chosen_labels=chosen_labels, dataset_dir=dataset_dir)
    features = preprocess(dataset.features)
    labels = dataset.labels
    algorithm = ['knn','wknn','svm']
    for algorithm in algorithm:
        print(f'Algorithm: {algorithm}')
        classify(algorithm,detail=False)