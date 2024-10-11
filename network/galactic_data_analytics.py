import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope

class GalacticDataAnalytics:
    def __init__(self, data):
        self.data = data

    def data_summary(self):
        return self.data.describe()

    def data_visualization(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x="galaxy_type", data=self.data)
        plt.title("Galaxy Type Distribution")
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="distance", y="velocity", data=self.data)
        plt.title("Distance vs Velocity")
        plt.show()

    def feature_engineering(self):
        self.data["log_distance"] = np.log(self.data["distance"])
        self.data["log_velocity"] = np.log(self.data["velocity"])

    def feature_selection(self):
        X = self.data.drop(["galaxy_type", "distance", "velocity"], axis=1)
        y = self.data["galaxy_type"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def linear_regression(self, X_train, X_test, y_train, y_test):
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def pca_analysis(self, X):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        return X_pca

    def tsne_analysis(self, X):
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)
        return X_tsne

    def kmeans_clustering(self, X):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        labels = kmeans.labels_
        return labels

    def anomaly_detection(self, X):
        ee = EllipticEnvelope(contamination=0.1)
        ee.fit(X)
        anomalies = ee.predict(X)
        return anomalies

def main():
    data = pd.read_csv("galactic_data.csv")
    gda = GalacticDataAnalytics(data)

    print("Data Summary:")
    print(gda.data_summary())

    gda.data_visualization()

    gda.feature_engineering()

    X_train, X_test, y_train, y_test = gda.feature_selection()

    mse, r2 = gda.linear_regression(X_train, X_test, y_train, y_test)
    print("Linear Regression:")
    print("MSE:", mse)
    print("R2:", r2)

    X_pca = gda.pca_analysis(X_train)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train)
    plt.title("PCA Analysis")
    plt.show()

    X_tsne = gda.tsne_analysis(X_train)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_train)
    plt.title("t-SNE Analysis")
    plt.show()

    labels = gda.kmeans_clustering(X_train)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=labels)
    plt.title("K-Means Clustering")
    plt.show()

    anomalies = gda.anomaly_detection(X_train)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=anomalies)
    plt.title("Anomaly Detection")
    plt.show()

if __name__ == "__main__":
    main()
