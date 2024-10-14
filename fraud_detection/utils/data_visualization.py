# data_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, feature):
    """
    Plot a histogram of the specified feature.

    Parameters:
    - data: Data to be plotted.
    - feature: Feature to be plotted.
    """
    plt.hist(data[feature], bins=50)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_scatter(data, feature1, feature2):
    """
    Plot a scatter plot of the specified features.

    Parameters:
    - data: Data to be plotted.
    - feature1: First feature to be plotted.
    - feature2: Second feature to be plotted.
    """
    plt.scatter(data[feature1], data[feature2])
    plt.title(f'Scatter Plot of {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

def plot_heatmap(data):
    """
    Plot a heatmap of the correlation matrix.

    Parameters:
    - data: Data to be plotted.
    """
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
