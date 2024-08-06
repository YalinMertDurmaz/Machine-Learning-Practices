# LDA vs PCA
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 

iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

colors=["red", "blue", "green"]
plt.figure()

for color, i ,target_names in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i,0], X_pca[y == i,1], color = color, alpha=0.8, label= target_names)
plt.legend()
plt.title("PCA of Iris Dataset")

plt.figure()
for color, i ,target_names in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i,0], X_lda[y == i,1], color = color, alpha=0.8, label= target_names)
plt.legend()
plt.title("LDA of Iris Dataset")






































