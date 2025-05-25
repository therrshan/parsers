from datasetloader.mnist import MNISTLoader
from datasetloader.fashion_mnist import FashionMNISTLoader
from datasetloader.newsgroups import TwentyNewsLoader

# MNIST
mnist = MNISTLoader(normalize=True, as_dataframe=True)
X, y = mnist.load("train")
print("MNIST:", X.shape, y.shape)

# Fashion MNIST
fashion = FashionMNISTLoader(normalize=True, as_dataframe=False)
Xf, yf = fashion.load("test")
print("Fashion MNIST:", Xf.shape, yf.shape)

# 20NG
news = TwentyNewsLoader(as_dataframe=True)
Xn, yn = news.load("train")
print("20NG:", Xn.shape, yn.shape)
