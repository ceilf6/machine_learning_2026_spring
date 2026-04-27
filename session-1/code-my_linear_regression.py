import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights  = self.weights - self.lr * dw
            self.bias  = self.bias -  self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


if __name__ == "__main__":

    # =========================
    # Load dataset
    # =========================

    dataset = pd.read_csv("Salary_Data.csv")

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values


    # =========================
    # Train/Test split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, random_state=0
    )


    # =========================
    # Train model
    # =========================

    model = MyOwnLinearRegression()
    model.fit(X_train, y_train)


    # =========================
    # Visualization
    # =========================

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # ====================================
    # LEFT : Training set
    # ====================================

    ax = axes[0]

    ax.scatter(X_train, y_train, color="red")

    # regression line
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1,1)
    y_line = model.predict(X_line)

    ax.plot(X_line, y_line, color="blue")

    ax.set_title("Salary vs Experience (Training set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")


    # ====================================
    # RIGHT : Test set
    # ====================================

    ax = axes[1]

    ax.scatter(X_test, y_test, color="red")

    # same regression line
    ax.plot(X_line, y_line, color="blue")

    ax.set_title("Salary vs Experience (Test set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")


    plt.tight_layout()
    plt.show()
    
