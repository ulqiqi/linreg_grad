#!/home/lain/linreg/.data/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, 0].tolist()
    Y = df.iloc[:, 1].tolist()
    return X, Y

def train_model(X, Y):
    X_max = max(X)
    Y_max = max(Y)
    
    # normalisation entre 0 et 1
    X_sc = [x / X_max for x in X]
    Y_sc = [y / Y_max for y in Y]
    
    m_sc = 0
    b_sc = 0
    lr = 0.1 # haut learning rate car les données sont scaled entre 0 et 1
    epochs = 10000
    n = len(X_sc)
    
    for _ in range(epochs):
        y_pred = [m_sc * x + b_sc for x in X_sc]
        m_grad = -(2/n) * sum([x * (y - y_p) for x, y, y_p in zip(X_sc, Y_sc, y_pred)])
        b_grad = -(2/n) * sum([y - y_p for y, y_p in zip(Y_sc, y_pred)])
        m_sc -= lr * m_grad
        b_sc -= lr * b_grad
        
    # remise à l'échelle
    m = m_sc * Y_max / X_max
    b = b_sc * Y_max
    
    return m, b

def plot_results(X, Y, m, b):
    plt.scatter(X, Y)
    y_line = [m * x + b for x in X]
    plt.plot(X, y_line, color='red')
    formula = f"y = {m:.2f}x + {b:.2f}"
    plt.text(0.05, 0.95, formula, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.savefig("regression_plot.png")
    print(f"Graph saved as regression_plot.png. Formula: {formula}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./linear_regression.py <csv_file>")
        sys.exit(1)
    X, Y = load_data(sys.argv[1])
    m, b = train_model(X, Y)
    plot_results(X, Y, m, b)
