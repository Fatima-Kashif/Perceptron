import numpy as np

class SimplePerceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0

    def activation_function(self, net):
        # Step function for activation
        return 1 if net >= 0 else 0

    def fit(self, X, y, max_epochs=1000):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(max_epochs):
            errors = 0

            # For each training pair
            for i in range(len(X)):
                net = np.dot(X[i], self.weights) + self.bias
                output = self.activation_function(net)

                # Update weights and bias if an error occurred
                if output != y[i]:
                    errors += 1
                    self.weights += self.learning_rate * (y[i] - output) * X[i]
                    self.bias += self.learning_rate * (y[i] - output)

            if errors == 0:
                break

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            net = np.dot(X[i], self.weights) + self.bias
            predictions.append(self.activation_function(net))
        return predictions


# Main program
if __name__ == "__main__":
    print("\nEnter the inputs for AND gate (8 values, space-separated, format: x1 y1 x2 y2 ...):")
    input_data = list(map(int, input().split()))

    # Reshape input to a 2D array (4 rows, 2 columns)
    X = np.array(input_data).reshape(4, 2)

    print("\nEnter the target outputs for AND gate (4 values, space-separated):")
    y = np.array(list(map(int, input().split())))

    if len(y) != 4:
        print("Error: The number of target outputs must match the number of input rows (4).")
        exit()

    perceptron = SimplePerceptron()
    perceptron.fit(X, y)

    predictions = perceptron.predict(X)

    print("\nPredictions for the given data:", predictions)
    print("Expected output:", y.tolist())
