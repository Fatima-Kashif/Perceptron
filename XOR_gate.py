import numpy as np
class NeuralNetwork:
    def __init__ (self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

        # Initialize biases
        self.bias_hidden = np.random.rand(self.hidden_size)
        self.bias_output = np.random.rand(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # Backward pass
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            # Monitor the training loss
            loss = np.mean(np.square(y - output))  # Mean Squared Error
            if loss < 0.01:  # Stop training if error is below a threshold
                break

    def predict(self, X):
        return self.forward(X)

def main():
    # Take user input for XOR gate in a horizontal form
    print("Training a Neural Network for XOR gate.")
    
    # Input XOR gate data in a single line
    print("\nEnter the inputs for XOR table (8 values, space-separated, format: x1 y1 x2 y2 ...):")
    input_data = list(map(int, input().split()))
    
    # Reshape input to a 2D array (4 rows, 2 columns)
    X = np.array(input_data).reshape(4, 2)

    print("\nEnter the target outputs for XOR gate (4 values, space-separated):")
    y = np.array(list(map(int, input().split()))).reshape(-1, 1)

    # Create a Neural Network instance and train it
    input_size = X.shape[1]
    hidden_size = 2  # Example: 2 neurons in hidden layer
    output_size = y.shape[1]
    learning_rate = 0.1

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    nn.train(X, y, epochs=10000)

    print("\nTraining complete.")
    
    # Testing the model
    print("\nTesting the model with XOR inputs:")
    predictions = nn.predict(X)
    rounded_predictions = np.round(predictions)

    # Display results clearly
    print("\nTest Results:")
    for i, prediction in enumerate(rounded_predictions):
        print(f"Input: {X[i]} => Predicted Output: {int(prediction[0])} (Expected: {y[i][0]})")

if __name__ == "__main__":
    main()
