import numpy as np

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 10000
lr = 0.1
inputLN, hiddenLN, outputLN = 2,2,1

hidden_weights = np.random.uniform(size=(inputLN, hiddenLN))
hidden_bias = np.random.uniform(size=(1,hiddenLN))
output_weights = np.random.uniform(size=(hiddenLN, outputLN))
output_bias = np.random.uniform(size=(1,outputLN))

print("Initial hidden weights: ", hidden_weights)
print("Initial hidden biases: ", hidden_bias)
print("Initial ouput weights: ", output_weights)
print("Initial output bias: ", output_bias)

for i in range(epochs):
	hidden_activation = np.dot(inputs, hidden_weights)
	hidden_activation += hidden_bias
	hidden_output = sigmoid(hidden_activation)

	output_activation = np.dot(hidden_output,output_weights)
	output_activation += output_bias
	predicted_output = sigmoid(output_activation)

	error = expected_output - predicted_output
	d_predicted_output = error * sigmoid_derivative(predicted_output)

	error_hidden_layer = d_predicted_output.dot(output_weights.T)
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)
	 
	output_weights += hidden_output.T.dot(d_predicted_output) * lr
	output_bias += np.sum(d_predicted_output,axis=0, keepdims=True) * lr
	hidden_weights += inputs.T.dot(d_hidden_layer) * lr
	hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


print("Final hidden weight : ", hidden_weights)
print("Final hidden bias : ", hidden_bias)
print("Final output weights : ", output_weights)
print("Final output bias : ", output_bias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predicted_output)

