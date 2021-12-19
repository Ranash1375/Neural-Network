# Introduction
In this project, the feedforward propagation and backpropagation algorithms for the neural network are implemented by C++. Then, [Wine recognition dataset](https://archive.ics.uci.edu/ml/datasets/wine) is used for training and prediction. The performance of the neural network is computed by accuracy.
# Neural Network
A neural network can have different architectures. Each network is constructed from at least three layers. The first and last layers are called input and output layers, respectively. Other layers are called hidden layers. Each of the layers has some neurons. The number of neurons in the first layer equals the number of features in the dataset. The number of neurons in the last layer equals the number of categories in the dataset. Also, a bias unit neuron must be added to all the layers except the last one. Each neuron of each layer is connected to all the neurons of the next layer. 
# Methods
First, the variables of the problem will be explained. Later, the algorithm will be discussed.
The variables are as follows.
- $x$: Input of the neural network which are the features of the dataset.
- $y$: output of the dataset as a vector in which all the elements except the $i^{th}$ element where $i$ is the category of this instance are 0.
- $s_l$: number of neurons in layer $l = 1, ..., L$ excluding the bias unit.
- $a^l$: activation of layer l which is a vector of size $s_l$ and each element is the activation for each neuron.
- $\delta^l$: error of layer l which is a vector of size $s_l$ and each element is the error for each neuron.
- $\theta^l$: matrix of weights of the edges connecting layer $l$ to $l+1$ with size $s_{l+1}*(s_l+1)$ .
- $\Delta^l$: matrix of delta of the edges connecting layer $l$ to $l+1$ with size $s_{l+1}*(s_l+1)$.
- $D^l$: matrix of gradient of the edges connecting layer $l$ to $l+1$ with size $s_{l+1}*(s_l+1)$.
- $\lambda$: Regularization parameter for the model.
- $\alpha$: Learning rate for the gradient descent algorithm.
- N: Number of iterations of the training algorithm.
## Forward propagation
Forward propagation algorithm, for instance $t$ of the dataset ($x^t$).
1. Set $a^1 = x^t$.
2. For each layer $l = 2:L-1$
    - $a^j = g(\theta^{j-1}a^{j-1})$ where $g(z)=1/(1+e^{-z})$.
    - Add 1 to the beginning of vector $a^j$.
3. Output layer activation $a^L$ equals $g(\theta^{j-1}a^{j-1})$.

This algorithm is used for both training and prediction. When used for prediction, the number of neuron with the highest activation value is chosen as the category prediction for the instance.
## Backward propagation
The training algorithm is as follows.
1. Add $x_0 = 1$ to all rows of the dataset.
2. Randomly initialize $\theta$ for all layers using $[-\epsilon_{init},\epsilon_{init}]$ interval, where $\epsilon_{init} = sqrt(6)/sqrt(s_l+s_{l+1})$.
3. Set $\Delta^l=0$ for all layers.
4. For each instance $t = 1, ..., m$ in the dataset
    - Do forward propagation.
    - $\delta^L = a^L - y^t$.
    - For L = L-1, ..., 2 $\delta^l = (\theta^l)^T\delta^{l+1}.*a^l(1-a(l))$.
    - Remove the first element of $\delta^l$.
    - $\Delta^l := \Delta^l + \delta^{l+1}(a^l)^T$.
5. Compute gradient
    - $D^l = (1/m)\Delta^l$ for $j=0$.
    - $D^l = (1/m)\Delta^l + (\lambda/m) \theta^l$ for $j>0$.
6. Update weights. $\theta^l := \theta^l-\alpha D^l$.
7. Repeat steps 3 to 6 for $N$ times.

# Implementaion
For training, the model $70\%$ of the dataset is chosen randomly. The other $30\%$ is used as the test set. Also, since the training may depend on the train set, we repeat the training and testing multiple times ($M$). Then, the average of the accuracies gained by the test sets is considered the model's performance.
The parameters of the model are given to the code as input files. Each of the files is described in the following sections.
## Input files 
### x.csv
The file x.csv includes the features of the dataset. Each row is an instance, and each column shows a feature. 
### y.csv
The file y.csv includes the outputs of the dataset. Each row shows the category of the instance. 
### layers.csv
The file layers.csv contains the number of neurons for each hidden layer. The numbers should be written in the file in one column. So, the number on row $i$ shows the number of neurons in hidden layer $i$.
### parameters.csv
This file contains four lines. The first line is the number of iterations ($N$) used for training the model. The second line is the number of iterations for cross-validation ($M$). The third line is the learning rate. Finally, the last line is the regularization parameter. 
## Output of the algorithm
The implemented algorithm is tested using the [Wine recognition dataset](https://archive.ics.uci.edu/ml/datasets/wine). The network tested has one hidden layer with 5 neurons. The number of iterations for the training ($N$) is 100, the number of iterations for cross-validation is 10, the learning rate is 0.06, and the regularization is set to 0.01. The average accuracy of the test set is about 0.773.
