# Introduction
In this project, the feedforward propagation and backpropagation algorithms for the neural network are implemented by C++. Then, [Wine recognition dataset](https://archive.ics.uci.edu/ml/datasets/wine) is used for training and prediction. The performance of the neural network is computed by accuracy of the prediction for the test set.
# Neural Network
A neural network can have different architectures. Each network is constructed from at least three layers. The first and last layers are called input and output layers, respectively. Other layers are called hidden layers. Each of the layers has some neurons. The number of neurons in the first layer equals the number of features in the dataset. The number of neurons in the last layer equals the number of categories in the dataset. Also, a bias unit neuron must be added to all the layers except the last one. Each neuron of each layer is connected to all the neurons of the next layer by edges. 
# Methods
First, the variables of the problem will be explained. Later, the algorithm will be discussed.
The variables are as follows.
- $x$: Input of the neural network which are the features of the dataset.
- $y$: output of the dataset as a vector in which all the elements except the $i^{th}$ element where $i$ is the category of this instance are 0 and the $i^{th}$ element equals $1$.
- $s_l$: number of neurons in layer $l = 1, ..., L$ excluding the bias unit.
- $a^l$: activation of layer $l$ which is a vector of size $s_l$ and each element is the activation for each neuron.
- $\delta^l$: error of layer $l$ which is a vector of size $s_l$ and each element is the error for each neuron.
- $\theta^l$: matrix of weights of the edges connecting layer $l$ to $l+1$ with size $s_{l+1}\times(s_l+1)$ .
- $\Delta^l$: matrix of delta of the edges connecting layer $l$ to $l+1$ with size $s_{l+1}\times(s_l+1)$.
- $D^l$: matrix of gradient of the edges connecting layer $l$ to $l+1$ with size $s_{l+1}\times(s_l+1)$.
- $\lambda$: Regularization parameter of the algorithm which is used for regularizing he objective function.
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

# Implementation
For training, the model $T\%$ of the dataset is chosen randomly. The other $1-T\%$ is used as the test set. Also, since the training may depend on the train set, we repeat the training and test multiple times ($M$), called cross-validation (CV). Then, the average accuracy of the test sets is considered the model's performance.
The parameters of the model are given to the code as input files. Each of the files is described in the following sections.
## Input files 
The implemented algorithm is tested using the [Wine recognition dataset](https://archive.ics.uci.edu/ml/datasets/wine). The outliers are removed from the dataset, and each feature is scaled. The inputs of the algorithm are as follows.
### x.csv
The file x.csv includes the features of the dataset. Each row is an instance, and each column shows a feature.
For example, the following numbers could be the first three instances of a x.csv file which has $13$ features. So, the number of numbers on each line shows the number of features, and the number of lines show the number of instances.

///////////////////////////////////////////////////////////////////////////////////////////
$1.51,-0.57,0.27,-1.24,2.29,0.82,1.05,-0.64,1.46,0.28,0.38,1.82,0.97$
$0.20,0.51,-0.92,2.72,0.11,0.58,0.74,0.81,-0.50,-0.30,0.42,1.09,0.93$
$0.15,0.03,1.26,-0.23,0.19,0.82,1.23,-0.48,2.47,0.29,0.33,0.77,1.35$
///////////////////////////////////////////////////////////////////////////////////////////

### y.csv
The file y.csv includes the outputs of the dataset. Each row shows the category of the instance. The following example shows the classes for the first three instances. The first two instances have category $1$, and the third one has category $2$. So, each line is related to one instance, and the file should have just one column.

///////////////////////////////////////////////////////////////////////////////////////////
$1$

$1$

$2$
///////////////////////////////////////////////////////////////////////////////////////////
### layers.csv
The file layers.csv contains the number of neurons for each hidden layer. The numbers should be written in the file in one column. So, the number on row $i$ shows the number of neurons in hidden layer $i$. The following example shows a network with two hidden layers, each with five neurons.
///////////////////////////////////////////////////////////////////////////////////////////
$5$

$5$
///////////////////////////////////////////////////////////////////////////////////////////
### parameters.csv
This file contains five lines. The first line is the number of iterations ($N$) used for training the model. The second line is the number of iterations for cross-validation ($M$). The third line is the percentage of data for the train set ($T$). The fourth line is the learning rate ($\alpha$). Finally, the last line is the regularization ($\lambda$). The following shows an example of the parameters.csv file. The first three lines should be integers since the first two are the number of iterations for training and the number of iterations for CV. The third one should also be an integer number less than $100$ since it shows a percentage. The last two lines could be any real numbers.
///////////////////////////////////////////////////////////////////////////////////////////
$300$

$10$

$80$

$0.06$

$0.01$
///////////////////////////////////////////////////////////////////////////////////////////
## Outputs of the algorithm
In this section, the algorithm results for two different setups of the network on the Wine recognition dataset will be presented. Remember that the algorithm parameters must be tuned to get better results.
### Test $1$
A network with the following properties is generated.
1. The network has one hidden layer with $5$ neurons.
2. The number of iterations for the training ($N$) is 100.
3. The number of iterations for cross-validation ($M$) is 10.
4. The percentage of data for the train set ($T$) is 80.
5. The learning rate is 0.06. 
6. The regularization is 0.01. 

The average accuracy for the train sets is about $82\%$ for this case. However, since there is some randomness in initializing the weights of the network edges and choosing the train and test set, the accuracy could be different by a small amount each time the code is executed.

### Test $2$
We increased the number of hidden layers for this test, and it is expected to increase the accuracy. However, $N$ should also be increased as the size of the network increases. This results in a much slower execution time. After tuning, a network with the following properties is generated.
1. The network has two hidden layers with $5$ neurons in each layer.
2. The number of iterations for the training ($N$) is 400.
3. The number of iterations for cross-validation ($M$) is 10.
4. The percentage of data for the train set ($T$) is 80.
5. The learning rate is 0.06. 
6. The regularization is 0.1. 

The average accuracy for the train sets is about $87\%$ for this case. The accuracy average has increased compared to test $1$, but it may not be efficient due to the high execution time. 