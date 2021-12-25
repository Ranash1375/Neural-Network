/**
 * @file main.cpp
 * @author Rana Shariat (rana.shariat@gmail.com)
 * @brief Neural network (NN) implementation using back propagation algorithm.
 * @version 0.1
 * @date 2021-12-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <stdexcept>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include "edge.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "network.hpp"
#include "read_x.hpp"
#include "read_y.hpp"
#include "configuration.hpp"

using namespace std;

/**
 * @brief Find an element in a vector and return the index number.
 * 
 * @tparam T Template.
 * @param a The vector in which the element is being searched.
 * @param b The element we want to find.
 * @return uint64_t The index number of b in a.
 */
template <typename T>
uint64_t find_in_vec(const vector<T> a, const T b)
{
     for (uint64_t i = 0; i < a.size(); i++)
     {
          if (b == a[i])
               return i;
     }
     return 0; // Just written for removing the warning. This line will never be executed in this problem.
}

/**
 * @brief Calculating the accuracy by comparing the predicted and actual values.
 * 
 * @param a The first vector (actual or predicted values)
 * @param b The second vector (predicted or actual values)
 * @return double The average number of elements which have the same value in both vectors.
 */
double accuracy(vector<uint64_t> a, vector<uint64_t> b)
{
     uint64_t true_prediction = 0;
     for (uint64_t i = 0; i < a.size(); i++)
     {
          if (a[i] == b[i])
               true_prediction++;
     }
     return (double)(true_prediction) / (double)a.size();
}

/**
 * @brief printing the elements of a vector.
 * 
 * @tparam T Template.
 * @param v Vector to be printed.
 */
template <typename T>
void print_elements(const vector<T> &v)
{
     for (const T &i : v)
          cout << i << ' ';
     cout << '\n';
}

/**
 * @brief Average of a vector's elements.
 * 
 * @tparam T Template.
 * @param v A vector.
 * @return T Average of elements.
 */
template <typename T>
T vec_average(const vector<T> &v)
{
     T sum = 0;
     for (const T &i : v)
          sum = sum + i;
     return sum / (double)v.size();
}

int main()
{
     try
     {
          // Reading file x.csv which contains features dataset.
          string filename = "x.csv";
          read_x x(filename);

          // Reading file y.csv which contains output (classes) dataset.
          filename = "y.csv";
          read_y classes(filename);

          // To check if both x.csv and y.csv have the same number of instances
          if (x.get_rows() != classes.get_rows())
          {
               cout << "Error in" << filename << ": Number of rows is not the same as features dataset file!";
               return -1;
          }

          uint64_t number_instances = classes.get_rows();          // Number of instances in the dataset.
          uint64_t number_features = x.get_cols();                 // Number of features in x.csv.
          uint64_t number_classes = classes.find_number_classes(); // Number of different classes in the dataset.

          // Creating output vector proper for neural vector. Instead of class number, the output should be a vector with size equal to number of classes. The ith element, where i is the class number, should be 1 and the rest 0.
          vector<vector<double>> y(number_instances, vector<double>(number_classes, 0));

          for (uint64_t i = 0; i < number_instances; i++)
          {
               y[i][classes.get_values()[i] - 1] = 1;
          }

          // Reading layers.csv which contains number of neurons in each layer (except the input and output layer)
          filename = "layers.csv";
          read_y number_neurons(filename);

          // Reading parameters from parameters.csv which contains the paramters of the model.
          filename = "parameters.csv";
          configuration parameters(filename);

          // Adding the number of features (number of neurons for the first layer), the number of classes (the number of neurons for the last layer), and the number of neurons of the hidden layers to the vector number_neurons_layer which shows the number of neurons per layer of the network.
          vector<uint64_t> number_neurons_layer = number_neurons.get_values();
          number_neurons_layer.insert(number_neurons_layer.begin(), number_features);
          number_neurons_layer.insert(number_neurons_layer.end(), number_classes);

          // Number of layers in the defined architecture.
          uint64_t number_layers = number_neurons_layer.size();

          // A vector for saving the accuracy of each trained model using NN.
          vector<double> cv_accuracy(parameters.get_num_cv());

          // Cross validation with num_cv iterations.
          for (uint64_t count = 0; count < parameters.get_num_cv(); count++)
          {
               // Creating train and test sets by splitting data randomly based on the train percentage.
               random_device rd;
               mt19937 mt(rd());
               vector<double> random_numbers(number_instances);                                                              // Creating random number for splitting data.
               vector<double> random_numbers_sorted(number_instances);                                                       // A vector of the sorted random numbers.
               vector<vector<double>> train_x(number_instances * parameters.get_train_percantage() / 100);                   // Train set of x (features).
               vector<vector<double>> test_x(number_instances - number_instances * parameters.get_train_percantage() / 100); // Test set of x (features).
               vector<vector<double>> train_y(number_instances * parameters.get_train_percantage() / 100);                   // Train set of y (outputs).
               vector<vector<double>> test_y(number_instances - number_instances * parameters.get_train_percantage() / 100); //Test set of y (outputs).
               vector<uint64_t> train_classes(number_instances * parameters.get_train_percantage() / 100);                   // Classes of the train set.
               vector<uint64_t> test_classes(number_instances - number_instances * parameters.get_train_percantage() / 100); // Classes of the test set.

               // Creating random numbers.
               for (uint64_t i = 0; i < number_instances; i++)
               {
                    uniform_real_distribution<double> urd(0, 1);
                    random_numbers[i] = urd(mt);
               }

               for (uint64_t i = 0; i < number_instances; i++)
               {
                    random_numbers_sorted[i] = random_numbers[i];
               }

               // Sorting random numbers and choosing the first train_percentage as train set.
               sort(random_numbers_sorted.begin(), random_numbers_sorted.end());

               // Train set.
               uint64_t j = 0;
               for (uint64_t i = 0; i < number_instances * parameters.get_train_percantage() / 100; i++)
               {
                    train_x[j] = x.get_values()[find_in_vec(random_numbers, random_numbers_sorted[i])];
                    train_y[j] = y[find_in_vec(random_numbers, random_numbers_sorted[i])];
                    train_classes[j] = classes.get_values()[find_in_vec(random_numbers, random_numbers_sorted[i])];
                    j++;
               }

               // Test set.
               j = 0;
               for (uint64_t i = number_instances * parameters.get_train_percantage() / 100; i < number_instances; i++)
               {
                    test_x[j] = x.get_values()[find_in_vec(random_numbers, random_numbers_sorted[i])];
                    test_y[j] = y[find_in_vec(random_numbers, random_numbers_sorted[i])];
                    test_classes[j] = classes.get_values()[find_in_vec(random_numbers, random_numbers_sorted[i])];
                    j++;
               }

               uint64_t ID = 0;        // Counter for edge or neuron ID.
               vector<layer> layers;   // Vector which includes the layers of NN.
               vector<neuron> neurons; // Vector which includes the neurons of NN.

               // Generating layers and their neurons.
               for (uint64_t i = 1; i <= number_neurons_layer.size(); i++)
               {
                    layer l(i);
                    l.gen_layer_neurons(number_neurons_layer, neurons, ID);
                    layers.push_back(l);
               }

               ID = 0;
               vector<edge> edges; //vector of all edges of NN.

               // Generating input edges for each neuron.
               for (neuron &i : neurons)
               {
                    i.gen_input_edges(number_neurons_layer, edges, ID);
               }

               // Generating output edges for each neuron.
               for (neuron &i : neurons)
               {
                    i.gen_output_edges(number_neurons_layer, edges);
               }

               // Generating the network.
               network N(neurons.size(), edges.size());

               // Training the network using train set for num_iteration iterations.
               for (uint64_t k = 0; k < parameters.get_num_iteration(); k++)
               {
                    // Setting delta equal to zero at the beginning of each iteration
                    for (edge &i : edges)
                         i.set_delta_zero();

                    // Train using instance number t.
                    for (uint64_t t = 0; t < number_instances * parameters.get_train_percantage() / 100; t++)
                    {

                         // Activate layers of the network.
                         for (layer &i : layers)
                         {
                              i.activate_layer(neurons, edges, train_x[t]);
                         }
                         // Find the error for layers of NN.
                         for (uint64_t i = number_layers; i > 1; i--)
                         {
                              layers[i - 1].error_layer(neurons, edges, train_y[t], number_layers);
                         }

                         // Update delta for each edge.
                         N.delta_update(neurons, edges);
                    }
                    // Update gradient of each edge.
                    N.gradient_update(edges, number_instances * parameters.get_train_percantage() / 100, parameters.get_lambda());

                    // Run gradient descent algorithm to update the weights of the edges.
                    N.gradient_descent(edges, parameters.get_learning_rate());
               }

               // Test the trained model on the test set.
               vector<uint64_t> predicted_classes(number_instances - number_instances * parameters.get_train_percantage() / 100); //Vector containing the predicted classes.

               for (uint64_t t = 0; t < number_instances - number_instances * parameters.get_train_percantage() / 100; t++)
               {
                    // Activate layers of the network
                    for (layer &i : layers)
                    {
                         i.activate_layer(neurons, edges, test_x[t]);
                    }

                    // Find category based on the neuron with maximum activation in the last layer.
                    double max_activation = (neuron(0, number_layers, 1).find_neuron(neurons)).get_activation();
                    uint64_t category = 1;
                    // Update the category of the instance.
                    for (uint64_t i = 2; i <= number_classes; i++)
                    {
                         if (((neuron(0, number_layers, i).find_neuron(neurons)).get_activation()) > max_activation)
                         {
                              max_activation = ((neuron(0, number_layers, i).find_neuron(neurons)).get_activation());
                              category = i;
                         }
                    }
                    predicted_classes[t] = category;
               }
               // Calculate the accuracy of the predicted classes for the test set.
               cv_accuracy[count] = accuracy(predicted_classes, test_classes);
               cout << "\nTest set " << count + 1 << "\n\nPrediction accuracy: " << cv_accuracy[count] << "\n";
               cout << "\nPreticted classes for the test set:\n";
               print_elements(predicted_classes);
               cout << "\nActual classes for the test set:\n";
               print_elements(test_classes);
          }
          // Average accuracy of all trained models.
          cout << "\nAverage accuracy: " << vec_average(cv_accuracy);
     }
     catch (const exception &e)
     {
          return -1;
     }
}