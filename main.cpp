/**
 * @file main.cpp
 * @author Rana Shariat (rana.shariat@gmail.com)
 * @brief Neural network (NN) implementation using back propagation algorithm
 * @version 0.1
 * @date 2021-12-18
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "edge.hpp"
#include "neuron.hpp"

using namespace std;

/**
 * @brief Checks a strign to see if that could be real number.
 * 
 * @param s The string to be converted to a number.
 * @return true If the string s is a real number.
 * @return false If the string s is a real number.
 */

bool is_number(const string &s)
{
     bool has_point = false;            //To check if there is more than 1 point in the string
     uint64_t ascii = s[0];             //To save the ascii number of each character
     if (!isdigit(s[0]) && ascii != 45) //If the first character is not a minus or number
          return false;
     for (uint64_t i = 1; i < s.size(); i++)
     {
          uint64_t ascii = s[i];
          if (!isdigit(s[i]) && ascii != 46) //If the character is not a number or point
               return false;
          if (ascii == 46 && has_point == true) //If there are more than one points in the number
               return false;
          if (ascii == 46) //If the char is a point update has-point
               has_point = true;
     }
     return true;
}

/**
 * @brief Reads the dataset of the features.
 * 
 * @param in Each line of the dataset in string fromat.
 * @param x Then vector in which the features will be stored.
 * @param line Number of row of the dataset.
 * @param number_features Number of columns in the dataset which shows the number of features.
 */
void read_x(const string &in, vector<vector<double>> &x, const uint64_t &line, const uint64_t &number_features)
{
     string s;
     uint64_t number_columns = 0; //number of columns of csv
     istringstream string_stream(in);
     //Error if the number of columns is less than number of features
     class column_shortage : public length_error
     {
     public:
          column_shortage() : length_error("Number of columns is less than number of number_features! All lines should have equal number of columns."){};
     };
     //Error if the number of columns is more than the number of features
     class column_excess : public length_error
     {
     public:
          column_excess() : length_error("Number of columns is more than number of number_features! All lines should have equal number of columns."){};
     };
     //Error if the data is not a number
     class not_number : public invalid_argument
     {
     public:
          not_number() : invalid_argument("Expected a number!"){};
     };
     try
     {
          while (getline(string_stream, s, ','))
          {
               number_columns++;
               if (number_columns > number_features)
                    throw column_excess();
               if (!is_number(s))
                    throw not_number();
               x[line - 1][number_columns] = stod(s);
          }
          if (number_columns < number_features)
               throw column_shortage();
     }

     catch (const out_of_range &e)
     {
          throw out_of_range("Number is out of range!");
     }
}

/**
 * @brief Used for reading the y.csv file containing the outputs of our dataset which are classes. Also used fot reading layers.csv, which has the number of neurons in each layer
 * 
 * @param in Each line of the dataset in string fromat.
 * @param y Then vector in which the outputs will be stored.
 * @param line Number of row of the dataset.
 */
void read_y(const string &in, vector<uint64_t> &y, const uint64_t &line)
{
     //Error if the data is not a class (A class should be an integer number).
     class not_class : public invalid_argument
     {
     public:
          not_class() : invalid_argument("Expected an integer number!"){};
     };

     string s;
     istringstream string_stream(in);
     try
     {
          getline(string_stream, s);
          for (char &i : s)
          {
               if (isdigit(i) == false) //If it is not a number
                    throw not_class();
          }
          y[line - 1] = stoll(s);
     }
     catch (const out_of_range &e)
     {
          throw out_of_range("Number is out of range!");
     }
}

/* Function read_param reads the csv file containing the parameters of our model. */
/**
 * @brief Reads the parameters.csv file containing the parameters of our model.
 * 
 * @tparam T 
 * @param in Each line of the dataset in string fromat.
 * @param y A variable in which the parameter will be stored. 
 * @param line Number of row of the file.
 */
template <typename T>
void read_param(const string &in, T &y, const uint64_t &line)
{
     //Error if the data is not a class which should be an integer number
     class not_integer : public invalid_argument
     {
     public:
          not_integer() : invalid_argument("Expected an integer number!"){};
     };

     //Error if the data is not a real number.
     class not_number : public invalid_argument
     {
     public:
          not_number() : invalid_argument("Expected a number!"){};
     };

     string s;
     istringstream string_stream(in);
     //The first and second lines are the number of iterations and the number of cross validation iteration. Both should be integers.
     if (line == 1 || line == 2)
     {
          try
          {
               getline(string_stream, s);
               for (char &i : s)
               {
                    if (isdigit(i) == false)
                         throw not_integer();
               }
               y = stoll(s);
          }
          catch (const out_of_range &e)
          {
               throw out_of_range("Number is out of range!");
          }
     }
     //The third and forth lines are the learning rate and regularization. Both should be real numbers.
     if (line == 3 || line == 4)
     {
          try
          {
               getline(string_stream, s);
               if (!is_number(s))
                    throw not_number();
               y = stod(s);
          }

          catch (const out_of_range &e)
          {
               throw out_of_range("Number is out of range!");
          }
     }
}

/**
 * @brief To see if an element is in a vector or not.
 * 
 * @tparam T 
 * @param a The vector in which the element is being searched.
 * @param b The element we want to find.
 * @return true If b is in a.
 * @return false If b is not in a.
 */
template <typename T>
bool is_in_vec(const vector<T> a, const T b)
{
     for (const T &i : a)
     {
          if (b == i)
               return true;
     }
     return false;
}

/**
 * @brief Find an element in a vector and return the index number.
 * 
 * @tparam T 
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
     return -1;
}

/**
 * @brief Find the number of different values in a vector. Used to get the number of classes in the datset.
 * 
 * @tparam T 
 * @param a Vector of classes.
 * @return uint64_t Number of different classes.
 */
template <typename T>
uint64_t find_number_classes(const vector<T> &a)
{
     //is a set of all classes
     vector<T> classes;
     for (const T &i : a)
     {
          if (!is_in_vec(classes, i))
               classes.push_back(i);
     }
     return classes.size();
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
 * @brief Sigmoid function which is equal to y(x) = 1 / (1 + exp(-x)).
 * 
 * @tparam T 
 * @param x input of function
 * @return T output of function
 */
template <typename T>
T sigmoid(const T &x)
{
     return 1 / (1 + exp(-x));
}

//Activation of layer 1
/**
 * @brief Activates the first layer of the NN
 * 
 * @param neurons A vector containing all the neurons of the network.
 * @param x A vector contaning one row of the features dataset.
 */
void activate_layer1(vector<neuron> &neurons, vector<double> x)
{
     for (neuron &i : neurons)
     {
          if (i.get_layer() == 1)
          {

               i.set_activation(x[i.get_number()]); //Setting activation of the first layer neurons equal to the features values in the dataset.
          }

          else
               break;
     }
}

/**
 * @brief Activates layers l = 2, ..., L of the NN
 * 
 * @param neurons A vector containing all the neurons of the network.
 * @param edges A vector containing all the edges of the network.
 * @param layer Layer number that we want to do the activation for.
 */
void activate_layer(vector<neuron> &neurons, vector<edge> &edges, uint64_t &layer)
{
     for (neuron &i : neurons)
     {
          if (i.get_layer() == layer)
          {
               if (i.get_number() != 0)
                    i.set_activation(sigmoid(i.activate_neuron(neurons, edges))); //Setting activation of the layer using activate_neuron function for each neuron of the layer.
               else
                    i.set_activation(1);
          }
     }
}

/**
 * @brief Calculating the errors for the last layer. 
 * 
 * @param neurons A vector containing all the neurons of the network.
 * @param y Layer number that we want to calclate the error for.
 * @param number_layers Number of layers in the network.
 */
void error_layerL(vector<neuron> &neurons, vector<double> y, uint64_t number_layers)
{
     for (neuron &i : neurons)
     {
          if (i.get_layer() == number_layers)
          {

               i.set_error(i.get_activation() - y[i.get_number() - 1]); //Setting error of the last layer using the output values of the dataset.
          }
     }
}

/**
 * @brief Calculating the errors for layers l = 2, ..., L-1. 
 * 
 * @param neurons A vector containing all the neurons of the network.
 * @param edges A vector containing all the edges of the network.
 * @param layer Layer number that we want to do the activation for.
 */
void error_layer(vector<neuron> &neurons, vector<edge> &edges, uint64_t &layer)
{
     for (neuron &i : neurons)
     {
          if (i.get_layer() == layer)
          {
               if (i.get_number() != 0)
                    i.set_error(i.error_neuron(neurons, edges)); //Setting error of the layer using error_neuron function for each neuron of the layer.
          }
     }
}

/**
 * @brief A function which calculates the delta of an edge.
 * 
 * @param neurons A vector containing all the neurons of the network.
 * @param edge A vector containing all the edges of the network.
 * @return double Delta value of an edge.
 */
double delta_edge(const vector<neuron> &neurons, edge &edge)
{
     double activation = (neuron(edge.get_start_layer(), edge.get_start_number()).find_neuron(neurons)).get_activation();
     double error = (neuron(edge.get_start_layer() + 1, edge.get_end_number()).find_neuron(neurons)).get_error();
     edge.set_delta(edge.get_delta() + activation * error); //Updating delta using the activation and error values of the neurons related to that edge.
     return edge.get_delta();
}

/**
 * @brief Updating the delta for all the edges in NN
 * 
 * @param neurons A vector containing all the neurons of the network.
 * @param edges A vector containing all the edges of the network.
 */
void delta_update(const vector<neuron> &neurons, vector<edge> &edges)
{
     for (edge &i : edges)
     {
          delta_edge(neurons, i); //delta_edge function is used to update each edge's delta.
     }
}

/**
 * @brief Updating gradient for all edges of NN
 * 
 * @param edges A vector containing all the edges of the network.
 * @param number_instances Number of rows of the dataset.
 * @param lambda Regularization
 */
void gradient_update(vector<edge> &edges, const uint64_t &number_instances, const double &lambda)
{
     for (edge &i : edges)
     {
          i.gradient_edge(number_instances, lambda); //Updating gradient using gradient_edge function.
     }
}

/**
 * @brief Gradient descent algorithm which updates the weights of edges. 
 * 
 * @param edges A vector containing all the edges of the network.
 * @param learning_rate learning rate of the algorithm.
 */
void gradient_descent(vector<edge> &edges, const double &learning_rate)
{
     for (edge &i : edges)
     {
          i.set_weight(i.get_weight() - learning_rate * i.get_gradient());
     }
}

/**
 * @brief printing the elements of a vector.
 * 
 * @tparam T 
 * @param v vector to be printed.
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
 * @tparam T 
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
     // Reading file x.csv
     string s;
     uint64_t line = 1;         //counting number of instances in the files
     string filename = "x.csv"; //file containing the dataset features
     ifstream input(filename);
     if (!input.is_open())
     {
          cout << "Error opening " << filename << " input file!";
          return -1;
     }
     getline(input, s);

     uint64_t number_features = count(s.begin(), s.end(), ',') + 1; //number of features in the dataset

     while (getline(input, s))
     {
          line++;
     }
     uint64_t number_instances = line; //Number of rows of the dataset.

     vector<vector<double>> x(line, vector<double>(number_features + 1, 1)); //vector for saving dataset features
     input.clear();
     input.seekg(0, input.beg);
     line = 0;
     //reading x.csv and saving in x vector
     while (getline(input, s))
     {
          line++;
          try
          {
               read_x(s, x, line, number_features);
          }
          catch (const exception &e)
          {
               cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
               return -1;
          }
     }
     if (input.eof())
          cout << "Reached end of " << filename << "\n";
     input.close();

     //Reading file y.csv
     filename = "y.csv"; //file containing the outputs (classes)
     input.open(filename);
     if (!input.is_open())
     {
          cout << "Error opening " << filename << " input file!";
          return -1;
     }
     line = 0;
     while (getline(input, s))
     {
          line++;
     }

     //To check if both x.csv and y.csv have the same number of instances
     if (line != number_instances)
     {
          cout << "Error in" << filename << ": Number of rows is not the same as x.csv!";
          return -1;
     }

     //Vector for saving dataset outputs (classes)
     vector<uint64_t> classes(line);
     input.clear();
     input.seekg(0, input.beg);
     line = 0;
     //Saving y.csv in classes vector
     while (getline(input, s))
     {
          line++;
          try
          {
               read_y(s, classes, line);
          }
          catch (const exception &e)
          {
               cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
               return -1;
          }
     }
     if (input.eof())
          cout << "Reached end of " << filename << "\n";
     input.close();

     uint64_t number_classes = find_number_classes(classes); //Number of classes in the dataset

     /*Creating output vector proper for neural veector. Instead of class number, the output should be a vector with size equal to number of classes. The ith element, where i is the class number, should be 1 and the rest 0.*/
     vector<vector<double>> y(line, vector<double>(number_classes, 0));

     for (uint64_t i = 0; i < line; i++)
     {
          y[i][classes[i] - 1] = 1;
     }

     //Reading layers.csv
     filename = "layers.csv"; //file containing number of neurons in each layer (except the input and output layer)
     input.open(filename);
     if (!input.is_open())
     {
          cout << "Error opening " << filename << " input file!";
          return -1;
     }
     line = 0;
     while (getline(input, s))
     {
          line++;
     }

     vector<uint64_t> number_nodes(line); //A vector containing the number of neurons for each layer
     input.clear();
     input.seekg(0, input.beg);
     line = 0;
     //Saving layers.csv in number_nodes vector
     while (getline(input, s))
     {
          line++;
          try
          {
               read_y(s, number_nodes, line);
          }
          catch (const exception &e)
          {
               cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
               return -1;
          }
     }
     if (input.eof())
          cout << "Reached end of " << filename << "\n";
     input.close();

     //Reading parameters from parameters.csv
     uint64_t num_iteration = 0;  //number of iterations for the training
     uint64_t num_cv = 0;         //number of iterations for the cross validation
     double learning_rate = 0;    //Learning rate for the gradient decsent method
     double lambda = 0;           //Regularization
     filename = "parameters.csv"; //file containing the parameters
     input.open(filename);
     if (!input.is_open())
     {
          cout << "Error opening " << filename << " input file!";
          return -1;
     }

     //Reading the number of iterations for training
     line = 0;
     getline(input, s);
     line++;
     try
     {
          read_param(s, num_iteration, line);
     }
     catch (const exception &e)
     {
          cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
          return -1;
     }
     //Reading the number of iterations for cross validation (CV)
     getline(input, s);
     line++;
     try
     {
          read_param(s, num_cv, line);
     }
     catch (const exception &e)
     {
          cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
          return -1;
     }

     //Reading the learning rate
     getline(input, s);
     line++;
     try
     {
          read_param(s, learning_rate, line);
     }
     catch (const exception &e)
     {
          cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
          return -1;
     }

     //Reading the regularization
     getline(input, s);
     line++;
     try
     {
          read_param(s, lambda, line);
     }
     catch (const exception &e)
     {
          cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
          return -1;
     }

     if (input.eof())
          cout << "Reached end of " << filename << "\n";
     input.close();

     // Adding the number of features (number of neurons for the first layer), and the number of classes (the number of neurons for the last layer) to the vector number_nodes.
     number_nodes.insert(number_nodes.begin(), number_features);
     number_nodes.insert(number_nodes.end(), number_classes);

     //Number of layers in the defined architecture.
     uint64_t number_layers = number_nodes.size();

     vector<double> cv_accuracy(num_cv); //A vector for saving the accuracy of each trained model using NN.

     //Cross validation with num_cv iterations.
     for (uint64_t count = 0; count < num_cv; count++)
     {
          //Creating train and test sets by splitting data randomly to 70/30
          random_device rd;
          mt19937 mt(rd());
          vector<double> random_numbers(number_instances);                               //Creating random number for splitting data
          vector<double> random_numbers_sorted(number_instances);                        //A vector of the sorted random numbers
          vector<vector<double>> train_x(number_instances * 70 / 100);                   //Train set of x
          vector<vector<double>> test_x(number_instances - number_instances * 70 / 100); //Test set of x
          vector<vector<double>> train_y(number_instances * 70 / 100);                   //Train set of y
          vector<vector<double>> test_y(number_instances - number_instances * 70 / 100); //Test set of y
          vector<uint64_t> train_classes(number_instances * 70 / 100);                   //Classes of the train set
          vector<uint64_t> test_classes(number_instances - number_instances * 70 / 100); //Classes of the test set

          //Creating random numbers
          for (uint64_t i = 0; i < number_instances; i++)
          {
               uniform_real_distribution<double> urd(0, 1);
               random_numbers[i] = urd(mt);
          }

          for (uint64_t i = 0; i < number_instances; i++)
          {
               random_numbers_sorted[i] = random_numbers[i];
          }

          //Sorting random numbers and choosing the first 70 as train set.
          sort(random_numbers_sorted.begin(), random_numbers_sorted.end());

          //Train set
          uint64_t j = 0;
          for (uint64_t i = 0; i < number_instances * 70 / 100; i++)
          {
               train_x[j] = x[find_in_vec(random_numbers, random_numbers_sorted[i])];
               train_y[j] = y[find_in_vec(random_numbers, random_numbers_sorted[i])];
               train_classes[j] = classes[find_in_vec(random_numbers, random_numbers_sorted[i])];
               j++;
          }

          //Test set
          j = 0;
          for (uint64_t i = number_instances * 70 / 100; i < number_instances; i++)
          {
               test_x[j] = x[find_in_vec(random_numbers, random_numbers_sorted[i])];
               test_y[j] = y[find_in_vec(random_numbers, random_numbers_sorted[i])];
               test_classes[j] = classes[find_in_vec(random_numbers, random_numbers_sorted[i])];
               j++;
          }

          vector<neuron> neurons; //vector which includes neurons of NN
          //Generating neurons of NN
          for (uint64_t i = 1; i <= number_layers; i++)
          {
               for (uint64_t j = 0; j <= number_nodes[i - 1]; j++)
               {
                    if ((j == 0) && (i == number_layers))
                         continue;
                    neuron n(i, j);
                    neurons.push_back(n);
               }
          }

          vector<edge> edges; //vector of all edges of NN

          uint64_t ID = 0; // Counter for edge (edge ID)
          //generating input edges for each neuron
          for (neuron &i : neurons)
          {
               i.gen_input_edges(number_nodes, edges, ID);
          }

          //generating output edges for each neuron
          for (neuron &i : neurons)
          {
               i.gen_output_edges(number_nodes, edges);
          }

          //Training the network using train set for num_iteration iterations.
          for (uint64_t k = 0; k < num_iteration; k++)
          {
               //Setting delta equal to zero at the beginning of each iteration
               for (edge &i : edges)
                    i.set_delta(0);
               //Train using instance t.
               for (uint64_t t = 0; t < number_instances * 70 / 100; t++)
               {
                    //Activate layer 1 neurons
                    activate_layer1(neurons, train_x[t]);

                    //Activate the next layers
                    for (uint64_t i = 2; i <= number_layers; i++)
                    {
                         activate_layer(neurons, edges, i);
                    }

                    //Find error of the last layer.
                    error_layerL(neurons, train_y[t], number_layers);
                    //Find the error for the previous layers.
                    for (uint64_t i = number_layers - 1; i > 1; i--)
                    {
                         error_layer(neurons, edges, i);
                    }
                    //Update delta for each edge.
                    delta_update(neurons, edges);
               }
               //Update gradient of each edge.
               gradient_update(edges, number_instances * 70 / 100, lambda);
               //Run gradient descent algorithm to update the weights of the edges.
               gradient_descent(edges, learning_rate);
          }

          //Test the trained model on the test set.
          vector<uint64_t> predicted_classes(number_instances - number_instances * 70 / 100); //Vector containing the predicted classes.

          for (uint64_t t = 0; t < number_instances - number_instances * 70 / 100; t++)
          {
               //Activate layer 1 neurons
               activate_layer1(neurons, test_x[t]);

               //Activate the next layers
               for (uint64_t i = 2; i <= number_layers; i++)
               {
                    activate_layer(neurons, edges, i);
               }

               //Find category based on the neuron with maximum activation in the last layer.
               double max_activation = (neuron(number_layers, 1).find_neuron(neurons)).get_activation();
               uint64_t category = 1;
               //Update the category of the instance
               for (uint64_t i = 2; i <= number_classes; i++)
               {
                    if (((neuron(number_layers, i).find_neuron(neurons)).get_activation()) > max_activation)
                    {
                         max_activation = ((neuron(number_layers, i).find_neuron(neurons)).get_activation());
                         category = i;
                    }
               }
               predicted_classes[t] = category;
          }
          //Calculating the accuracy of the predicted classes
          cv_accuracy[count] = accuracy(predicted_classes, test_classes);
          cout
              << "\nTest set " << count + 1 << "\n\nPrediction accuracy: " << cv_accuracy[count] << "\n";
          cout << "\nPreticted classes for the test set:\n";
          print_elements(predicted_classes);
          cout << "\nActual classes for the test set:\n";
          print_elements(test_classes);
     }
     //Average accuracy of the all trained models.
     cout << "\nAverage accuracy: " << vec_average(cv_accuracy);
}