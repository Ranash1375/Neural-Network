#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
using namespace std;

// =========
// Interface
// =========

class network
{

public:
    /**
    * @brief Construct a new network::network object
    * 
    * @param _number_neurons The number of neurons of the network.
    * @param _number_edges The number of edges of the network.
    */
    network(const uint64_t &, const uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the number of neurons of the network.
    * 
    * @return uint64_t Number of neurons of the network.
    */
    uint64_t get_neurons_number() const;

    /**
    * @brief Member function to obtain (but not modify) the number of edges of the network.
    * 
    * @return uint64_t Number of edges of the network.
    */
    uint64_t get_edges_number() const;

    /**
    * @brief Member function to update the delta for all the edges of the network.
    * 
    * @param neurons A vector containing all the neurons of the network.
    * @param edges A vector containing all the edges of the network.
    */
    void delta_update(const vector<neuron> &, vector<edge> &);

    /**
    * @brief Member function to update gradient for all the edges of the network.
    * 
    * @param edges A vector containing all the edges of the network.
    * @param number_instances Number of rows of the dataset.
    * @param lambda Regularization parameter.
    */
    void gradient_update(vector<edge> &, const uint64_t &, const double &);

    /**
    * @brief Gradient descent algorithm which updates the weights of the edges. 
    * 
    * @param edges A vector containing all the edges of the network.
    * @param learning_rate Learning rate of the gradient descent algorithm.
    */
    void gradient_descent(vector<edge> &, const double &);

private:
    /**
     * @brief The number of neurons of the network.
     * 
     */
    uint64_t number_neurons = 0;

    /**
     * @brief The number of edges of the network.
     * 
     */
    uint64_t number_edges = 0;
};

// ==============
// Implementation
// ==============

network::network(const uint64_t &_number_neurons, const uint64_t &_number_edges)
    : number_neurons(_number_neurons), number_edges(_number_edges)
{
}

uint64_t network::get_neurons_number() const
{
    return number_neurons;
}

uint64_t network::get_edges_number() const
{
    return number_edges;
}

void network::delta_update(const vector<neuron> &neurons, vector<edge> &edges)
{
    for (edge &i : edges)
    {
        double activation = (neuron(0, i.start_layer, i.start_number).find_neuron(neurons)).activation;
        double error = (neuron(0, i.start_layer + 1, i.end_number).find_neuron(neurons)).error;
        i.delta += activation * error; // Updating delta using the activation and error values of the neurons related to this edge.
    }
}

void network::gradient_update(vector<edge> &edges, const uint64_t &number_instances, const double &lambda)
{
    for (edge &i : edges)
    {
        i.gradient_edge(number_instances, lambda); // Updating gradient using gradient_edge function.
    }
}

void network::gradient_descent(vector<edge> &edges, const double &learning_rate)
{
    for (edge &i : edges)
    {
        i.weight -= learning_rate * i.gradient;
    }
}