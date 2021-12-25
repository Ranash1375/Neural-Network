#include <iostream>
#include <vector>
using namespace std;

// =========
// Interface
// =========

class layer
{

public:
    /**
    * @brief Construct a new layer::layer object 
    * 
    * @param _layer_number Layer number.
    */
    layer(const uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the layer number of the layer.
    * 
    * @return uint64_t Layer number.
    */
    uint64_t get_layer_number() const;

    // Member function to obtain (but not modify) the layer number of a layer.
    /**
    * @brief Member function to generate the neurons of the layer.
    * 
    * @param number_nodes Vector containing the number of neurons in each layer.
    * @param neurons The set of all neurons of the NN.
    * @param start_ID ID of the first neuron for this layer (Or the number of neurons created before plus 1).
    */
    void gen_layer_neurons(const vector<uint64_t> &, vector<neuron> &, uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the neurons of the layer.
    * 
    * @return vector<uint64_t> The neurons of the layer.
    */
    vector<uint64_t> get_layer_neurons() const;

    /**
    * @brief  Member function to activate the layer.
    * 
    * @param neurons A vector containing all the neurons of the network.
    * @param edges A vector containing all the edges of the network.
    * @param x Feature values dataset.
    */
    void activate_layer(vector<neuron> &, vector<edge> &, vector<double> &);

    /**
    * @brief  Member function to calculate the errors for the layer. 
    * 
    * @param neurons A vector containing all the neurons of the network.
    * @param edges A vector containing all the edges of the network.
    * @param y The output values of the dataset.
    * @param number_layers Number of layers of the NN.
    */
    void error_layer(vector<neuron> &, vector<edge> &, vector<double> &, uint64_t &);

private:
    /**
     * @brief The number of the layer.
     * 
     */
    uint64_t layer_number = 0;

    /**
     * @brief Vector of neurons IDs of the layer
     * 
     */
    vector<uint64_t> layer_neurons;
};

/**
 * @brief Overloaded binary operator << to easily print out a layer a stream.
 * 
 * @param out Output stream.
 * @param m The layer.
 * @return ostream& The layer member variables.
 */
ostream &operator<<(ostream &, const layer &);

/**
 * @brief Sigmoid function which is equal to y(x) = 1 / (1 + exp(-x)).
 * 
 * @tparam T Template.
 * @param x Input of the function.
 * @return T Output of the function.
 */
template <typename T>
T sigmoid(const T &);

// ==============
// Implementation
// ==============

layer::layer(const uint64_t &_layer_number)
    : layer_number(_layer_number)
{
}

uint64_t layer::get_layer_number() const
{
    return layer_number;
}

void layer::gen_layer_neurons(const vector<uint64_t> &number_nodes, vector<neuron> &neurons, uint64_t &start_ID)
{

    for (uint64_t j = 0; j <= number_nodes[layer_number - 1]; j++)
    {
        if ((j == 0) && (layer_number == number_nodes.size()))
            continue;
        start_ID++;
        neuron n(start_ID, layer_number, j);
        neurons.push_back(n);
        layer_neurons.push_back(start_ID);
    }
}

vector<uint64_t> layer::get_layer_neurons() const
{
    return layer_neurons;
}

void layer::activate_layer(vector<neuron> &neurons, vector<edge> &edges, vector<double> &x)
{
    if (layer_number == 1)
    {
        for (neuron &i : neurons)
        {
            if (i.layer == layer_number)
            {
                i.activation = x[i.number]; // Setting activation of the first layer neurons equal to the features values in the dataset.
            }
        }
    }

    else
    {

        for (neuron &i : neurons)
        {
            if (i.layer == layer_number)
            {
                if (i.number != 0)
                    i.activation = sigmoid(i.activate_neuron(neurons, edges)); // Setting activation of the layer using activate_neuron() function for each neuron of the layer.
                else
                    i.activation = 1;
            }
        }
    }
}

void layer::error_layer(vector<neuron> &neurons, vector<edge> &edges, vector<double> &y, uint64_t &number_layers)
{
    if (layer_number == number_layers)
    {
        for (neuron &i : neurons)
        {
            if (i.layer == layer_number)
            {
                i.error = i.activation - y[i.number - 1]; // Setting error of the last layer using the output values of the dataset.
            }
        }
    }
    else
    {
        for (neuron &i : neurons)
        {
            if (i.layer == layer_number)
            {
                if (i.number != 0)
                    i.error = i.error_neuron(neurons, edges); // Setting error of the layer using error_neuron() function for each neuron of the layer.
            }
        }
    }
}

ostream &
operator<<(ostream &out, const layer &m)
{
    out << "\n layer number: " << m.get_layer_number();
    return out;
}

template <typename T>
T sigmoid(const T &x)
{
    return 1 / (1 + exp(-x));
}