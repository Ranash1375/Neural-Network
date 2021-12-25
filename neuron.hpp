#include <iostream>
#include <vector>
using namespace std;

// =========
// Interface
// =========

class neuron
{
    /**
     * @brief Class layer is a friend of class neuron.
     * 
     */
    friend class layer;

    /**
     * @brief Class network is a friend of class neuron.
     * 
     */
    friend class network;

public:
    /**
    * @brief Construct a new neuron::neuron object.
    * 
    * @param _ID ID of neuron.
    * @param _layer Number of layer for that neuron.
    * @param _number Neuron number in that layer.
    */
    neuron(const uint64_t &, const uint64_t &, const uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the ID of a neuron.
    * 
    * @return uint64_t Neuron ID.
    */
    uint64_t get_ID() const;

    /**
    * @brief Member function to obtain (but not modify) the number of layer of a neuron.
    * 
    * @return uint64_t Layer number.
    */
    uint64_t get_layer() const;

    /**
    * @brief Member function to obtain (but not modify) the number of neuron in its layer.
    * 
    * @return uint64_t Neuron number in its layer.
    */
    uint64_t get_number() const;

    /**
    * @brief Member function to obtain (but not modify) the activation of a neuron.
    * 
    * @return double Activation of the neuron.
    */
    double get_activation() const;

    /**
    * @brief Member function to obtain (but not modify) the error of a neuron.
    * 
    * @return double Error of the neuron.
    */
    double get_error() const;

    /**
    * @brief Member function to generate input edges of the neuron.
    * 
    * @param number_nodes Vector containing the number of nodes in each layer (Except the bias unit).
    * @param edges Vector in which all the edges for the NN will be stored.
    * @param start_ID ID of the first edge for this neuron (Or the number of edges created before plus 1).
    */
    void gen_input_edges(const vector<uint64_t> &, vector<edge> &, uint64_t &);

    /**
    * @brief Member function to generate output edges of the neuron.
    * 
    * @param number_nodes Vector containing the number of nodes in each layer.
    * @param edges Vector containing all the edges of the NN.
    */
    void gen_output_edges(const vector<uint64_t> &, const vector<edge> &);

    /**
    * @brief Member function to obtain (but not modify) the ID of input edges of the neuron.
    * 
    * @return vector<uint64_t> The vector of input edges IDs.
    */
    vector<uint64_t> get_input_edges() const;

    /**
    * @brief Member function to obtain (but not modify) the ID of output edges of the neuron.
    * 
    * @return vector<uint64_t> The vector of output edges IDs.
    */
    vector<uint64_t> get_output_edges() const;

    /**
    * @brief Member function to find the neuron in a set of a neurons based on its ID or other variables.
    * 
    * @param neurons The set of all neurons of the NN.
    * @return neuron The neuron we are searching for.
    */
    neuron find_neuron(const vector<neuron> &) const;

    /**
    * @brief Member function to activate neurons of the hidden and output layers neurons.
    * 
    * @param neurons  Vector containing all the neurons of the NN.
    * @param edges Vector containing all the edges of the NN.
    * @return double Activation of the neuron.
    */
    double activate_neuron(const vector<neuron> &, const vector<edge> &);

    /**
    * @brief Member function to calculate the error for neurons of the hidden and output layers neurons.
    * 
    * @param neurons Vector containing all the neurons of the NN.
    * @param edges Vector containing all the edges of the NN.
    * @return double Error of the neuron.
    */
    double error_neuron(const vector<neuron> &, const vector<edge> &);

private:
    /**
     * @brief The ID of the neuron.
     * 
     */
    uint64_t ID = 0;

    /**
     * @brief The layer of the neuron.
     * 
     */
    uint64_t layer = 0;

    /**
     * @brief The number of the neuron in its layer.
     * 
     */
    uint64_t number = 0;

    /**
     * @brief Activation of the neuron.
     * 
     */
    double activation = 0;

    /**
     * @brief Error of the neuron.
     * 
     */
    double error = 0;

    /**
     * @brief Vector of input edges IDs of the neuron.
     * 
     */
    vector<uint64_t> input_edges;

    /**
     * @brief Vector of output edges IDs of the neuron.
     * 
     */
    vector<uint64_t> output_edges;
};

/**
 * @brief Overloaded binary operator << to easily print out a neuron to a stream.
 * 
 * @param out Output ostream.
 * @param m Neuron.
 * @return ostream& The neuron member variables (ID, layer, number, activation, and error).
 */
ostream &operator<<(ostream &, const neuron &);

// ==============
// Implementation
// ==============

neuron::neuron(const uint64_t &_ID, const uint64_t &_layer, const uint64_t &_number)
    : ID(_ID), layer(_layer), number(_number)
{
}

uint64_t neuron::get_ID() const
{
    return ID;
}

uint64_t neuron::get_layer() const
{
    return layer;
}

uint64_t neuron::get_number() const
{
    return number;
}

double neuron::get_activation() const
{
    return activation;
}

double neuron::get_error() const
{
    return error;
}

vector<uint64_t> neuron::get_input_edges() const
{
    return input_edges;
}

vector<uint64_t> neuron::get_output_edges() const
{
    return output_edges;
}

void neuron::gen_input_edges(const vector<uint64_t> &number_nodes, vector<edge> &edges, uint64_t &start_ID)
{

    if ((layer > 1) && (number > 0))
    {

        for (uint64_t i = 0; i <= number_nodes[layer - 2]; i++)
        {
            start_ID++;
            edge e(start_ID, layer - 1, i, number);
            e.weight_initializer(number_nodes);
            edges.push_back(e);
            input_edges.push_back(start_ID);
        }
    }
}

void neuron::gen_output_edges(const vector<uint64_t> &number_nodes, const vector<edge> &edges)
{

    if (layer < number_nodes.size())
    {
        for (uint64_t i = 1; i <= number_nodes[layer]; i++)
        {
            edge e(0, layer, number, i);
            output_edges.push_back((e.find_edge(edges).ID));
        }
    }
}

neuron neuron::find_neuron(const vector<neuron> &neurons) const
{
    for (const neuron &i : neurons)
    {
        if (i.ID == ID)
            return i;
    }
    for (const neuron &i : neurons)
    {
        if (i.layer == layer && i.number == number)
        {

            return i;
        }
    }
    return neurons[0]; // Just written for removing the warning. This line will never be executed in this problem.
}

double neuron::activate_neuron(const vector<neuron> &neurons, const vector<edge> &edges)
{
    if (number == 0)
        activation = 1;
    else
    {
        activation = 0;
        for (uint64_t &i : input_edges)
        {
            edge e = edge(i, 0, 0, 0).find_edge(edges);
            activation += e.weight * (neuron(0, e.start_layer, e.start_number).find_neuron(neurons)).activation;
        }
        return activation;
    }
    return 0; // Just written for removing the warning. This line will never be executed in this problem.
}

double neuron::error_neuron(const vector<neuron> &neurons, const vector<edge> &edges)
{

    error = 0;
    for (uint64_t &i : output_edges)
    {
        edge e = edge(i, 0, 0, 0).find_edge(edges);
        error += e.weight * (neuron(0, e.start_layer + 1, e.end_number).find_neuron(neurons)).error;
    }
    return error * activation * (1 - activation);
}

ostream &operator<<(ostream &out, const neuron &m)
{
    out << "\n ID: " << m.get_ID() << " layer: " << m.get_layer() << " number: " << m.get_number() << " activation: " << m.get_activation() << " error: " << m.get_error();

    return out;
}
