#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std;

// =========
// Interface
// =========

class neuron
{
public:
    friend class edge;

    /**
    * @brief Construct a new neuron::neuron object.
    * 
    * @param _layer Number of layer for that neuron.
    * @param _number Neuron number in that layer.
    */
    neuron(const uint64_t &, const uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the number of layer of a neuron.
    * 
    * @return uint64_t layer number
    */
    uint64_t get_layer() const;

    /**
    * @brief Member function to obtain (but not modify) the number of neuron in a layer.
    * 
    * @return uint64_t neuron number in a layer
    */
    uint64_t get_number() const;

    /**
    * @brief Member function to obtain (but not modify) the activation of a neuron.
    * 
    * @return double Activation
    */
    double get_activation() const;

    /**
    * @brief Member function to obtain (but not modify) the error of a neuron.
    * 
    * @return double error
    */
    double get_error() const;

    /**
    * @brief Member function to modify the activation of a neuron.
    * 
    * @param a activation value 
    */
    void set_activation(const double &);

    /**
    * @brief Member function to modify the error of a neuron.
    * 
    * @param a error value 
    */
    void set_error(const double &);

    /**
    * @brief Generate input edges of the neuron
    * 
    * @param number_nodes vector containing number of nodes in each layer.
    * @param edges vector in which all the edges for the NN will be stores.
    * @param start_ID ID of the first edge for this neuron (Or the number of edges created before).
    */
    void gen_input_edges(const vector<uint64_t> &, vector<edge> &, uint64_t &);

    /**
    * @brief Generate output edges of the neuron
    * 
    * @param number_nodes vector containing number of nodes in each layer.
    * @param edges vector containing all the edges of the NN.
    */
    void gen_output_edges(const vector<uint64_t> &, const vector<edge> &);

    /**
    * @brief Member function to obtain (but not modify) the ID of input edges of a neuron.
    * 
    * @return vector<uint64_t> The vector of input edges IDs.
    */
    vector<uint64_t> get_input_edges() const;

    /**
    * @brief Member function to obtain (but not modify) the ID of output edges of a neuron.
    * 
    * @return vector<uint64_t> The vector of output edges IDs.
    */
    vector<uint64_t> get_output_edges() const;

    /**
    * @brief Member function to find a neuron in a set of a neuron
    * 
    * @param neurons  The set of all neurons of the NN.
    * @return neuron The neuron we are searching for.
    */
    neuron find_neuron(const vector<neuron> &) const;

    /**
    * @brief Activates neurons of the inner and output layer neurons.
    * 
    * @param neurons  Vector containing all the neurons of the NN.
    * @param edges Vector containing all the edges of the NN.
    * @return double activation of the neuron
    */
    double activate_neuron(const vector<neuron> &, const vector<edge> &);

    /**
    * @brief Calculates the error for neurons of the inner and output layer neurons.
    * 
    * @param neurons Vector containing all the neurons of the NN.
    * @param edges Vector containing all the edges of the NN.
    * @return double Error of the neuron
    */
    double error_neuron(const vector<neuron> &, const vector<edge> &);

private:
    // The layer of neuron
    uint64_t layer = 0;

    // The number of neuron.
    uint64_t number = 0;

    //Activation of neuron
    double activation = 0;

    //error of neuron
    double error = 0;

    //Vector of input edges IDs
    vector<uint64_t> input_edges;

    //Vector of output edges IDs
    vector<uint64_t> output_edges;
};

//  matrix to a stream.
/**
 * @brief Overloaded binary operator << to easily print out a neuron to a steam.
 * 
 * @param out 
 * @param m neuron
 * @return ostream& neuron variables (layer, number, activation, and error).
 */
ostream &operator<<(ostream &, const neuron &);

// ==============
// Implementation
// ==============

neuron::neuron(const uint64_t &_layer, const uint64_t &_number)
    : layer(_layer), number(_number)
{
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

void neuron::set_activation(const double &a)
{
    activation = a;
}

void neuron::set_error(const double &a)
{
    error = a;
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
            input_edges.push_back(e.get_ID());
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
            output_edges.push_back((e.find_edge(edges).get_ID()));
        }
    }
}

neuron neuron::find_neuron(const vector<neuron> &neurons) const
{
    for (const neuron &i : neurons)
    {
        if (i.layer == layer && i.number == number)
        {

            return i;
        }
    }
    return neurons[0];
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
            activation += e.get_weight() * (neuron(e.get_start_layer(), e.get_start_number()).find_neuron(neurons)).activation;
        }
        return activation;
    }
    return -1;
}

double neuron::error_neuron(const vector<neuron> &neurons, const vector<edge> &edges)
{

    error = 0;
    for (uint64_t &i : output_edges)
    {
        edge e = edge(i, 0, 0, 0).find_edge(edges);
        error += e.get_weight() * (neuron(e.get_start_layer() + 1, e.get_end_number()).find_neuron(neurons)).error;
    }
    return error * activation * (1 - activation);
}

ostream &operator<<(ostream &out, const neuron &m)
{
    out << "\n layer: " << m.get_layer() << " number: " << m.get_number() << " activation: " << m.get_activation() << " error: " << m.get_error();

    return out;
}
