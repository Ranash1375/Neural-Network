#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>
using namespace std;

//For random initialization of weight of an edge
random_device rd;
mt19937 mt(rd());

// =========
// Interface
// =========

class edge
{
public:
    /**
    * @brief Construct a new edge::edge object
    * 
    * @param _ID ID of edge
    * @param _start_layer layer where the edge starts
    * @param _start_number number of neuron for the start of the edge
    * @param _end_number number of neuron for the end of the edge
    */
    edge(const uint64_t &, const uint64_t &, const uint64_t &, const uint64_t &);

    /**
    * @brief  Member function to obtain (but not modify) the start layer of the edge.
    * 
    * @return uint64_t start layer number
    */
    uint64_t get_start_layer() const;

    /**
    * @brief  Member function to obtain (but not modify) the start number of the edge.
    * 
    * @return uint64_t start number
    */
    uint64_t get_start_number() const;

    /**
    * @brief  Member function to obtain (but not modify) the end number of the edge.
    * 
    * @return uint64_t start number
    */
    uint64_t get_end_number() const;

    /**
    * @brief  Member function to obtain (but not modify) the weight of the edge.
    * 
    * @return double weight
    */
    double get_weight() const;

    /**
    * @brief  Member function to obtain (but not modify) the delta of the edge.
    * 
    * @return double delta
    */
    double get_delta() const;

    /**
    * @brief  Member function to obtain (but not modify) the gradient of the edge.
    * 
    * @return double gradient
    */
    double get_gradient() const;

    /**
    * @brief  Member function to obtain (but not modify) the ID of the edge.
    * 
    * @return uint64_t ID
    */
    uint64_t get_ID() const;

    /**
    * @brief Weight initializer for randomly assigning weights to edges based on the number of neurons
    * 
    * @param number_nodes vector containing number of neurons in each layer
    */
    void weight_initializer(const vector<uint64_t> &);

    /**
    * @brief Compute gradient of the edge
    * 
    * @param number_instances number of instances in the dataset.
    * @param lambda Regularization.
    */
    void gradient_edge(const uint64_t &, const double &);

    /**
    * @brief Member function to modify the delta of the edge.
    * 
    * @param a delta value
    */
    void set_delta(const double &);

    /**
    * @brief Member function to modify the wight of the edge.
    * 
    * @param a weight value
    */
    void set_weight(const double &);

    /**
    * @brief Finds the edge in a vector of edges.
    * 
    * @param edges Vector of edges containing all the edges of the NN.
    * @return edge Found edge.
    */
    edge find_edge(const vector<edge> &);

private:
    //edge ID
    uint64_t ID = 0;

    // The start layer of neuron
    uint64_t start_layer = 0;

    // The start neuron number.
    uint64_t start_number = 0;

    // The end neuron number.
    uint64_t end_number = 0;

    //weight of edge
    double weight = 0;

    //Delta of edge
    double delta = 0;

    //Gradient of edge
    double gradient = 0;
};

/**
 * @brief Overloaded binary operator << to easily print out an edge to a steam.
 * 
 * @param out 
 * @param m edge
 * @return ostream& edge variables.
 */
ostream &operator<<(ostream &, const edge &);

// ==============
// Implementation
// ==============

edge::edge(const uint64_t &_ID, const uint64_t &_start_layer, const uint64_t &_start_number, const uint64_t &_end_number)
    : ID(_ID), start_layer(_start_layer), start_number(_start_number), end_number(_end_number)
{
}

uint64_t edge::get_start_layer() const
{
    return start_layer;
}

uint64_t edge::get_start_number() const
{
    return start_number;
}

uint64_t edge::get_end_number() const
{
    return end_number;
}

double edge::get_weight() const
{
    return weight;
}

double edge::get_delta() const
{
    return delta;
}

double edge::get_gradient() const
{
    return gradient;
}

uint64_t edge::get_ID() const
{
    return ID;
}

void edge::weight_initializer(const vector<uint64_t> &number_nodes)
{
    double epsilon = sqrt(6) / sqrt(number_nodes[start_layer - 1] + number_nodes[start_layer]);
    uniform_real_distribution<double> urd(-epsilon, epsilon);
    weight = urd(mt);
}

void edge::gradient_edge(const uint64_t &number_instances, const double &lambda)
{
    if (start_number == 0)
        gradient = delta / (double)number_instances;
    else
        gradient = (delta + lambda * weight) / (double)number_instances;
}

void edge::set_delta(const double &a)
{
    delta = a;
}

void edge::set_weight(const double &a)
{
    weight = a;
}

edge edge::find_edge(const vector<edge> &edges)
{
    for (const edge &i : edges)
    {
        if (i.ID == ID)
            return i;
    }

    for (const edge &i : edges)
    {
        if (i.start_layer == start_layer && i.start_number == start_number && i.end_number == end_number)
            return i;
    }
    return edges[0];
}

ostream &operator<<(ostream &out, const edge &m)
{
    out << "\n ID: " << m.get_ID()
        << " start layer: " << m.get_start_layer()
        << " start number: " << m.get_start_number()
        << " end number: " << m.get_end_number()
        << " weight: " << m.get_weight()
        << " delta: " << m.get_delta()
        << " gradient: " << m.get_gradient();

    return out;
}