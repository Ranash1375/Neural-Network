#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
using namespace std;

// =========
// Interface
// =========

class edge
{
    /**
     * @brief Class neuron is a friend of class edge.
     * 
     */
    friend class neuron;

    /**
     * @brief Class network is a friend of class edge.
     * 
     */
    friend class network;

public:
    /**
    * @brief Construct a new edge::edge object.
    * 
    * @param _ID ID of the edge.
    * @param _start_layer Layer where the edge starts.
    * @param _start_number Neuron number for the start of the edge.
    * @param _end_number Neuron number for the end of the edge.
    */
    edge(const uint64_t &, const uint64_t &, const uint64_t &, const uint64_t &);

    /**
    * @brief  Member function to obtain (but not modify) the start layer of the edge.
    * 
    * @return uint64_t Start layer number of the edge.
    */
    uint64_t get_start_layer() const;

    /**
    * @brief  Member function to obtain (but not modify) the start number of the edge.
    * 
    * @return uint64_t Start number of the edge.
    */
    uint64_t get_start_number() const;

    /**
    * @brief  Member function to obtain (but not modify) the end number of the edge.
    * 
    * @return uint64_t End number of the edge.
    */
    uint64_t get_end_number() const;

    /**
    * @brief  Member function to obtain (but not modify) the weight of the edge.
    * 
    * @return double Weight of the edge.
    */
    double get_weight() const;

    /**
    * @brief  Member function to obtain (but not modify) the delta of the edge.
    * 
    * @return double Delta of the edge.
    */
    double get_delta() const;

    /**
    * @brief  Member function to obtain (but not modify) the gradient of the edge.
    * 
    * @return double Gradient of the edge.
    */
    double get_gradient() const;

    /**
    * @brief  Member function to obtain (but not modify) the ID of the edge.
    * 
    * @return uint64_t ID of the edge.
    */
    uint64_t get_ID() const;

    /**
    * @brief Weight initializer for randomly assigning weights to edges based on the number of neurons.
    * 
    * @param number_nodes Vector containing the number of neurons in each layer.
    */
    void weight_initializer(const vector<uint64_t> &);

    /**
    * @brief Member function to compute the gradient of the edge.
    * 
    * @param number_instances Number of instances in the dataset.
    * @param lambda Regularization.
    */
    void gradient_edge(const uint64_t &, const double &);

    /**
    * @brief Member function to modify the delta of the edge and set it equal to zero.
    * 
    */
    void set_delta_zero();

    /**
    * @brief Member function to find the edge in a vector of edges.
    * 
    * @param edges Vector of edges containing all the edges of the NN based on it ID or other variables.
    * @return edge Found edge.
    */
    edge find_edge(const vector<edge> &);

private:
    /**
     * @brief The ID of the edge.
     * 
     */
    uint64_t ID = 0;

    /**
     * @brief The start layer of the edge.
     * 
     */
    uint64_t start_layer = 0;

    /**
     * @brief The start number of the edge which is the number of neuron where the edge starts.
     * 
     */
    uint64_t start_number = 0;

    /**
     * @brief The end number of the edge which is the number of neuron where the edge ends.
     * 
     */
    uint64_t end_number = 0;

    /**
     * @brief Weight of edge.
     * 
     */
    double weight = 0;

    /**
     * @brief Delta of the edge.
     * 
     */
    double delta = 0;

    /**
     * @brief Gradient of the edge.
     * 
     */
    double gradient = 0;

    /**
     * @brief Random device for initializing weights.
     * 
     */
    inline static random_device rd;

    /**
     * @brief Pseudo-random number generator.
     * 
     */
    inline static mt19937 mt;
};

/**
 * @brief Overloaded binary operator << to easily print out an edge to a stream.
 * 
 * @param out Output stream.
 * @param m Edge.
 * @return ostream& The edge member variables.
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
    mt.seed(rd());
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

void edge::set_delta_zero()
{
    delta = 0;
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
    return edges[0]; // Just written for removing the warning. This line will never be executed in this problem.
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
