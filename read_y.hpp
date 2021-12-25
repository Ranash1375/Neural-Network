#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
using namespace std;

// =========
// Interface
// =========

class read_y
{

public:
    /**
    * @brief Construct a new read y::read y object which reads the output data from a file and saves to the vector of values.
    * 
    * @param filename The file name that contains the output data.
    */
    read_y(const string &);

    /**
    * @brief Member function to read each line of the y.csv file containing the outputs of our dataset which are classes and layers.csv which has the number of neurons in each layer.
    * 
    * @param in Each line of the dataset in string fromat.
    * @param line The number of row of the dataset.
    */
    void read_values(const string &, const uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the read data.
    * 
    * @return vector<uint64_t> Values of the output dataset.
    */
    vector<uint64_t> get_values() const;

    /**
    * @brief Member function to obtain (but not modify) the number of instances of the dataset.
    * 
    * @return uint64_t Number of rows (instances) of the dataset.
    */
    uint64_t get_rows() const;

    /**
    * @brief Member function to obtain the number of different values in member variable values. Used to get the number of classes of the datset.
    * 
    * @return uint64_t Number of different classes.
    */
    uint64_t find_number_classes() const;

    /**
     * @brief Error if the data is not a class (A class should be an integer number).
     * 
     */
    class not_class : public invalid_argument
    {
    public:
        not_class() : invalid_argument("Expected an integer number!"){};
    };

    /**
     * @brief Error if there is a problem with the file.
     * 
     */
    class invalid_file : public invalid_argument
    {
    public:
        invalid_file() : invalid_argument(""){};
    };

private:
    /**
     * @brief Number of rows (instances) of the dataset.
     * 
     */
    uint64_t rows = 0;

    /**
     * @brief A vector containing the data of the dataset.
     * 
     */
    vector<uint64_t> values;
};

/**
 * @brief Overloaded binary operator << to easily print out the output data to a stream.
 * 
 * @param out Output stream.
 * @param m The read_y object.
 * @return ostream& The values of the object.
 */
ostream &operator<<(ostream &, const read_y &);

/**
 * @brief To see if an element is in a vector or not.
 * 
 * @tparam T 
 * @param a The vector in which the element is being searched.
 * @param b The element we want to find.
 * @return true If b is in a.
 * @return false If b is not in a.
 */

bool is_in_vec(const vector<uint64_t>, const uint64_t);

// ==============
// Implementation
// ==============

vector<uint64_t> read_y::get_values() const
{
    return values;
}

uint64_t read_y::get_rows() const
{
    return rows;
}

read_y::read_y(const string &filename)
{
    // Reading file y.csv.
    ifstream input(filename);
    if (!input.is_open())
    {
        cout << "Error opening " << filename << " input file!";
        throw invalid_file();
    }
    uint64_t line = 0;
    string s;
    while (getline(input, s))
    {
        line++;
    }
    rows = line;
    values = vector<uint64_t>(rows); // Vector for saving dataset outputs (classes).
    input.clear();
    input.seekg(0, input.beg);
    line = 0;
    // Saving y.csv in values vector.
    while (getline(input, s))
    {
        line++;
        try
        {
            read_values(s, line);
        }
        catch (const exception &e)
        {
            cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
            throw invalid_file();
        }
    }
    if (input.eof())
        cout << "Reached end of " << filename << "\n";
    input.close();
}

void read_y::read_values(const string &in, const uint64_t &line)
{

    string s;
    istringstream string_stream(in);
    try
    {
        getline(string_stream, s);
        for (char &i : s)
        {
            if (isdigit(i) == false) // If it is not a number.
                throw not_class();
        }
        values[line - 1] = stoll(s);
    }
    catch (const out_of_range &e)
    {
        throw out_of_range("Number is out of range!");
    }
}

uint64_t read_y::find_number_classes() const
{
    // A set of all classes.
    vector<uint64_t> classes;
    for (const uint64_t &i : values)
    {
        if (!is_in_vec(classes, i))
            classes.push_back(i);
    }
    return classes.size();
}

ostream &operator<<(ostream &out, const read_y &m)
{
    out << '\n';
    out << "( ";
    for (uint64_t i = 0; i < m.get_rows(); i++)
        out << (m.get_values())[i] << '\t';
    out << ")\n";
    return out;
}

bool is_in_vec(const vector<uint64_t> a, const uint64_t b)
{
    for (const uint64_t &i : a)
    {
        if (b == i)
            return true;
    }
    return false;
}
