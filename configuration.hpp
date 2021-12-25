#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

// =========
// Interface
// =========

class configuration
{

public:
    /**
    * @brief Construct a new configuration::configuration object which reads the parameters of the algorithm.
    * 
    * @param filename The file name that contains the parameters.
    */
    configuration(const string &);

    /**
    * @brief Member function to read the lines of the file which contain integer numbers.
    * 
    * @param in The line of the dataset in string fromat.
    * @return uint64_t The integer parameter.
    */
    uint64_t read_int_values(const string &);

    /**
    * @brief Member function to read the lines of the file which contain double numbers.
    * 
    * @param in The line of the dataset in string fromat.
    * @return double The double parameter.
    */
    double read_double_values(const string &);

    /**
    * @brief Member function to obtain (but not modify) the number of iterations for training the network.
    * 
    * @return uint64_t Number of iterations.
    */
    uint64_t get_num_iteration() const;

    /**
    * @brief Member function to obtain (but not modify) the number of cross-validations performed.
    * 
    * @return uint64_t Number of cross-validations performed.
    */
    uint64_t get_num_cv() const;

    /**
    * @brief Member function to obtain (but not modify) the percentage of data chosen for training for each cross-validation.
    * 
    * @return uint64_t Training percentage.
    */
    uint64_t get_train_percantage() const;

    /**
    * @brief Member function to obtain (but not modify) the learning rate for gradient descent.
    * 
    * @return double Learning rate.
    */
    double get_learning_rate() const;

    /**
    * @brief Member function to obtain (but not modify) the regularization parameter.
    * 
    * @return double Regularization parameter.
    */
    double get_lambda() const;

    /**
     * @brief Error if the data is not a class which should be an integer number.
     * 
     */
    class not_integer : public invalid_argument
    {
    public:
        not_integer() : invalid_argument("Expected an integer number!"){};
    };

    /**
     * @brief Error if the data is not a real number.
     * 
     */
    class not_number : public invalid_argument
    {
    public:
        not_number() : invalid_argument("Expected a number!"){};
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

    /**
     * @brief Error if the percentage is larger than 100.
     * 
     */
    class invalid_percentage : public invalid_argument
    {
    public:
        invalid_percentage() : invalid_argument("The training percentage should be less than 100!"){};
    };

private:
    /**
    * @brief Number of iterations for training network.
    * 
    */
    uint64_t num_iteration = 0;

    /**
    * @brief Number of cross-validation iterations.
    * 
    */
    uint64_t num_cv = 0;

    /**
     * @brief Percantage of data chosen as train set.
     * 
     */
    uint64_t train_percentage = 0;

    /**
     * @brief Learning rate for the gradient decsent algorithm.
     * 
     */
    double learning_rate = 0;

    /**
     * @brief Regularization parameter.
     * 
     */
    double lambda = 0;
};

/**
 * @brief Overloaded binary operator << to easily print out the parameters to a stream.
 * 
 * @param out Output stream.
 * @param m The configuration object.
 * @return ostream& The parameters.
 */
ostream &operator<<(ostream &, const configuration &);

// ==============
// Implementation
// ==============

configuration::configuration(const string &filename)
{
    ifstream input(filename);
    if (!input.is_open())
    {
        cout << "Error opening " << filename << " input file!";
        throw invalid_file();
    }

    // Reading the number of iterations for training.
    uint64_t line = 0;
    string s;

    getline(input, s);
    line++;
    try
    {
        num_iteration = read_int_values(s);
    }
    catch (const exception &e)
    {
        cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
        throw invalid_file();
    }

    // Reading the number of iterations for cross validation (CV).
    getline(input, s);
    line++;
    try
    {
        num_cv = read_int_values(s);
    }
    catch (const exception &e)
    {
        cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
        throw invalid_file();
    }

    // Reading the training percentage.
    getline(input, s);
    line++;
    try
    {
        train_percentage = read_int_values(s);
        if (train_percentage >= 100)
            throw invalid_percentage();
    }
    catch (const exception &e)
    {
        cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
        throw invalid_file();
    }

    // Reading the learning rate.
    getline(input, s);
    line++;
    try
    {
        learning_rate = read_double_values(s);
    }
    catch (const exception &e)
    {
        cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
        throw invalid_file();
    }

    // Reading the regularization.
    getline(input, s);
    line++;
    try
    {
        lambda = read_double_values(s);
    }
    catch (const exception &e)
    {
        cout << "Error in line " << line << " " << filename << ": " << e.what() << '\n';
        throw invalid_file();
    }

    if (input.eof())
        cout << "Reached end of " << filename << "\n";
    input.close();
}

uint64_t configuration::get_num_iteration() const
{
    return num_iteration;
}

uint64_t configuration::get_num_cv() const
{
    return num_cv;
}

uint64_t configuration::get_train_percantage() const
{
    return train_percentage;
}

double configuration::get_learning_rate() const
{
    return learning_rate;
}

double configuration::get_lambda() const
{
    return lambda;
}

uint64_t configuration::read_int_values(const string &in)
{

    uint64_t y = 0;
    string s;
    istringstream string_stream(in);
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
    return y;
}

double configuration::read_double_values(const string &in)
{

    double y = 0;
    string s;
    istringstream string_stream(in);
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
    return y;
}

ostream &operator<<(ostream &out, const configuration &m)
{
    out << '\n';
    out << "\n num_iteration: " << m.get_num_iteration();
    out << "\n num_cv: " << m.get_num_cv();
    out << "\n train_percentage: " << m.get_train_percantage();
    out << "\n learning_rate: " << m.get_learning_rate();
    out << "\n lambda: " << m.get_lambda();
    out << '\n';
    return out;
}
