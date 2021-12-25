#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

// =========
// Interface
// =========

class read_x
{

public:
    /**
    * @brief Construct a new read x::read x object which reads the features data from a file and saves to the vector of values.
    * 
    * @param filename The file name that contains the features data.
    */
    read_x(const string &);

    /**
    * @brief Member function to read each line of the dataset of the features (x.csv).
    * 
    * @param in Each line of the dataset in string fromat.
    * @param line The number of row of the dataset.
    * @param number_features Number of columns in the dataset which shows the number of features.
    */
    void read_values(const string &, const uint64_t &, const uint64_t &);

    /**
    * @brief Member function to obtain (but not modify) the read data.
    * 
    * @return vector<vector<double>> Values of the features dataset.
    */
    vector<vector<double>> get_values() const;

    /**
    * @brief Member function to obtain (but not modify) the number of instances of the dataset.
    * 
    * @return uint64_t Number of rows (instances) of the dataset.
    */
    uint64_t get_rows() const;

    /**
    * @brief Member function to obtain (but not modify) the number of features of the dataset.
    * 
    * @return uint64_t Number of columns (features) of the dataset.
    */
    uint64_t get_cols() const;

    /**
     * @brief Error if the number of columns is less than the number of features.
     * 
     */
    class column_shortage : public length_error
    {
    public:
        column_shortage() : length_error("Number of columns is less than number of features! All lines should have equal number of columns."){};
    };

    /**
     * @brief Error if the number of columns is more than the number of features.
     * 
     */
    class column_excess : public length_error
    {
    public:
        column_excess() : length_error("Number of columns is more than number of features! All lines should have equal number of columns."){};
    };

    /**
     * @brief Error if the data in the file is not a number.
     * 
     */
    class not_number : public invalid_argument
    {
    public:
        not_number() : invalid_argument("Expected a number!"){};
    };

    /**
     * @brief Error if there are any problems with the file.
     * 
     */
    class invalid_file : public invalid_argument
    {
    public:
        invalid_file() : invalid_argument(""){};
    };

private:
    /**
     * @brief The number of rows in the dataset file.
     * 
     */
    uint64_t rows = 0;

    /**
     * @brief The number of columns in the dataset file.
     * 
     */
    uint64_t columns = 0;

    /**
     * @brief A vector containing the data of the dataset.
     * 
     */
    vector<vector<double>> values;
};

/**
 * @brief Overloaded binary operator << to easily print out the data of the features dataset to a stream.
 * 
 * @param out Output stream.
 * @param m The read_x object.
 * @return ostream& The values of the object.
 */
ostream &operator<<(ostream &, const read_x &);

/**
 * @brief Checks a string to see if that could be a real number.
 * 
 * @param s The string to be converted to a number.
 * @return true If the string s is a real number.
 * @return false If the string s is a real number.
 */

bool is_number(const string &s);

// ==============
// Implementation
// ==============

vector<vector<double>> read_x::get_values() const
{
    return values;
}

uint64_t read_x::get_rows() const
{
    return rows;
}

uint64_t read_x::get_cols() const
{
    return columns;
}

read_x::read_x(const string &filename)
{
    ifstream input(filename);
    if (!input.is_open())
    {
        cout << "Error opening " << filename << " input file!";
        throw invalid_file();
    }
    string s;
    getline(input, s);
    columns = count(s.begin(), s.end(), ',') + 1; // Number of features of the dataset.
    uint64_t line = 1;
    while (getline(input, s))
    {
        line++;
    }
    rows = line;                                                           // Number of rows of the dataset.
    values = vector<vector<double>>(rows, vector<double>(columns + 1, 1)); // Vector for saving dataset features.
    // Reading input and saving in values vector.
    input.clear();
    input.seekg(0, input.beg);
    line = 0;
    while (getline(input, s))
    {
        line++;
        try
        {
            read_values(s, line, columns);
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

void read_x::read_values(const string &in, const uint64_t &line, const uint64_t &number_features)
{
    string s;
    uint64_t number_columns = 0; // Number of columns counter.
    istringstream string_stream(in);
    try
    {
        while (getline(string_stream, s, ','))
        {
            number_columns++;
            if (number_columns > number_features)
                throw column_excess();
            if (!is_number(s))
                throw not_number();
            values[line - 1][number_columns] = stod(s);
        }
        if (number_columns < number_features)
            throw column_shortage();
    }

    catch (const out_of_range &e)
    {
        throw out_of_range("Number is out of range!");
    }
}

bool is_number(const string &s)
{
    bool has_point = false;            // To check if there is more than 1 point in the string.
    uint64_t ascii = s[0];             // To save the ascii number of each character.
    if (!isdigit(s[0]) && ascii != 45) // If the first character is not a minus or number.
        return false;
    for (uint64_t i = 1; i < s.size(); i++)
    {
        uint64_t ascii = s[i];
        if (!isdigit(s[i]) && ascii != 46) // If the character is not a number or point.
            return false;
        if (ascii == 46 && has_point == true) // If there are more than one points in the number.
            return false;
        if (ascii == 46) // If the char is a point has-point to true.
            has_point = true;
    }
    return true;
}

ostream &operator<<(ostream &out, const read_x &m)
{
    out << '\n';
    for (uint64_t i = 0; i < m.get_rows(); i++)
    {
        out << "( ";
        for (uint64_t j = 0; j < m.get_cols(); j++)
            out << (m.get_values())[i][j] << '\t';
        out << ")\n";
    }
    return out;
}