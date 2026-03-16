
#include <iostream>
#include "numerical/linalg.hpp"
#include "numerical/matrix_type.hpp"

std::ostream& operator<<(std::ostream& os, const LIN::Matrix<double>& matr)
{

    os << "rows: ";
    os << matr.get_rows();
    os << ", cols ";
    os << matr.get_cols();
    os << "\n";

    for(size_t i = 0; i < matr.get_rows(); i++)
    {
        for(size_t j = 0; j < matr.get_cols(); j++)
        {
            os << matr(i, j);
            os << " ";
        }
        os << "\n";
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const LIN::Vector<double>& vec)
{
       
    for(size_t i = 0; i < vec.getSize(); i++)
    {
        if(vec.isColumn()){
            os << vec[i] << "\n";

        }else
            os << vec[i] << " ";
    }

    return os;
}
