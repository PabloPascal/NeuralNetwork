#include <iostream>
#include "NeuralNetwork.hpp"
#include "activation_function.hpp"
#include "numerical/linalg.hpp"
#include "numerical/matrix_type.hpp"

#include <vector>
#include <memory>


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
    os << "size: ";
    os << vec.getSize();
    os << "\n";
    
    
    for(size_t i = 0; i < vec.getSize(); i++)
    {
        if(vec.isColumn()){
            os << vec[i] << "\n";

        }else
            os << vec[i] << " ";
    }

    return os;
}



int main()
{

    double mean_h;
    double mean_m;
    double var_h;
    double var_m;

    size_t sample_volume = 5;
    std::vector<LIN::vec_d> data = { 
        {165, 50},
        {181, 80},
        {176, 62},
        {192, 90},
        {160, 45}
    };
    std::vector<LIN::vec_d> target
    {
        {0},
        {1},
        {1},
        {1},
        {0}
    };

    std::unique_ptr<sigmoid> sigm = std::make_unique<sigmoid>();
    std::vector<size_t> arc = {2,2,1};
    
    NeuralNetwork net(arc, std::move(sigm));
    
    net.fit(data, target);
    net.train(1000);
    
    vec test1({155, 51});

    std::cout << net.predict(test1, true);


}