#include <iostream>
#include "NeuralNetwork.hpp"


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

void normalize(std::vector<vec>& data, double& mean_h, double& mean_m, double& var_h, double& var_m)
{
    mean_h = 0, mean_m = 0;
    var_h = 0, var_m = 0;


    for(size_t i = 0; i < data.size(); i++)
    {
        mean_h += data[i][0];
        mean_m += data[i][1];

        var_h += data[i][0] * data[i][0];
        var_m += data[i][1] * data[i][1];
    }

    mean_h /= data.size();
    mean_m /= data.size();
    
    var_h /= data.size();
    var_m /= data.size();

    var_h -= mean_h*mean_h;
    var_m -= mean_m*mean_m;

    for(size_t i = 0; i < data.size(); i++)
    {
        data[i].set(0, (data[i][0] - mean_h) / std::sqrt(var_h));
        data[i].set(1, (data[i][1] - mean_m) / std::sqrt(var_m));  
    }
}

int main()
{

    double mean_h;
    double mean_m;
    double var_h;
    double var_m;

    size_t sample_volume = 5;
    std::vector<vec> data = { 
        {165, 50},
        {181, 80},
        {176, 62},
        {192, 90},
        {160, 45}
    };
    std::vector<vec> target
    {
        {0},
        {1},
        {1},
        {1},
        {0}
    };


    normalize(data, mean_h, mean_m, var_h, var_m);

    for(auto v : data){
        std::cout << v << std::endl;
    }

    NeuralNetwork net(2);
    std::cout << net.get_weight(0) << std::endl;

    int epoch = 1000;
    for(size_t i = 0; i < epoch; i++)
    {
        for(size_t j = 0; j < sample_volume; j++)
        {
            net.forward(data[j]);
            net.back_propogation(target[j]);
        }
    }


    std::cout << net.get_weight(0) << std::endl;

    vec human(2);
    vec Y(1, 0.0);
    human.set(0, 155);
    human.set(1, 50);

    human.set(0, (human[0] - mean_h) / std::sqrt(var_h));
    human.set(1, (human[1] - mean_m) / std::sqrt(var_m));

    
    std::cout << net.predict(human);
    std::cout << "metric CROSS entropy = " << net.getMetric(Y) << std::endl;


}