#ifndef NeuralNetwork_HPP
#define NeuralNetwork_HPP

#include "activation_function.hpp"

#include "numerical/linalg.hpp"
#include <cmath>
#include <memory>

using vec = LIN::Vector<double>;
using mat = LIN::Matrix<double>;


inline double MSE(const vec& y,const vec& y_pred)
{
    return len(y - y_pred);
}

inline double d_MSE(const vec& y, const vec& y_pred)
{
    double res = 0;
    for(size_t i = 0; i < y.getSize(); i++)
        res += y[i] - y_pred[i];
    
    return res;
}


inline double CROSS_ENTROPY(const vec& y,const vec& y_pred)
{
    return y[0] * std::log(y_pred[0]) + (1-y[0])*std::log(1-y_pred[0]);
}

inline double d_CROSS_ENTROPY(const vec& y, const vec& y_pred)
{
    return (y - y_pred)[0];
}




class NeuralNetwork
{
    
    std::vector<vec> hidden;
    std::vector<mat> weights;
    std::vector<vec> bias;
    vec input;
    vec output;
    
    sigmoid sigm;
    activation& activate = sigm;
    double train_speed = 1; 

public:


    NeuralNetwork(size_t len);
    void set_activation(activation* func) {
        activate = *func;
    }
    void forward(const vec& train_data);
    void back_propogation(const vec& target);

    vec predict(vec data){
        forward(data);
        return output;
    }

    double getMetric(vec y_true){
        return CROSS_ENTROPY(y_true, output);
    }

    inline vec get_output()const {return output;}
    inline vec get_hidden(size_t index)const {return hidden[index];}
    inline mat get_weight(size_t index) {return weights[index];}



};

#endif 