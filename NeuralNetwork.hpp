#ifndef NeuralNetwork_HPP
#define NeuralNetwork_HPP

#include "activation_function.hpp"

#include "numerical/linalg.hpp"
#include "numerical/matrix_type.hpp"

#include <cmath>
#include <memory>
#include <vector>

using vec = LIN::Vector<double>;
using mat = LIN::Matrix<double>;



inline double MSE(const vec& y_true, const vec& y_pred)
{
    return LIN::dot_product(y_true - y_pred, y_true - y_pred) / y_true.getSize();
}

inline vec d_MSE(const vec& y_true, const vec& y_pred)
{
    vec res(y_true.getSize());
    for(size_t i = 0; i < y_true.getSize(); i++)
        res[i] = 2.0 * (y_pred[i] - y_true[i]) / y_true.getSize();
    
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
    std::vector<size_t> m_architecture;  
    std::vector<vec>    m_hidden;
    std::vector<vec>    m_zValues;
    std::vector<mat>    m_weights;
    std::vector<vec>    m_bias;
    vec                 m_input;
    vec                 m_output;
    

    std::unique_ptr<activation> m_active;
    double                      m_learning_rate = 0.5; 

    std::vector<vec>    m_train_data;
    std::vector<vec>    m_target_data;

    std::vector<double> means;
    std::vector<double> variance;
public:


    NeuralNetwork(std::vector<size_t>& architecture, 
                 std::unique_ptr<activation> a_func = std::make_unique<sigmoid>());

    void set_activation_function(std::unique_ptr<activation>&& func) {
        m_active = std::move(func);
    }
    void set_learning_rate(double learning_rate) {m_learning_rate = learning_rate;}


    vec predict(vec data, bool normalize);

    double getMetric(vec y_true){
        return CROSS_ENTROPY(y_true, m_output);
    }


    vec get_output()const {return m_output;}
    vec get_hidden(size_t index)const {return m_hidden[index];}
    mat get_weight(size_t index) {return m_weights[index];}
    std::vector<vec> get_train_data() {return m_train_data;}
    std::vector<vec> get_target_data() {return m_target_data;}

    void fit(const std::vector<vec>& train_data, const std::vector<vec>& target_data);
    void train(size_t EPOCH_NUM);

private:
    void normalize_data();
    void forward(const vec& train_data);
    void back_propogation(const vec& target);

};

#endif 