#include "NeuralNetwork.hpp"
#include <iostream>


NeuralNetwork::NeuralNetwork(
    std::vector<size_t>& architecture, 
    std::unique_ptr<activation> a_func):
m_active(std::move(a_func)),
m_architecture(architecture), 
m_input(m_architecture[0]),
m_output(m_architecture[m_architecture.size()-1])
{   

    for(size_t i = 0; i < m_architecture.size() - 1; i++)
    {
        mat w(m_architecture[i + 1], m_architecture[i]); 
        w.random_init(0, 100, true);
        vec bias(m_architecture[i + 1]);
        bias.random_init(0, 100, true);

        m_weights.emplace_back(std::move(w));
        m_bias.emplace_back(std::move(bias));
        m_hidden.emplace_back(vec(m_architecture[i + 1]));
        m_zValues.emplace_back(vec(m_architecture[i+1]));
    }   


}


void NeuralNetwork::forward(const vec& input_data)
{
    m_input = input_data;

    m_zValues[0] = m_weights[0]*m_input + m_bias[0];
    m_hidden[0] = (*m_active)(m_zValues[0]);


    for(size_t layer = 1; layer < m_weights.size(); layer++)
    {
        m_zValues[layer] = m_weights[layer] * m_hidden[layer - 1] + m_bias[layer];
        m_hidden[layer] = (*m_active)(m_zValues[layer]);
    }

    m_output = m_hidden[m_weights.size() - 1];

}



void NeuralNetwork::back_propogation(const vec& target)
{
    size_t last_layer = m_weights.size() - 1;

    //1 step 
    vec delta_output = LIN::hadamar_product(d_MSE(target, m_output), m_active->diff(m_zValues[last_layer])); 
    
    mat dW_last = LIN::outer_product(delta_output, m_hidden[last_layer - 1]);
    vec db_last = delta_output;
    
    std::vector<mat> gradients_W(m_weights.size());
    std::vector<vec> gradients_b(m_bias.size());
    std::vector<vec> deltas(m_weights.size());

    deltas[last_layer] = delta_output;
    gradients_W[last_layer] = dW_last;
    gradients_b[last_layer] = db_last;     

    for(int l = last_layer - 1; l >= 0; l--) 
    {

        vec delta_current;
        vec prev_hidden;

        delta_current = LIN::hadamar_product(LIN::transpose(m_weights[l + 1]) * deltas[l + 1], m_active->diff(m_zValues[l])); 

        if(l==0)
        {
            prev_hidden = m_input;
        }
        else prev_hidden = m_hidden[l - 1];

        gradients_W[l] = LIN::outer_product(delta_current, prev_hidden);
        gradients_b[l] = delta_current;
        deltas[l] = delta_current;
    }
    

    for(int l = 0; l <= last_layer; l++)
    {
        m_weights[l] = m_weights[l] - m_learning_rate*gradients_W[l];
        m_bias[l] = m_bias[l] - m_learning_rate * gradients_b[l];
    }


}




void NeuralNetwork::fit(const std::vector<vec>& train_data, const std::vector<vec>& target_data)
{
    m_train_data = train_data;
    m_target_data = target_data;

    normalize_data();
}



void NeuralNetwork::normalize_data()
{
    size_t object_len = m_train_data[0].getSize();
    size_t feature_len = m_train_data.size();

    means.resize(object_len);
    variance.resize(object_len);

    for(size_t row = 0; row < feature_len ; row++)
    {
        for(size_t col = 0; col < object_len; col++)
        {
            means[col] += m_train_data[row][col] / feature_len;
            variance[col] += m_train_data[row][col] * m_train_data[row][col] / feature_len; 
        }
    }
    
    for(size_t col = 0; col < object_len; col++){
        std::cout << "mean = " << means[col] << ", ";
        variance[col] -= means[col]*means[col];  
        variance[col] = std::sqrt(variance[col]);

        std::cout << "var = " << variance[col] << std::endl;

    }
    

    for(size_t row = 0; row < feature_len ; row++)
    {
        for(size_t col = 0; col < object_len; col++)
        {
            double cur_data = m_train_data[row][col];
            m_train_data[row][col] = (cur_data - means[col]) / variance[col];
        }
    }
}




void NeuralNetwork::train(size_t EPOCH_NUM)
{   
    for(size_t epoch = 0; epoch < EPOCH_NUM; epoch++)
    {
        for(size_t i = 0; i < m_train_data.size(); i++)
        {
            forward(m_train_data[i]);
            back_propogation(m_target_data[i]);
        }
    }

}




vec NeuralNetwork::predict(vec data, bool normalize)
{
    if(normalize)
    {
        for(size_t i = 0; i < data.getSize(); i++)
        {
            data[i] = (data[i] - means[i]) / variance[i];
        }
    }

    forward(data);
    return m_output;

}