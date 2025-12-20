#include "NeuralNetwork.hpp"
#include <iostream>


NeuralNetwork::NeuralNetwork(size_t len) : input(len), output(1)
{   
    LIN::Matrix<double> W1(2, 2);
    LIN::Matrix<double> W2(1, 2);

    W1.random_init();
    W2.random_init();

    W1 = W1 * (1.0 / 100.0);
    W2 = W2 * (1.0 / 100.0);

    vec b1(2);
    vec b2(1);

    b1.random_init();
    b2.random_init();

    b1 = (1.0/ 100.0) * b1;
    b2 = (1.0/ 100.0) * b2;

    weights.push_back(W1);
    weights.push_back(W2);
    
    bias.push_back(b1);
    bias.push_back(b2);
    
    hidden.push_back(vec(2));
    hidden.push_back(vec(1));
    
}


void NeuralNetwork::forward(const vec& train_data)
{
    input = train_data;

    hidden[0] = weights[0] * input + bias[0];
    hidden[0] = activate(hidden[0]);

    hidden[1] = weights[1] * hidden[0] + bias[1];
    output = activate(hidden[1]);
}



void NeuralNetwork::back_propogation(const vec& target)
{
    //1 step 
    double delta_o = d_CROSS_ENTROPY(output, target) * activate.diff(hidden[1][0]);
    
    double delta_h1 = delta_o * weights[1](0,0) * hidden[0][0] * (1 - hidden[0][0]);
    double delta_h2 = delta_o * weights[1](0,1) * hidden[0][1] * (1 - hidden[0][1]);

    double dw1E = delta_o * hidden[0][0];
    double dw2E = delta_o * hidden[0][1]; 

    double db3E = delta_o;

    double dw11E = delta_h1 * input[0];
    double dw12E = delta_h1 * input[1];
    double dw21E = delta_h2 * input[0];
    double dw22E = delta_h2 * input[1];
    
    double db1E = delta_h1; 
    double db2E = delta_h2;

    mat dWE(2,2);
    dWE.set(0, 0, dw11E);
    dWE.set(0, 1, dw12E);
    dWE.set(1, 0, dw21E);
    dWE.set(1, 1, dw22E);

    mat dWE1(1,2);
    dWE1.set(0,0, dw1E);
    dWE1.set(0,1, dw2E);

    vec dbE(2);
    dbE.set(0, db1E);
    dbE.set(1, db2E);
     
    vec dbE1(1);
    dbE1.set(0, db3E);


    weights[0] = weights[0] + (-train_speed) * dWE;


    weights[1] = weights[1] + (-train_speed) * dWE1;
    bias[0] = bias[0] - train_speed * dbE;
    bias[1] = bias[1] - train_speed * dbE1;
    
}