#include "NeuralNetwork.hpp"
#include "logger.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>


std::pair<std::vector<LIN::vec_d>, std::vector<LIN::vec_d>> 
generateLinearRegression(size_t samples = 100, double noise_std = 0.1) {
    std::vector<LIN::vec_d> data;
    std::vector<LIN::vec_d> target;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> x_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    
    for (size_t i = 0; i < samples; ++i) {
        double x = x_dist(gen);
        double y = 2.0 * x + 1.0 + noise_dist(gen);
        
        data.push_back({x});
        target.push_back({y});
    }
    
    return {data, target};
}





void test_male_female()
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
    
    net.set_learning_rate(0.01);
    net.fit(data, target, true);
    net.train(1000);
    
    vec test_sample({165, 51});

    std::cout << "probability of men: " << net.predict(test_sample, true) << std::endl;

}




void test_regression()
{
    double mean_h;
    double mean_m;
    double var_h;
    double var_m;

    auto generated_data = generateLinearRegression();

    for(int i = 0; i < generated_data.first.size(); i++)
        std::cout << "x = " << generated_data.first[i][0] 
                  << "|     y = " << generated_data.second[i][0] 
                  << std::endl;



    std::vector<LIN::vec_d> data = generated_data.first;
    std::vector<LIN::vec_d> target = generated_data.second;


    std::unique_ptr<ReLu> relu = std::make_unique<ReLu>();
    
    std::vector<size_t> architecture = {1,10,1};
    
    NeuralNetwork net(architecture, std::move(relu));
    net.set_learning_rate(0.01);
    net.fit(data, target, true);
    net.train(1000);
    
    LIN::vec_d test_input = {0.75};
    auto prediction = net.predict(test_input, false);
    std::cout << "Prediction for x=0.75: " << prediction[0] << std::endl;

}


// Входные данные: 50 примеров, каждый вектор из 6 признаков
std::vector<LIN::vec_d> titanic_data = {
    {1.0, -0.592231, 1.0, 0.0, -0.668081},
    {0.0, 0.413008, 1.0, 0.0, 0.499008},
    {0.0, -0.389871, 0.0, 0.0, -0.615979},
    {0.0, 0.311028, 1.0, 0.0, 0.252676},
    {1.0, 0.311028, 0.0, 0.0, -0.608605},
    {1.0, -1.514492, 0.0, 0.0, -0.584259},
    {1.0, 0.413008, 0.0, 0.0, 0.703378},
    {0.0, 1.335349, 0.0, 2.0, 0.096783},
    {0.0, 1.031049, 1.0, 1.0, -0.444322},
    {1.0, -0.084591, 2.0, 0.0, 0.027965},
    {1.0, -1.007172, 0.0, 0.0, -0.273476},
    {0.0, -0.084591, 1.0, 0.0, 1.193526},
    {0.0, 0.209047, 0.0, 0.0, 0.235308},
    {1.0, -0.084591, 0.0, 0.0, -0.700377},
    {0.0, 0.717297, 0.0, 0.0, -0.686367},
    {1.0, 0.311028, 0.0, 0.0, -0.234403},
    {1.0, -1.617453, 0.0, 0.0, -0.504253},
    {0.0, 0.005389, 0.0, 2.0, -0.676374},
    {1.0, -1.209943, 0.0, 0.0, -0.221285},
    {0.0, 0.717297, 1.0, 0.0, -0.604821},
    {0.0, 2.155670, 1.0, 2.0, 0.567825},
    {1.0, -0.389871, 0.0, 0.0, -0.708670},
    {1.0, -0.389871, 0.0, 0.0, -0.572325},
    {1.0, 0.005389, 0.0, 0.0, 0.033332},
    {0.0, -0.592231, 0.0, 0.0, -0.543870},
    {1.0, -1.311924, 1.0, 1.0, 0.358570},
    {1.0, 1.031049, 0.0, 0.0, -0.476017},
    {1.0, -0.796192, 0.0, 0.0, -0.653207},
    {0.0, 0.005389, 0.0, 0.0, 1.174181},
    {1.0, 0.311028, 0.0, 0.0, -0.602109},
    {0.0, 0.413008, 0.0, 0.0, 0.831103},
    {1.0, 0.311028, 0.0, 0.0, -0.184608},
    {0.0, 0.615317, 1.0, 1.0, 0.096783},
    {0.0, 0.413008, 1.0, 0.0, -0.200407},
    {1.0, 2.155670, 0.0, 0.0, -0.484309},
    {0.0, -0.084591, 0.0, 0.0, 1.085745},
    {1.0, 0.005389, 0.0, 0.0, -0.704886},
    {0.0, 0.819278, 1.0, 0.0, -0.101862},
    {1.0, 0.209047, 0.0, 0.0, -0.687464},
    {1.0, 1.133030, 0.0, 0.0, -0.703604},
    {0.0, 0.921258, 0.0, 2.0, -0.107229},
    {0.0, -0.084591, 0.0, 1.0, -0.534269},
    {1.0, -0.084591, 1.0, 0.0, -0.558485},
    {0.0, -0.389871, 1.0, 0.0, 0.139528},
    {1.0, -0.592231, 0.0, 0.0, -0.479801},
    {1.0, -1.413511, 0.0, 0.0, -0.573422},
    {1.0, 0.717297, 1.0, 0.0, -0.442281},
    {0.0, -0.084591, 0.0, 0.0, -0.206995},
    {0.0, -0.694212, 1.0, 1.0, -0.435866},
    {1.0, -0.084591, 0.0, 0.0, -0.613315}
};

// Целевые значения (Survived) для каждого примера
std::vector<LIN::vec_d> titanic_target = {
    {0.0}, // 1
    {1.0}, // 2
    {1.0}, // 3
    {1.0}, // 4
    {0.0}, // 5
    {0.0}, // 6
    {0.0}, // 7
    {1.0}, // 8
    {1.0}, // 9
    {0.0}, // 10
    {0.0}, // 11
    {1.0}, // 12
    {1.0}, // 13
    {0.0}, // 14
    {1.0}, // 15
    {1.0}, // 16
    {0.0}, // 17
    {1.0}, // 18
    {1.0}, // 19
    {1.0}, // 20
    {1.0}, // 21
    {0.0}, // 22
    {0.0}, // 23
    {0.0}, // 24
    {1.0}, // 25
    {1.0}, // 26
    {0.0}, // 27
    {0.0}, // 28
    {1.0}, // 29
    {0.0}, // 30
    {1.0}, // 31
    {0.0}, // 32
    {1.0}, // 33
    {1.0}, // 34
    {0.0}, // 35
    {1.0}, // 36
    {0.0}, // 37
    {1.0}, // 38
    {0.0}, // 39
    {0.0}, // 40
    {1.0}, // 41
    {1.0}, // 42
    {0.0}, // 43
    {1.0}, // 44
    {0.0}, // 45
    {0.0}, // 46
    {0.0}, // 47
    {1.0}, // 48
    {1.0}, // 49
    {0.0}  // 50
};


void split_train_test(
    const std::vector<LIN::vec_d>& data,
    const std::vector<LIN::vec_d>& target,
    std::vector<LIN::vec_d>& train_data,
    std::vector<LIN::vec_d>& train_target,
    std::vector<LIN::vec_d>& test_data,
    std::vector<LIN::vec_d>& test_target,
    double train_ratio = 0.8,
    bool shuffle = true)
{
    size_t total = data.size();
    size_t train_size = static_cast<size_t>(total * train_ratio);
    
    // Создаем индексы
    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        // Перемешиваем с random seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
    }
    
    train_data.clear();
    train_target.clear();
    test_data.clear();
    test_target.clear();
    
    for (size_t i = 0; i < train_size; ++i) {
        train_data.push_back(data[indices[i]]);
        train_target.push_back(target[indices[i]]);
    }
    for (size_t i = train_size; i < total; ++i) {
        test_data.push_back(data[indices[i]]);
        test_target.push_back(target[indices[i]]);
    }
}




void test_titanic()
{


    double mean_h;
    double mean_m;
    double var_h;
    double var_m;

    std::vector<LIN::vec_d> train_data, train_target, test_data, test_target;
    split_train_test(titanic_data, titanic_target, train_data, train_target, test_data, test_target, 0.8, true);
    
    std::cout << "Train size: " << train_data.size() << ", Test size: " << test_data.size() << std::endl;
    
    // Архитектура сети
    std::vector<size_t> architecture = {5, 8, 1};  // 5 входов, скрытый слой 8 нейронов, выход 1
    auto hidden_act = std::make_unique<ReLu>();
    auto output_act = std::make_unique<sigmoid>();
    
    NeuralNetwork net(architecture, std::move(hidden_act), std::move(output_act));
    net.set_learning_rate(0.1);
    
    // Обучаем на тренировочных данных с нормализацией
    net.fit(train_data, train_target, true);   // третий параметр = true для нормализации
    net.train(5000);  // количество эпох
    
    // Проверка на тестовых данных
    int correct = 0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        auto pred = net.predict(test_data[i], true);  // применяем ту же нормализацию
        int predicted_class = (pred[0] > 0.5) ? 1 : 0;
        int true_class = static_cast<int>(test_target[i][0]);
        if (predicted_class == true_class) {
            ++correct;
        }
        // Для отладки можно вывести предсказания
        // std::cout << "Sample " << i << ": pred=" << pred[0] << " class=" << predicted_class << " true=" << true_class << std::endl;
    }
    
    double accuracy = static_cast<double>(correct) / test_data.size();
    std::cout << "Test accuracy: " << accuracy << " (" << correct << "/" << test_data.size() << ")" << std::endl;


    
}




void testXOR()
{
    std::vector<LIN::vec_d> xor_data = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<LIN::vec_d> xor_target = {{0},{1},{1},{0}};

    std::vector<size_t> arch = {2, 4, 1}; // скрытый слой 4 нейрона
    auto hidden = std::make_unique<ReLu>();
    auto output = std::make_unique<sigmoid>();
    NeuralNetwork net(arch, std::move(hidden), std::move(output));
    net.set_learning_rate(0.5);
    net.fit(xor_data, xor_target, false);
    net.train(2000);

    for (size_t i = 0; i < xor_data.size(); ++i) {
        auto pred = net.predict(xor_data[i], false);
        std::cout << xor_data[i][0] << "," << xor_data[i][1] << " -> " << pred[0] << " (true " << xor_target[i][0] << ")\n";
    }

}



