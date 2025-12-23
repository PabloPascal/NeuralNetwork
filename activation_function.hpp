#ifndef ACTIVATE_FUNC
#define ACTIVATE_FUNC

#include <cmath>
#include "numerical/linalg.hpp"

struct activation
{
    virtual double operator()(double x) = 0;
    virtual LIN::Vector<double> operator()(const LIN::Vector<double>& vec) = 0;
    virtual double diff(double x) = 0;
    virtual LIN::Vector<double> diff(const LIN::Vector<double>& v) = 0;

};


struct sigmoid : activation
{
    double operator()(double x) override
    {
        return 1.0 /(1 + std::exp(-x));
    }
    LIN::Vector<double> operator()(const LIN::Vector<double>& vec) override
    {
        LIN::Vector<double> result(vec.getSize());
        for(size_t i = 0; i < vec.getSize(); i++)
        {
            result.set(i, this->operator()(vec[i]));
        }
        return result;

    }   
    double diff(double x) override
    {
        return this->operator()(x) * (1 - this->operator()(x) );
    }

    LIN::Vector<double> diff(const LIN::Vector<double>& v) override
    {
        LIN::Vector<double> res(v.getSize());
        for(int i = 0; i < v.getSize(); i++)
        {
            res[i] = diff(v[i]);
        }
        return res;
    }


};


#endif 