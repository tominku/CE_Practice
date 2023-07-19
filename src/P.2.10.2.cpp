#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

const double n_int = 1e10;    
const double T = 300; // (K)
const double V_T = k_B * T / q;

double f(double phi, double dop)
{            
    double f_value = n_int * ( exp(phi / V_T) - exp(-phi / V_T) ) - dop;
    return f_value;
}

double df_over_dx(double phi)
{
    double derivative_value = n_int * ( (1.0 / V_T)*exp(phi / V_T) - (-1.0 / V_T)*exp(-phi / V_T) );
    return derivative_value;
}

void plot(stdvec &doping_list, stdvec &phis);

int main() {

    double x_0 = 3;    
    double dop = 1e18;
    int num_iters = 500;
    stdvec xs;
    stdvec residuals;

    int num_experiments = 9;
    vec phis(num_experiments);
    vec doping_list(num_experiments);
    doping_list[0] = 1e10;
    for (int i=1; i<doping_list.n_elem; ++i)
    {
        doping_list[i] = doping_list[0] * pow(10, i); 
    }

    for (int k=0; k<doping_list.n_elem; ++k)
    {
        double x_i = x_0;
        double dop = doping_list[k];
        for (int i=0; i<num_iters; i++)
        {
            double residual = f(x_i, dop);
            xs.push_back(x_i);        
            residuals.push_back(residual);
            double delta_x = (1 / df_over_dx(x_i) ) * ( - residual );
            x_i += delta_x;                    
        }
        phis[k] = x_i;
        printf("doping: 1e%d, x_sol: %f \n", 10+k, x_i);
    }

    plot_args args;        
    args.x_label = "Doping Density (/cm^3)";
    args.y_label = "Electrostatic Potential(V)";
    args.logscale = 10;

    plot(doping_list, phis, args);
}