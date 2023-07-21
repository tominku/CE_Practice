#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

double f(double x)
{
    return x*x - 1;
}

double df_over_dx(double x)
{
    return 2*x;
}

void plot(stdvec &xs, stdvec &residuals);

int main() {

    double x_0 = -2;    
    int num_iters = 20;
    stdvec xs;
    stdvec residuals;
    double x_i = x_0;
    for (int i=0; i<num_iters; i++)
    {
        double residual = f(x_i);
        xs.push_back(x_i);        
        residuals.push_back(residual);
        double delta_x = (1 / df_over_dx(x_i) ) * ( - residual );
        x_i += delta_x;        
        printf("x_i: %f \n", x_i);
    }

    plot_args args;        
    args.x_label = "x_i";
    args.y_label = "residual: f(x)";    
    vec vec_xs = arma::conv_to<vec>::from(xs);
    vec vec_residuals = arma::conv_to<vec>::from(residuals);
    plot(vec_xs, vec_residuals, args);
}