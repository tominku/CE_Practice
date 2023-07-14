#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

vec f(vec vec_phi)
{
    double phi_1 = vec_phi(0);
    double phi_2 = vec_phi(1);
    double phi_3 = vec_phi(2);
    
    double f1 = phi_2 - 2*phi_1 - exp(phi_1);
    double f2 = phi_3 - 2*phi_2 + phi_1 - exp(phi_2);
    double f3 = -2*phi_3 + phi_2 - exp(phi_3) + 4;

    vec vec_f = {f1, f2, f3};
    return vec_f;
}


mat jacobian(vec vec_phi)
{
    double phi_1 = vec_phi(0);
    double phi_2 = vec_phi(1);
    double phi_3 = vec_phi(2);

    mat J = {   { -2 - exp(phi_1),   1,   0 },
                { 1,   -2 - exp(phi_2),   1},
                { 0,   1,   -2 - exp(phi_3)}   };

    return J;
}

void plot(stdvec &xs, stdvec &residuals);

int main() {

    vec x_0 = {1.0, 2.0, 3.0};    
    int num_iters = 10;
    mat xs(num_iters, 3, arma::fill::zeros); // each row i represents the solution at iter i.
    mat residuals(num_iters, 3, arma::fill::zeros); // each row i represents the residual at iter i.
    vec x_i = x_0;
    for (int i=0; i<num_iters; i++)
    {
        vec residual = f(x_i);
        xs.row(i) = x_i.t();         
        residuals.row(i) = residual.t();
        mat jac = jacobian(x_i);
        vec delta_x = arma::solve(jac, -residual);
        x_i += delta_x;        
        
        printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_x)), max(abs(residual)));        
    }

    x_i.print("found solution (phi):");
}