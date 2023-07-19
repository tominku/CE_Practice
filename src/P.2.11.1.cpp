#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

const int N = 61;
double deltaX = 0.1e-9; // in meter  
double dop = 1e18 * 1e6; // in meter    
int n_int = 1e10 * 1e6;;
double T = 300;    
double total_width = 6.0;    
double t_ox = 0.5;
double t_si = 5;
int interface1_i = 6;
int interface2_i = 56;
int si_begin_i = interface1_i + 1;
int si_end_i = interface2_i - 1;

// residual(phi): the size of r(phi) is (N - 2), but the size of phi is N.
vec r(vec phi)
{
    int start_i = 2;
    int last_i = N-1;    
    vec phi_i_minus_1 = phi(span(start_i - 1 - 1, last_i - 1 - 1));
    vec phi_i = phi(span(start_i - 1, last_i - 1));
    vec phi_i_plus_1 = phi(span(start_i - 1 + 1, last_i - 1 + 1));
    vec r_i = (eps_si/deltaX) * (phi_i_plus_1 - 2.0*phi_i + phi_i_minus_1);     
    r_i(span(si_begin_i-1, si_end_i-1)) += ( deltaX*q*dop - deltaX*q*n_int*exp(q*phi(span(si_begin_i-1, si_end_i-1))/(k_B*T)) );
    return r_i;
}

// the jacobian matrix size is (N - 2) by (N - 2)
mat jacobian(vec phi)
{
    mat jac(N-2, N-2, fill::zeros);    
    int offset = 2;
    for (int i=2; i<=(interface1_i - 1); ++i)
    {
        jac(i - offset, i + 1 - offset) = eps_ox / deltaX;
        jac(i - offset, i - offset) =  -2.0 * eps_ox / deltaX;
        jac(i - offset, i - 1 - offset) = eps_ox / deltaX; 
    }

    for (int i=si_begin_i; i<=si_end_i; ++i)
    {
        jac(i - 1, i + 1 - 1) = eps_si / deltaX;
        jac(i - 1, i - 1) =  -2.0 * eps_si / deltaX - deltaX*q*n_int*exp(q*phi(i-1)/(k_B*T)) ;
        jac(i - 1, i - 1 - 1) = eps_si / deltaX; 
    }

    for (int i=(interface2_i + 1); i<=(N-1); ++i)
    {
        jac(i - 1, i + 1 - 1) = eps_ox / deltaX;
        jac(i - 1, i - 1) =  -2.0 * eps_ox / deltaX;
        jac(i - 1, i - 1 - 1) = eps_ox / deltaX; 
    }

    return jac;
}

int main() {

    vec phi_0(N, arma::fill::ones);
    int num_iters = 10;
    //mat xs(num_iters, 3, arma::fill::zeros); // each row i represents the solution at iter i.
    //mat residuals(num_iters, 3, arma::fill::zeros); // each row i represents the residual at iter i.    
    vec phi_i = phi_0;
    for (int i=0; i<num_iters; i++)
    {
        vec residual = r(phi_i);
        mat jac = jacobian(phi_i);
        printf("test");
        // xs.row(i) = x_i.t();         
        // residuals.row(i) = residual.t();
        // mat jac = jacobian(x_i);
        // vec delta_x = arma::solve(jac, -residual);
        // x_i += delta_x;        
        
        // printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_x)), max(abs(residual)));        
    }

    //x_i.print("found solution (phi):");
}