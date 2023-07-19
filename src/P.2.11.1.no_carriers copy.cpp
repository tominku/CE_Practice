#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

const int N = 61;
double deltaX = 0.1e-9; // in meter  
double dop = 1e18 * 1e6; // in meter    
//int n_int = 1e10;
//double n_int = 1e16;
//double T = 300;    
// double total_width = 6.0;    
// double t_ox = 0.5;
// double t_si = 5;
int interface1_i = 6;
int interface2_i = 56;
int si_begin_i = interface1_i + 1;
int si_end_i = interface2_i - 1;

// residual(phi): the size of r(phi) is N.
vec r(vec phi)
{   
    vec r_k(N, arma::fill::zeros);

    r_k(span(1, interface1_i-1-1)) = (eps_ox) * (-2*phi(span(1, interface1_i-1-1)) + phi(span(0, interface1_i-1-1-1)) + phi(span(2, interface1_i-1)));
    
    r_k(interface1_i-1) = -(eps_ox)*phi(interface1_i-1) - (eps_si)*phi(interface1_i-1) + 
        (eps_ox)*phi(interface1_i-1-1) + (eps_si)*phi(interface1_i-1+1) - 0.5 * deltaX*deltaX*q*dop;

    r_k(span(si_begin_i-1, si_end_i-1)) = (eps_si) * (-2*phi(span(si_begin_i-1, si_end_i-1)) + phi(span(si_begin_i-2, si_end_i-2)) + phi(span(si_begin_i, si_end_i)));    
    r_k(span(si_begin_i-1, si_end_i-1)) -= deltaX*deltaX*q*dop;

    r_k(interface2_i-1) = -(eps_si)*phi(interface2_i-1) - (eps_ox)*phi(interface2_i-1) + 
        (eps_si)*phi(interface2_i-1-1) + (eps_ox)*phi(interface2_i-1+1) - 0.5 * deltaX*deltaX*q*dop;

    r_k(span(interface2_i-1+1, N-1-1)) = (eps_ox) * (-2*phi(span(interface2_i-1+1, N-1-1)) + phi(span(interface2_i-1, N-1-1-1)) + phi(span(interface2_i-1+1+1, N-1-1+1)));
    
    return r_k;
}

// the jacobian matrix size is (N - 2) by (N - 2)
mat jacobian(vec phi)
{
    mat jac(N, N, arma::fill::zeros);    
    jac(0, 0) = 1.0;
    jac(N-1, N-1) = 1.0;    
    for (int i=2; i<=(interface1_i - 1); ++i)    
    {
        jac(i - 1, i + 1 - 1) = eps_ox ;
        jac(i - 1, i - 1) =  -2.0 * eps_ox ;        
        jac(i - 1, i - 1 - 1) = eps_ox ; 
    }    

    int i = interface1_i;
    jac(i - 1, i + 1 - 1) = eps_ox ;
    jac(i - 1, i - 1) =  -2.0 * eps_ox ;        
    jac(i - 1, i - 1 - 1) = eps_ox ; 

    for (int i=si_begin_i; i<=si_end_i; ++i)
    {
        jac(i - 1, i + 1 - 1) = eps_si ;
        jac(i - 1, i - 1) =  -2.0 * eps_si ;        
        jac(i - 1, i - 1 - 1) = eps_si ; 
    }

    i = interface2_i;
    jac(i - 1, i + 1 - 1) = eps_ox ;
    jac(i - 1, i - 1) =  -2.0 * eps_ox ;        
    jac(i - 1, i - 1 - 1) = eps_ox ; 

    for (int i=(interface2_i + 1); i<=(N-1); ++i)    
    {
        jac(i - 1, i + 1 - 1) = eps_ox ;
        jac(i - 1, i - 1) =  -2.0 * eps_ox ;        
        jac(i - 1, i - 1 - 1) = eps_ox ; 
    }

    return jac;
}

int main() {

    //vec phi_0(N, arma::fill::ones);    
    vec phi_0(N, arma::fill::zeros);
    //vec phi_0(N, arma::fill::ones);
    double bc_left = 0.33374;
    double bc_right = 0.33374;
    phi_0(0) = bc_left;
    phi_0(N - 1) = bc_right;
    int num_iters = 30;
    //mat xs(num_iters, 3, arma::fill::zeros); // each row i represents the solution at iter i.
    //mat residuals(num_iters, 3, arma::fill::zeros); // each row i represents the residual at iter i.    
    vec phi_i = phi_0;
    for (int i=0; i<num_iters; i++)
    {
        vec residual = r(phi_i);
        mat jac = jacobian(phi_i);
        //jac.print("jac:");
        //printf("test");
        // xs.row(i) = x_i.t();         
        // residuals.row(i) = residual.t();     
        //residual.print("residual: ");   
        vec delta_phi_i = arma::solve(jac(span(1, N - 1 - 1), span(1, N - 1 - 1)), -residual(span(1, N - 1 - 1)));
        //phi_i(span(1, N - 1 - 1)) += delta_phi_i;                
        phi_i(span(1, N - 1 -1)) += delta_phi_i;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));        
    }

    phi_i.print("found solution (phi):");
        
    // Potential
    plot_args args;
    args.total_width = 6.0;
    args.N = N;    
    args.y_label = "Potential (V)";
    plot(phi_i, args);
}