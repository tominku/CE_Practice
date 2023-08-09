#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int N = 61;
double deltaX = 0.1e-9; // in meter  
double dop = 1e18 * 1e6; // in meter    
//int n_int = 1e10;
//double n_int = 1e16;
double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
// double total_width = 6.0;    
// double t_ox = 0.5;
double t_si = 5;
int interface1_i = 6;
int interface2_i = 56;
int si_begin_i = interface1_i + 1;
int si_end_i = interface2_i - 1;
//bool include_nonlinear_terms = true;
bool include_nonlinear_terms = true;
bool use_normalizer = false;
double thermal = k_B * T / q;
double coeff = deltaX*deltaX*q;
double start_potential = 0.33374;

int dop_left = 0;
int dop_center = 0;
int dop_right = 0;

// residual(phi): the size of r(phi) is N.
vec r_and_jacobian(vec phi_n, double boundary_voltage)
{   
    vec r(2*N + 1, arma::fill::zeros);
    mat jac(2*N + 1, 2*N + 1, arma::fill::zeros);    
    int offset = N;

    r(1) = phi_n(1) - 0.0;
    r(N) = phi_n(N) - 0.0;
    r(offset+1) = phi_n(offset + 1) - 0.0;
    r(offset+N) = phi_n(offset + N) - 0.0;

    jac(1, 1) = 1.0; 
    jac(N, N) = 1.0; 
    jac(offset+1, offset+1) = 1.0; 
    jac(offset+N, offset+N) = 1.0;     

    /*
    r = [r_poisson; r_continuity]
    Jacobian = r w.r.t. phi_n
    */     

    for (int i=(1+1); i<N; i++)
    {        
        double eps_i_p_0_5 = eps_si;
        double eps_i_m_0_5 = eps_si;                        
        // residual for poisson
        r(i) = eps_i_p_0_5*phi_n(i+1) -(eps_i_p_0_5 + eps_i_m_0_5)*phi_n(i) + eps_i_m_0_5*phi_n(i-1);
        
        double coeff = deltaX*deltaX*q;
        double n_i = phi_n(offset+i);
        if (i < interface1_i)
            r(i) += - coeff*(dop_left + n_i); 
        else if (i == interface1_i)
            r(i) += - coeff*(0.5*dop_left + 0.5*dop_center + n_i); 
        else if (i > interface1_i & i < interface2_i)
            r(i) += - coeff*(dop_center + n_i); 
        else if (i == interface2_i)
            r(i) += - coeff*(0.5*dop_center + 0.5*dop_right + n_i); 
        else if (i > interface2_i)
            r(i) += - coeff*(dop_right + n_i);             

        // poisson w.r.t phis
        jac(i, i+1) = eps_i_p_0_5;
        jac(i, i) = -(eps_i_p_0_5 + eps_i_m_0_5);
        jac(i, i-1) = eps_i_m_0_5;
        
        // poisson w.r.t ns
        jac(i, i+offset) = - coeff;
    }

    for (int i=(N+1+1); i<2*N; i++)
    {                        
        double n_avg1 = (phi_n(i) + phi_n(i+1)) / 2.0 ;
        double n_avg2 = (phi_n(i) + phi_n(i-1)) / 2.0 ;    
        double phi_diff1 = phi_n(i+1-offset) - phi_n(i-offset);
        double phi_diff2 = phi_n(i-offset) - phi_n(i-1-offset);
        double n_diff1 = phi_n(i+1) - phi_n(i);
        double n_diff2 = phi_n(i) - phi_n(i-1);
        
        // residual for continuity
        r(i) = -n_avg1*phi_diff1 + thermal*n_diff1 + n_avg2*phi_diff2 - thermal*n_diff2;        
        
        // continuity w.r.t. phis
        jac(i, i+1-offset) = -n_avg1;
        jac(i, i-offset) = n_avg1 + n_avg2;
        jac(i, i-1-offset) = -n_avg2;

        // continuity w.r.t. ns
        jac(i, i+1) = -0.5*phi_diff1 + thermal;
        jac(i, i) = -0.5*phi_diff1 + 0.5*phi_diff2 - 2*thermal;
        jac(i, i-1) = 0.5*phi_diff2 + thermal;        
    }                

    return r;
}


std::pair<vec, vec> solve_phi(vec phi_0, double boundary_potential, bool plot_error)
{    
    //vec phi_0(N, arma::fill::ones);
    //vec phi_0(N, arma::fill::randn);
    //double boundary_voltage = 0.33374;
    //phi_0 *= boundary_potential;
    double bc_left = boundary_potential;
    double bc_right = boundary_potential;
    // phi_0(0) = bc_left;
    // phi_0(N - 1) = bc_right;
    int num_iters = 20;
    //mat xs(num_iters, 3, arma::fill::zeros); // each row i represents the solution at iter i.
    //mat residuals(num_iters, 3, arma::fill::zeros); // each row i represents the residual at iter i.    
    vec phi_i = phi_0;
    printf("boundary voltage: %f V \n", boundary_potential);
    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);
    for (int i=0; i<num_iters; i++)
    {
        vec residual = r(phi_i, boundary_potential);
        mat jac = jacobian(phi_i);
        //jac.print("jac:");
        //printf("test");
        // xs.row(i) = x_i.t();         
        // residuals.row(i) = residual.t();     
        //residual.print("residual: ");   
        vec delta_phi_i = arma::solve(jac, -residual);
        //phi_i(span(1, N - 1 - 1)) += delta_phi_i;                
        phi_i += delta_phi_i;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        double log_residual = log10(max(abs(residual)));        
        double log_delta = log10(max(abs(delta_phi_i)));        
        log_residuals[i] = log_residual;
        log_deltas[i] = log_delta;
        printf("[iter %d]   log detal_x: %f   log residual: %f\n", i, log_delta, log_residual);  
        
        if (log_delta < - 10)
            break;
    }
}

int main() {    

}