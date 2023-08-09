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
//int n_int = 1e10;
//double n_int = 1e16;
double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
// double total_width = 6.0;    
// double t_ox = 0.5;
double t_si = 5;
int interface1_i = 6;
int interface2_i = 56;
bool use_normalizer = false;
double thermal = k_B * T / q;
double coeff = deltaX*deltaX*q;

int dop_left = 0;
int dop_center = 0;
int dop_right = 0;

// residual(phi): the size of r(phi) is N.
std::pair<vec, mat> r_and_jacobian(vec phi_n)
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

    std::pair<vec, mat> result(r, jac);
    return result;
}


std::pair<vec, vec> solve_for_phi_n()
{        
    int num_iters = 20;   
    vec phi_n_k(2*N, arma::fill::zeros);    
    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);
    for (int k=0; k<num_iters; k++)
    {        
        std::pair<vec, vec> result = r_and_jacobian(phi_n_k);        
        vec r = result.first;
        mat jac = result.second;        
        
        vec delta_phi = arma::solve(jac, -r);        
        phi_n_k += delta_phi;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        double log_residual = log10(max(abs(r)));        
        double log_delta = log10(max(abs(delta_phi)));        
        log_residuals[k] = log_residual;
        log_deltas[k] = log_delta;
        printf("[iter %d]   log detal_x: %f   log residual: %f\n", k, log_delta, log_residual);  
        
        if (log_delta < - 10)
            break;
    }
}

int main() {    
    solve_for_phi_n();
}