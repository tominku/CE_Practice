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
double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
double thermal = k_B * T / q;

double left_part_width = 0.5 * 1e-9;
double center_part_width = 5 * 1e-9;
double total_width = left_part_width*2 + center_part_width;
double deltaX = total_width / (N-1); // in meter  
double coeff = deltaX*deltaX*q;

double dop_left = 0; // in m^3
double dop_center = 1e18 * 1e6; // in m^3
double dop_right = dop_left;
int interface1_i = round(left_part_width/deltaX) + 1;
int interface2_i = round((left_part_width + center_part_width)/deltaX) + 1;

void r_and_jacobian(vec &r, mat &jac, vec &phi, double boundary_potential)
{
    r.fill(0.0);        
    jac.fill(0.0);    

    r(1) = phi(1) - boundary_potential;
    r(N) = phi(N) - boundary_potential;    

    jac(1, 1) = 1.0; 
    jac(N, N) = 1.0;             

    double eps_i_p_0_5 = 0.0;
    double eps_i_m_0_5 = 0.0;                        
    double dop_term = 0.0;

    for (int i=(1+1); i<N; i++)
    {                               
        double r_term_due_to_n_p = coeff*n_int*( - exp(phi(i)/thermal) + exp(-phi(i)/thermal) );            
        double jac_term_due_to_n_p = coeff*n_int*(1.0/thermal)*( - exp(phi(i)/thermal) - exp(-phi(i)/thermal) );
        
        if (i < interface1_i)
        {                        
            eps_i_m_0_5 = eps_ox;
            eps_i_p_0_5 = eps_ox;
            dop_term = dop_left;                                           
        }
        else if (i == interface1_i)
        {            
            eps_i_m_0_5 = eps_ox;
            eps_i_p_0_5 = eps_si;
            dop_term = 0.5*(dop_left) + 0.5*(dop_center);                      
            r(i) += 0.5*r_term_due_to_n_p;            
            jac(i, i) = 0.5*jac_term_due_to_n_p;            
        }
        else if (i > interface1_i & i < interface2_i)
        {
            eps_i_m_0_5 = eps_si;
            eps_i_p_0_5 = eps_si;
            dop_term = dop_center;               
            r(i) += r_term_due_to_n_p;
            jac(i, i) = jac_term_due_to_n_p;            
        }
        else if (i == interface2_i)
        {
            eps_i_m_0_5 = eps_si;
            eps_i_p_0_5 = eps_ox;
            dop_term = 0.5*(dop_center) + 0.5*(dop_right);            
            r(i) += 0.5*r_term_due_to_n_p;
            jac(i, i) = 0.5*jac_term_due_to_n_p;            
        }
        else if (i > interface2_i)
        {
            eps_i_m_0_5 = eps_ox;
            eps_i_p_0_5 = eps_ox;
            dop_term = dop_right;                         
        }
                        
        r(i) += eps_i_p_0_5*phi(i+1) -(eps_i_p_0_5 + eps_i_m_0_5)*phi(i) + eps_i_m_0_5*phi(i-1);
        r(i) -= coeff*dop_term;            

        jac(i, i) += -(eps_i_p_0_5 + eps_i_m_0_5);
        jac(i, i+1) = eps_i_p_0_5;        
        jac(i, i-1) = eps_i_m_0_5;                                 
    }      
}

vec solve_phi(double boundary_potential, vec &phi_0)
{                
    int num_iters = 20;    
    printf("boundary voltage: %f \n", boundary_potential);
    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);
        
    vec r(N + 1, arma::fill::zeros);
    mat jac(N + 1, N + 1, arma::fill::zeros);    
    
    vec phi_k(N + 1, arma::fill::zeros);     
    phi_k = phi_0;

    for (int k=0; k<num_iters; k++)
    {        
        r_and_jacobian(r, jac, phi_k, boundary_potential);   
        //jac.print("jac:");
        //printf("test");
        // xs.row(i) = x_i.t();         
        // residuals.row(i) = residual.t();     
        //residual.print("residual: ");   
        vec delta_phi_k = arma::solve(jac(span(1, N), span(1, N)), -r(span(1, N)));
        //phi_i(span(1, N - 1 - 1)) += delta_phi_i;                
        phi_k(span(1, N)) += delta_phi_k;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        double log_residual = log10(max(abs(r)));        
        double log_delta = log10(max(abs(delta_phi_k)));        
        log_residuals[k] = log_residual;
        log_deltas[k] = log_delta;
        printf("[iter %d]   log detal_x: %f   log residual: %f\n", k, log_delta, log_residual);  
        
        if (log_delta < - 10)
            break;
    }
        
    return phi_k;
    //plot(potentials, args);
    
    // if (plot_error)
    //     plot(log_deltas, args);
        //plot(log_residuals, args);

    //phi_i.print("found solution (phi):");    
    // vec n(N, arma::fill::zeros);
    // n(span(interface1_i-1, interface2_i-1)) = n_int * exp(q * phi_i(span(interface1_i-1, interface2_i-1)) / (k_B * T));
    // std::pair<vec, vec> result(phi_i, n);
    // return result;
}


int main() {    

    double start_potential = 0.33374;    

    vec phi_0(N+1, arma::fill::zeros);
    for (int i=0; i<10; i++)
    {        
        vec phi = solve_phi(start_potential + (0.1*i), phi_0); 
        phi_0 = phi;   
        
        std::string log = fmt::format("BD {:.2f} V \n", start_potential + (0.1*i));            
        cout << log;
        // plot_args args;
        // //args.total_width = 6.0;
        // args.N = N;
        // //args.y_label = "log(max residual)";        
        // args.y_label = fmt::format("log(delta) V_g: {:.2f} V", boundary_potential - start_potential);
        // vec potentials = phi_k(span(1, N));            
        // args.total_width = 6.0;
        // args.N = N;    
        // args.y_label = "Potential (V)";

        plot_args args;
        args.total_width = 6.0;
        args.N = N;    
        vec n(N+1, arma::fill::zeros);
        n(span(interface1_i, interface2_i)) = n_int * exp(phi(span(interface1_i, interface2_i)) / thermal);
        n /= 1e6;
        args.y_label = "eDensity (cm^3)";
        vec eDensity = n(span(1, N));

        std::string n_file_name = fmt::format("eDensity_{:.2f}.csv", (0.1*i));
        eDensity.save(n_file_name, csv_ascii);

        //..plot(eDensity, args);
    }
}