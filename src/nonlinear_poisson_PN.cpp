#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int N = 101;
double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
//double n_int = 1.0*1e16;
double T = 300;    
double thermal = k_B * T / q;

double left_part_width = 5e-7;
double total_width = left_part_width*2;
double deltaX = total_width / (N-1); // in meter  
double coeff = deltaX*deltaX*q / eps_0;

// double dop_left = 5e25; // in m^3
// double dop_center = 2e23; // in m^3
double dop_left = 1e23; // in m^3
double dop_right = -1e23;

int interface_i = round(left_part_width/deltaX) + 1;


double compute_eq_phi(double doping_density)
{
    double phi = 0;
    if (doping_density > 0)
        phi = thermal * log(doping_density/n_int);
    else            
        phi = - thermal * log(abs(doping_density)/n_int);
    
    return phi;        
}

void r_and_jacobian(vec &r, mat &jac, vec &phi, double boundary_potential)
{
    r.fill(0.0);        
    jac.fill(0.0);    

    // r(1) = phi(1) - boundary_potential;
    // r(N) = phi(N) - boundary_potential;    
    
    r(1) = phi(1) - compute_eq_phi(dop_left);
    r(N) = phi(N) - compute_eq_phi(dop_right);

    jac(1, 1) = 1.0; 
    jac(N, N) = 1.0;             

    double eps_i_p_0_5 = 0.0;
    double eps_i_m_0_5 = 0.0;                        
    double ion_term = 0.0;

    for (int i=(1+1); i<N; i++)
    {                               
        double r_term_due_to_n_p = coeff*n_int*( - exp(phi(i)/thermal) + exp(-phi(i)/thermal) );            
        double jac_term_due_to_n_p = coeff*n_int*(1.0/thermal)*( - exp(phi(i)/thermal) - exp(-phi(i)/thermal) );
        
        if (i < interface_i)
        {                        
            eps_i_m_0_5 = eps_si_rel;
            eps_i_p_0_5 = eps_si_rel;
            ion_term = dop_left;                                           
        }
        else if (i == interface_i)
        {            
            eps_i_m_0_5 = eps_si_rel;
            eps_i_p_0_5 = eps_si_rel;
            ion_term = 0.5*(dop_left) + 0.5*(dop_right);                                  
        }
        else if (i > interface_i)
        {
            eps_i_m_0_5 = eps_si_rel;
            eps_i_p_0_5 = eps_si_rel;
            ion_term = dop_right;                         
        }
        
        r(i) = r_term_due_to_n_p;                    
        r(i) += eps_i_p_0_5*phi(i+1) -(eps_i_p_0_5 + eps_i_m_0_5)*phi(i) + eps_i_m_0_5*phi(i-1);
        r(i) += coeff*ion_term;            

        jac(i, i) = jac_term_due_to_n_p;                                      
        jac(i, i) += -(eps_i_p_0_5 + eps_i_m_0_5);
        jac(i, i+1) = eps_i_p_0_5;        
        jac(i, i-1) = eps_i_m_0_5;                                 
    }      
}

vec solve_phi(double boundary_potential, vec &phi_0)
{                
    int num_iters = 30;    
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
        double cond_jac = arma::cond(jac);
        printf("[iter %d]   log detal_x: %f   log residual: %f jac_cond: %f \n", k, log_delta, log_residual, cond_jac);  
        
        // if (log_delta < - 10)
        //     break;
    }
        
    return phi_k;
    //plot(potentials, args);
    
    // if (plot_error)
    //     plot(log_deltas, args);
        //plot(log_residuals, args);

    //phi_i.print("found solution (phi):");    
    // vec n(N, arma::fill::zeros);
    // n(span(interface_i-1, interface2_i-1)) = n_int * exp(q * phi_i(span(interface_i-1, interface2_i-1)) / (k_B * T));
    // std::pair<vec, vec> result(phi_i, n);
    // return result;
}

void save_current_densities(vec &phi_n)
{
    vec phi = phi_n(span(1, N));
    vec n = phi_n(span(N+1, 2*N));
    vec current_densities(N+2, arma::fill::zeros);
    for (int i=2; i<=N-2; i++)
    {            
        double mu = 1417;
        double J_term1 = -q * mu * ((n(i+1) + n(i)) / 2.0) * ((phi(i+1) - phi(i)) / deltaX);
        double J_term2 = q * mu * thermal*(n(i+1) - n(i))/deltaX;        
        //double J = q * mu * (((n(j+1) + n(j)) / 2.0) * ((phi(j+1) - phi(j)) / deltaX) - thermal*(n(j+1) - n(j))/deltaX);
        double J = J_term1 + J_term2;
        J *= 1e-8;
        current_densities(i) = J;
        printf("Result Current Density J: %f, term1: %f, term2: %f \n", J, J_term1, J_term2);
    }
    //current_densities.save("current_densities.txt", arma::raw_ascii);
}


int main() {    

    double start_potential = 0;    

    vec one_vector(N+1, arma::fill::ones);
    vec phi_0(N+1, arma::fill::zeros);
    phi_0(span(1, interface_i)) = compute_eq_phi(dop_left) * one_vector(span(1, interface_i));    
    phi_0(span(interface_i+1, N)) = compute_eq_phi(dop_right) * one_vector(span(interface_i+1, N));
    printf("phi_left: %f, phi_right: %f", compute_eq_phi(dop_left), compute_eq_phi(dop_right));
    //for (int i=0; i<10; i++)
    {        
        int i = 0;
        vec phi = solve_phi(start_potential + (0.1*i), phi_0); 
        phi_0 = phi;   
        
        std::string log = fmt::format("BD {:.2f} V \n", start_potential + (0.1*i));            
        cout << log;        

        plot_args args;
        args.total_width = 6.0;
        args.N = N;    
        vec n(N+1, arma::fill::zeros);
        n(span(1, N)) = n_int * exp(phi(span(1, N)) / thermal);
        n /= 1e6;        
        vec eDensity = n(span(1, N));        

        std::string n_file_name = fmt::format("NP_eDensity_{:.2f}.csv", (0.1*i));
        eDensity.save(n_file_name, csv_ascii);        

        vec h(N+1, arma::fill::zeros);
        h(span(1, N)) = n_int * exp(- phi(span(1, N)) / thermal);
        h /= 1e6;        
        vec holeDensity = h(span(1, N));        

        std::string h_file_name = fmt::format("NP_holeDensity_{:.2f}.csv", (0.1*i));
        holeDensity.save(h_file_name, csv_ascii);        
        
        vec phi_for_plot = phi(span(1, N));
        std::string phi_file_name = fmt::format("NP_phi_{:.2f}.csv", (0.1*i));
        phi_for_plot.save(phi_file_name, csv_ascii);
        
        args.y_label = "eDensity (cm^3)";
        args.logscale_y = 10;        
        plot(eDensity, args);
        args.y_label = "holeDensity (cm^3)";
        args.logscale_y = 10;
        plot(holeDensity, args);
        args.y_label = "Potential (V)";
        args.logscale_y = -1;
        plot(phi_for_plot, args);
        //plot(phi_for_plot, args);

        vec phi_n(2*N+1, arma::fill::zeros);
        phi_n(span(1, N)) = phi(span(1, N));
        phi_n(span(N+1, 2*N)) = eDensity;        

        save_current_densities(phi_n);
    }
}
