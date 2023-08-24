#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int Nx = 101;
const int Ny = 31;
const int N = Nx * Ny;
//double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double n_int = 1.0*1e16;
double T = 300;    
double thermal = k_B * T / q;

double left_part_width = 2e-7;
double total_width = left_part_width*2;
double deltaX = total_width / (N-1); // in meter  
double total_height = 1e-7;
double deltaY = (total_height) / (N-1); // in meter  
double coeff = deltaX*deltaY*q;
const double DelYDelX = deltaY / deltaX;
const double DelXDelY = deltaX / deltaY;

#define ijTok(i, j) (Nx*(i-1) + j)
#define eps_i_p(i, j) ((i+0.5) < Ny ? eps_si : 0)
#define eps_i_m(i, j) ((i-0.5) > 1 ? eps_si : 0)
#define eps_j_p(i, j) eps_si
#define eps_j_m(i, j) eps_si
#define phi_at(i, j, phi_name, phi_center_name) ((i) > 1 && (i) < Ny ? phi_name(ijTok(i, j)) : phi_center_name)

// double dop_left = 5e25; // in m^3
// double dop_center = 2e23; // in m^3
double dop_left = 5e23; // in m^3
double dop_right = -2e23;

int interface_j = round(left_part_width/deltaX) + 1;


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
        
    // set boundary condition
    int j = 0;
    for (int i=1; i<=Ny; ++i)
    {
        // left boundary
        j = 1;      
        int k = ijTok(i, j);
        r(k) = phi(k) - compute_eq_phi(dop_left);
        jac(k, k) = 1.0; 
        // right boundary      
        j = Nx;            
        k = ijTok(i, j);
        r(k) = phi(k) - compute_eq_phi(dop_right);
        jac(k, k) = 1.0; 
    }        
    
    double ion_term = 0.0;

    for (int i=1; i<=Ny; i++)
    {
        for (int j=(1+1); j<Nx; j++)
        {   
            int k = ijTok(i, j);
            double phi_ij = phi(ijTok(i, j));
            double r_term_due_to_n_p = coeff*n_int*( exp(phi_ij/thermal) - exp(-phi_ij/thermal) );            
            double jac_term_due_to_n_p = coeff*n_int*(1.0/thermal)*( exp(phi(i)/thermal) + exp(-phi(i)/thermal) );
            
            if (j < interface_j)                                                    
                ion_term = -dop_left;                                                       
            else if (j == interface_j)            
                ion_term = 0.5*(-dop_left) + 0.5*(-dop_right);                                              
            else if (j > interface_j)            
                ion_term = -dop_right;                                     
            
            r(k) = r_term_due_to_n_p;     

            double phi_ipj = phi_at(i+1, j, phi, phi_ij);                           
            double phi_imj = phi_at(i-1, j, phi, phi_ij);                           
            double phi_ijp = phi_at(i, j+1, phi, phi_ij);                           
            double phi_ijm = phi_at(i, j-1, phi, phi_ij);                           

            r(k) += - DelYDelX*eps_i_p(i, j)*(phi_ipj - phi_ij) +
                    DelYDelX*eps_i_m(i, j)*(phi_ij - phi_imj) -
                    DelXDelY*eps_j_p(i, j)*(phi_ijp - phi_ij) +
                    DelXDelY*eps_j_m(i, j)*(phi_ij - phi_ijm);
            r(k) += coeff*ion_term;            

            jac(k, k) = jac_term_due_to_n_p;                                      
            jac(k, k) += eps_i_p(i, j)*DelYDelX + eps_i_m(i, j)*DelYDelX +
                eps_j_p(i, j)*DelXDelY + eps_j_m(i, j)*DelXDelY;

            if ((i+1) <= Ny)
                jac(k, ijTok(i+1, j)) = - eps_i_p(i, j) * DelYDelX;
            if ((i-1) >= 1)
                jac(k, ijTok(i-1, j)) = - eps_i_m(i, j) * DelYDelX;
            jac(k, ijTok(i, j+1)) = - eps_j_p(i, j) * DelXDelY;
            jac(k, ijTok(i, j-1)) = - eps_j_m(i, j) * DelXDelY;
        }      
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
    // n(span(interface_j-1, interface2_i-1)) = n_int * exp(q * phi_i(span(interface_j-1, interface2_i-1)) / (k_B * T));
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
    phi_0(span(1, interface_j)) = compute_eq_phi(dop_left) * one_vector(span(1, interface_j));    
    phi_0(span(interface_j+1, N)) = compute_eq_phi(dop_right) * one_vector(span(interface_j+1, N));
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

        std::string n_file_name = fmt::format("NP_PN_2D_eDensity_{:.2f}.csv", (0.1*i));
        eDensity.save(n_file_name, csv_ascii);        

        vec h(N+1, arma::fill::zeros);
        h(span(1, N)) = n_int * exp(- phi(span(1, N)) / thermal);
        h /= 1e6;        
        vec holeDensity = h(span(1, N));        

        std::string h_file_name = fmt::format("NP_PN_holeDensity_{:.2f}.csv", (0.1*i));
        holeDensity.save(h_file_name, csv_ascii);        
        
        vec phi_for_plot = phi(span(1, N));
        std::string phi_file_name = fmt::format("NP_PN_phi_{:.2f}.csv", (0.1*i));
        phi_for_plot.save(phi_file_name, csv_ascii);
                
        vec phi_n(2*N+1, arma::fill::zeros);
        phi_n(span(1, N)) = phi(span(1, N));
        phi_n(span(N+1, 2*N)) = eDensity;        

        //save_current_densities(phi_n);
    }
}
