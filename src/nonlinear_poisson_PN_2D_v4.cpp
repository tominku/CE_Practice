#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int Nx = 201;
const int Ny = 41;
const int N = Nx * Ny;
//double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double n_int = 1.0*1e16;
double T = 300;    
double thermal = k_B * T / q;

double left_part_width = 2e-7;
double total_width = left_part_width*2;
double deltaX = total_width / (Nx-1); // in meter  
double total_height = 0.8e-7;
double deltaY = (total_height) / (Ny-1); // in meter  
double coeff = deltaX*deltaY*q;
const double DelYDelX = deltaY / deltaX;
const double DelXDelY = deltaX / deltaY;

#define ijTok(i, j) (Nx*(j-1) + i)
//#define eps_i_p(i, j) ((i+0.5) < Ny ? eps_si : 0)
//#define eps_i_m(i, j) ((i-0.5) > 1 ? eps_si : 0)
#define eps_i_p(i, j) eps_si
#define eps_i_m(i, j) eps_si
#define eps_j_p(i, j) eps_si
#define eps_j_m(i, j) eps_si
//#define phi_at(i, j, phi_name, phi_center_name) ((i) > 1 && (i) < Ny ? phi_name(ijTok(i, j)) : phi_center_name)
#define phi_at(i, j, phi_name, phi_center_name) ((j >= 1 && j <= Ny) ? phi_name(ijTok(i, j)) : phi_center_name)

const string subject_name = "PN_2D_NP";

// double dop_left = 5e25; // in m^3
// double dop_center = 2e23; // in m^3
double dop_left = 5e23; // in m^3
//double dop_right = -2e24;
double dop_right = -2e23;

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

void r_and_jacobian(vec &r, sp_mat &jac, vec &phi, double boundary_potential)
{
    r.fill(0.0);        
    jac.zeros();       
        
    // set boundary condition
    int i = 0;
    for (int j=1; j<=Ny; ++j)
    {
        // left boundary
        i = 1;      
        int k = ijTok(i, j);
        r(k) = phi(k) - compute_eq_phi(dop_left);
        jac(k, k) = 1.0; 
        // right boundary      
        i = Nx;            
        k = ijTok(i, j);
        r(k) = phi(k) - compute_eq_phi(dop_right);
        jac(k, k) = 1.0; 
    }        
    
    double ion_term = 0.0;

    for (int j=1; j<=Ny; j++)    
    {
        for (int i=(1+1); i<Nx; i++)        
        {   
            int k = ijTok(i, j);
            double phi_ij = phi(k);
            double r_term_due_to_n_p = coeff*n_int*( exp(phi_ij/thermal) - exp(-phi_ij/thermal) );            
            double jac_term_due_to_n_p = coeff*n_int*(1.0/thermal)*( exp(phi_ij/thermal) + exp(-phi_ij/thermal) );
            
            if (i < interface_i)                                                    
                ion_term = -dop_left;                                                       
            else if (i == interface_i)            
                ion_term = 0.5*(-dop_left) + 0.5*(-dop_right);                                              
            else if (i > interface_i)            
                ion_term = -dop_right;                                                             

            double phi_ipj = phi_at(i+1, j, phi, phi_ij);                           
            double phi_imj = phi_at(i-1, j, phi, phi_ij);                           
            double phi_ijp = phi_at(i, j+1, phi, phi_ij);                           
            double phi_ijm = phi_at(i, j-1, phi, phi_ij);                           

            double eff_ipj = 1.0;
            double eff_imj = 1.0;
            double eff_ijp = 1.0;
            double eff_ijm = 1.0;
            double total_eff = 1.0;
            if (j == 1)
            {
                eff_ipj = 0.5;
                eff_imj = 0.5;
                eff_ijm = 0.0;
                total_eff = 0.5;
            }            
            else if(j == Ny)
            {
                eff_ipj = 0.5;
                eff_imj = 0.5;
                eff_ijp = 0.0;
                total_eff = 0.5;
            }

            r(k) = total_eff*r_term_due_to_n_p;     
            r(k) += - eff_ipj*DelYDelX*eps_i_p(i, j)*(phi_ipj - phi_ij) +
                    eff_imj*DelYDelX*eps_i_m(i, j)*(phi_ij - phi_imj) -
                    eff_ijp*DelXDelY*eps_j_p(i, j)*(phi_ijp - phi_ij) +
                    eff_ijm*DelXDelY*eps_j_m(i, j)*(phi_ij - phi_ijm);
            // r(k) += - eff_ipj*DelYDelX*eps_i_p(i, j)*(phi_ipj - phi_ij) +
            //         eff_imj*DelYDelX*eps_i_m(i, j)*(phi_ij - phi_imj);                    
            r(k) += total_eff*coeff*ion_term;            

            jac(k, k) = total_eff*jac_term_due_to_n_p;                                      
            jac(k, k) += eff_ipj*eps_i_p(i, j)*DelYDelX + eff_imj*eps_i_m(i, j)*DelYDelX +
                eff_ijp*eps_j_p(i, j)*DelXDelY + eff_ijm*eps_j_m(i, j)*DelXDelY;
            //jac(k, k) += eff_ipj*eps_i_p(i, j)*DelYDelX + eff_imj*eps_i_m(i, j)*DelYDelX;                

            
            jac(k, ijTok(i+1, j)) = - eff_ipj*eps_i_p(i, j) * DelYDelX;                        
            jac(k, ijTok(i-1, j)) = - eff_imj*eps_i_m(i, j) * DelYDelX;   
            if (eff_ijp > 0)
                jac(k, ijTok(i, j+1)) = - eff_ijp*eps_j_p(i, j) * DelXDelY;
            if (eff_ijm > 0)                                
                jac(k, ijTok(i, j-1)) = - eff_ijm*eps_j_m(i, j) * DelXDelY;                                             
        }      
    }
}

vec solve_phi(double boundary_potential, vec &phi_0)
{                
    int num_iters = 500;    
    printf("boundary voltage: %f \n", boundary_potential);
    auto start = high_resolution_clock::now();
    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);
        
    vec r(N + 1, arma::fill::zeros);
    sp_mat jac(N + 1, N + 1);
    jac = jac.zeros();    
    
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
        sp_mat jac_part = jac(span(1, N), span(1, N));  

        superlu_opts opts;
        opts.allow_ugly  = true;
        opts.equilibrate = true;
        //opts.refine = superlu_opts::REF_DOUBLE;

        vec delta_phi_k = arma::spsolve(jac_part, -r(span(1, N)));
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
        //double cond_jac = arma::cond(jac);
        printf("[iter %d]   log detal_x: %f   log residual: %f \n", k, log_delta, log_residual);  
        
        if (log_delta < - 10)
            break;
    }

    auto stop = high_resolution_clock::now();
    // Subtract stop and start timepoints and
    // cast it to required unit. Predefined units
    // are nanoseconds, microseconds, milliseconds,
    // seconds, minutes, hours. Use duration_cast()
    // function.
    auto duration = duration_cast<milliseconds>(stop - start);        
    cout << "duration: " << duration.count() << endl;

    std::string convergence_file_name = fmt::format("{}_conv_{:.2f}.csv", subject_name, boundary_potential);
    log_deltas.save(convergence_file_name, csv_ascii);            
        
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


void fill_initial(vec &phi, string method)
{        
    double phi_1 = compute_eq_phi(dop_left);
    double phi_Nx = compute_eq_phi(dop_right);
    for (int j=1; j<=Ny; j++)
    {
        for (int i=1; i<=Nx; i++)
        { 
            int k = ijTok(i, j);            
            if (i==1)  
                phi(k) = phi_1;
            else if (i==Nx)                            
                phi(k) = phi_Nx;
            else
            {
                if (method.compare("uniform") == 0)
                {
                    if (i <= interface_i)                
                        phi(k) = phi_1;
                    else
                        phi(k) = phi_Nx;                            
                }
                else if (method.compare("linear") == 0)
                {
                    phi(k) = phi_1 + (phi_Nx - phi_1) * ((i-1)/(Nx-1));
                }
            }
        }
     }      
}


int main() {    

    double start_potential = 0;    

    vec one_vector(N+1, arma::fill::ones);
    vec phi_0(N+1, arma::fill::zeros);
    fill_initial(phi_0, "uniform");
    //fill_initial(phi_0, "linear");
    //for (int i=0; i<10; i++)
    {        
        int i = 0;
        vec phi = solve_phi(start_potential + (0.1*i), phi_0); 
        phi_0 = phi;   
        
        std::string log = fmt::format("BD {:.2f} V \n", start_potential + (0.1*i));            
        cout << log;        

        plot_args args;
        args.total_width = total_width;
        args.N = N;    
        vec n(N+1, arma::fill::zeros);
        n(span(1, N)) = n_int * exp(phi(span(1, N)) / thermal);
        n /= 1e6;        
        vec eDensity = n(span(1, N));        
        std::string n_file_name = fmt::format("{}_eDensity_{:.2f}.csv", subject_name, (0.1*i));
        eDensity.save(n_file_name, csv_ascii);        

        vec h(N+1, arma::fill::zeros);
        h(span(1, N)) = n_int * exp(- phi(span(1, N)) / thermal);
        h /= 1e6;        
        vec holeDensity = h(span(1, N));        

        std::string h_file_name = fmt::format("{}_holeDensity_{:.2f}.csv", subject_name, (0.1*i));
        holeDensity.save(h_file_name, csv_ascii);        
        
        vec phi_for_plot = phi(span(1, N));
        std::string phi_file_name = fmt::format("{}_phi_{:.2f}.csv", subject_name, (0.1*i));
        phi_for_plot.save(phi_file_name, csv_ascii);                        

        vec phi_n(2*N+1, arma::fill::zeros);
        phi_n(span(1, N)) = phi(span(1, N));
        phi_n(span(N+1, 2*N)) = eDensity;        

        //save_current_densities(phi_n);
    }
}
