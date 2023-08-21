#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include<cmath>
#include <fstream> // ofstream header

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int N = 101;
//int n_int = 1e10;
//double n_int = 1e16;
double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
// double total_width = 6.0;    
// double t_ox = 0.5;
bool use_normalizer = false;
double thermal = k_B * T / q;

double left_part_width = 1e-7;
double total_width = left_part_width*2;
double deltaX = (total_width) / (N-1); // in meter  
double coeff = deltaX*deltaX*q;

double dop_left = 1e23; // in m^3, n-type
double dop_right = 1e23; // p-type
int interface1_i = round(left_part_width/deltaX) + 1;
vec one_vector(2*N, fill::ones);

double B(double x)
{
    double result = 0.0;
    if (abs(x) < 0.0252)    
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0));    
    else if (abs(x) < 0.15)
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0 * (1 - pow(x, 2.0)/40 * (1 - 0.02525225525252525*pow(x, 2.0)))));
    else
        result = x / (exp(x) - 1);
    return result;
}

double deriveB(double x)
{
    double result = 0.0;
    if (abs(x) < 0.0252)
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0));
    else if (abs(x) < 0.15)
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0 * (1 - pow(x, 2.0)/30 * (1 - 0.0315656565656565656565*pow(x, 2.0)))));
    else
        result = 1.0/(exp(x)-1) - B(x)*(1.0 / (exp(x) - 1) + 1);
    return result;
}

// residual(phi): the size of r(phi) is N.
void r_and_jacobian(vec &r, mat &jac, vec &phi_n_p, double bias)
{
    r.fill(0.0);        
    jac.fill(0.0);
    int offset = N;    

    // from B.C. for phi
    double phi1 = thermal * log(dop_left/n_int);
    double phiN = - thermal * log(dop_right/n_int) + bias;
    r(1) = phi_n_p(1) - phi1;
    r(N) = phi_n_p(N) - phiN;
    // from B.C. for electron density
    r(offset+1) = phi_n_p(offset + 1) - dop_left;
    r(offset+N) = phi_n_p(offset + N) - 0;
    // from B.C. for hole density
    // double holeDensity1 = n_int*exp(-phi1/thermal);
    // double holeDensityN = n_int*exp(-phiN/thermal);
    r(offset+offset+1) = phi_n_p(offset+offset+1) - 0;
    r(offset+offset+N) = phi_n_p(offset+offset+N) - dop_right;

    jac(1, 1) = 1.0; 
    jac(N, N) = 1.0; 
    jac(offset+1, offset+1) = 1.0; 
    jac(offset+N, offset+N) = 1.0;       
    jac(offset+offset+1, offset+offset+1) = 1.0; 
    jac(offset+offset+N, offset+offset+N) = 1.0;     

    /*
    r = [r_poisson; r_elec_continuity, r_hole_continuity]
    Jacobian = r w.r.t. phi_n_p
    */     

    double eps_i_p_0_5 = eps_si;
    double eps_i_m_0_5 = eps_si;                        

    for (int i=(1+1); i<N; i++)
    {                
        // residual for poisson
        r(i) = eps_i_p_0_5*phi_n_p(i+1) -(eps_i_p_0_5 + eps_i_m_0_5)*phi_n_p(i) + eps_i_m_0_5*phi_n_p(i-1);            

        double n_i = phi_n_p(offset+i);
        double p_i = phi_n_p(offset+offset+i);
        if (i < interface1_i)
            r(i) += - coeff*((-dop_left) + n_i - p_i); 
        else if (i == interface1_i)
            r(i) += - coeff*(0.5*(-dop_left) + 0.5*( dop_right) + n_i - p_i);                 
        else if (i > interface1_i)
            r(i) += - coeff*(( dop_right) + n_i - p_i);             

        // poisson w.r.t phis
        jac(i, i+1) = eps_i_p_0_5;
        jac(i, i) = -(eps_i_p_0_5 + eps_i_m_0_5);
        jac(i, i-1) = eps_i_m_0_5;
        
        // poisson w.r.t n
        jac(i, i+offset) = - coeff;
        // poisson w.r.t p
        jac(i, i+offset+offset) = coeff;
    }

    for (int i=(N+1+1); i<2*N; i++)
    {                                        
        // residual for electron continuity        
        r(i) = phi_n_p(i+1) * B((phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) - 
            phi_n_p(i) * B((phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) -
            phi_n_p(i) * B((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) +
            phi_n_p(i-1) * B((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);        

        // continuity w.r.t. ns
        jac(i, i+1) = 
            B((phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal);
        jac(i, i) = 
            - B((phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) 
            - B((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal);
        jac(i, i-1) = 
            B((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        // continuity w.r.t. phis
        jac(i, i+1-offset) = 
            phi_n_p(i+1)*deriveB((phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) +
            phi_n_p(i)*deriveB((phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal);
        
        jac(i, i-offset) = -jac(i, i+1-offset) - 
            phi_n_p(i)*deriveB((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) -
            phi_n_p(i-1)*deriveB((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);
        
        jac(i, i-1-offset) = 
            phi_n_p(i)*deriveB((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) +
            phi_n_p(i-1)*deriveB((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        jac(i, i+1-offset) /= thermal;
        jac(i, i-offset) /= thermal;
        jac(i, i-1-offset) /= thermal;

        // residual for hole continuity        
        r(i+offset) = -phi_n_p(i+1+offset) * B(-(phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) + 
            phi_n_p(i+offset) * B(-(phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) +
            phi_n_p(i+offset) * B(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) -
            phi_n_p(i-1+offset) * B(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);        

        // hole continuity w.r.t. ps
        jac(i+offset, i+1+offset) = 
            - B(-(phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal);
        jac(i+offset, i+offset) = 
            + B(-(phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) 
            + B(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal);
        jac(i+offset, i-1+offset) = 
            - B(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        // hole continuity w.r.t. phis
        jac(i+offset, i+1-offset) = 
            phi_n_p(i+1+offset)*deriveB(-(phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) +
            phi_n_p(i+offset)*deriveB(-(phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal);
        
        jac(i+offset, i-offset) = -jac(i+offset, i+1-offset) - 
            phi_n_p(i+offset)*deriveB(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) -
            phi_n_p(i-1+offset)*deriveB(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);
        
        jac(i+offset, i-1-offset) = 
            phi_n_p(i+offset)*deriveB(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) +
            phi_n_p(i-1+offset)*deriveB(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        jac(i+offset, i+1-offset) /= thermal;
        jac(i+offset, i-offset) /= thermal;
        jac(i+offset, i-1-offset) /= thermal;        
    }                        
}

void save_mat(std::string file_name, mat &m)
{
    std::ofstream ofile(file_name);        
    for (int i=0; i<m.n_rows; ++i)
    {
        for (int j=0; j<m.n_cols; ++j)
        {   
            std::string str = fmt::format("{:.4f} ", m(i, j));      
            ofile << str;        
        }
        ofile << "\n";
    }    
    ofile.close();
}

void solve_for_phi_n(vec &phi_n_p_k, double bias)
{        
    vec r(3*N + 1, arma::fill::zeros);
    mat jac(3*N + 1, 3*N + 1, arma::fill::zeros);    

    int num_iters = 15;   

    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);

    for (int k=0; k<num_iters; k++)
    {        
        r_and_jacobian(r, jac, phi_n_p_k, bias);   
        
        //r.print("r:");        
        //r.save("r.txt", arma::raw_ascii);
        //jac.print("jac:");
        //jac.save("jac.txt", arma::raw_ascii);        
                    
        vec c_vector(3*N + 1, fill::zeros);        
        c_vector(span(1, N)) = thermal * one_vector(span(1, N));
        c_vector(span(N+1, 2*N)) = dop_left * one_vector(span(1, N));        
        c_vector(span(2*N+1, 3*N)) = dop_left * one_vector(span(1, N));        
        // c_vector(span(0+1, N-1-1)) = thermal * one_vector(span(0, N-1-2));
        // c_vector(span(N+1, 2*N-1-1)) = dop_left * one_vector(span(N, 2*N-1-2));        
        mat C = diagmat(c_vector(span(1, 3*N)));  
        //mat C = eye(2*N, 2*N);
        mat jac_scaled = jac(span(1, 3*N), span(1, 3*N)) * C;
        
        colvec r_vector_temp = arma::sum(abs(jac_scaled), 1);
        vec r_vector(3*N, fill::zeros);
        for (int p=0; p<3*N; p++)        
            r_vector(p) = 1 / (r_vector_temp(p) + 1e-10);                
        mat R = diagmat(r_vector);              
        //mat R_eye = eye(2*N, 2*N);
        //R = R_eye;
        jac_scaled = R * jac_scaled;
        vec r_scaled = R * r(span(1, 3*N));

        double cond_jac = arma::cond(jac_scaled);
        printf("[iter %d]   condition number of scaled jac: %f \n", k, cond_jac); 
        
        //jac_scaled.print("jac_scaled: ");
        //jac.print("jac:");
        //jac_scaled.save("jac_scaled.txt", arma::raw_ascii);        
        //save_mat("jac_scaled.txt", jac_scaled);
        vec delta_phi_n = arma::solve(jac_scaled, -r_scaled);        
        //vec delta_phi = arma::solve(jac(span(1, 2*N), span(1, 2*N)), -r(span(1, 2*N)));        
        phi_n_p_k(span(1, 3*N)) += C * delta_phi_n;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        double log_residual = log10(max(abs(r_scaled)));        
        //double log_delta = log10(max(abs(C * delta_phi)));                
        vec F = C * delta_phi_n;
        double log_delta = log10(max(abs(F(span(0, N-1)))));                
        //double log_delta = log10(max(abs(F(span(N, 2*N-1)))));                
        log_deltas[k] = log_delta;
        printf("[iter %d]   log_delta_x: %f   log_residual: %f \n", k, log_delta, log_residual);  

        // if (log_residual < - 10)
        //     break;
    }
    
    vec potential = phi_n_p_k(span(1, N));        
    vec eDensities = phi_n_p_k(span(N+1, 2*N));    
    vec holeDensities = phi_n_p_k(span(2*N+1, 3*N));    

    eDensities = eDensities / 1e6;
    std::string eDensities_file_name = fmt::format("Poisson_DD_eDensity_{:.2f}.csv", bias);
    eDensities.save(eDensities_file_name, csv_ascii);        
    holeDensities = holeDensities / 1e6;
    std::string holeDensities_file_name = fmt::format("Poisson_DD_holeDensity_{:.2f}.csv", bias);
    holeDensities.save(holeDensities_file_name, csv_ascii);

    double phi_bi = thermal * log(dop_left*dop_right/pow(n_int, 2));
    printf("phi_bi: %f \n", phi_bi);  

    bool do_plot = true;
    if (do_plot)
    {
        if (bias == 0 || bias > 0.9)
        {
            plot_args args;
            args.total_width = total_width;
            args.N = N;        
            args.y_label = "Potential (V)";    
            plot(potential, args);

            args.y_label = "eDensity (/cm^3)";  
            args.logscale_y = 10;
            plot(eDensities, args);

            args.y_label = "holeDensity (/cm^3)";  
            args.logscale_y = 10;
            plot(holeDensities, args);

            args.y_label = "log (delta phi)"; 
            args.logscale_y = -1;
            plot(log_deltas, args);    
        }
    }
}

void save_current_densities(vec &phi_n_p)
{
    vec phi = phi_n_p(span(1, N));
    vec n = phi_n_p(span(N+1, 2*N));
    vec current_densities(N+2, arma::fill::zeros);
    for (int i=2; i<=N-1; i++)
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
    current_densities.save("current_densities.txt", arma::raw_ascii);

}

void compute_I_V_curve()
{    
    vec phi_n_p_k(3*N + 1, arma::fill::zeros);  
    
    phi_n_p_k(span(1, N)) = thermal * log(dop_left/n_int) * one_vector(span(1, N));
    phi_n_p_k(span(N+1, 2*N)) = dop_left * one_vector(span(1, N));    

    bool load_initial_solution_from_NP = false;    

    int num_biases = 0;
    vec current_densities(num_biases+1, arma::fill::zeros);    
    for (int i=0; i<=(num_biases); ++i)
    {
        double bias = i * 0.05;
        printf("Applying Bias: %f V \n", bias);
        solve_for_phi_n(phi_n_p_k, bias);
        save_current_densities(phi_n_p_k);      
    }    
}

void save_B(std::string file_name)
{
    std::ofstream ofile(file_name);
    int N = 1000;        
    //vec a = arma::linspace(-0.2, 0.2, N);
    vec a = arma::linspace(-4, 4, N);
    for (int i=0; i<N; ++i)
    {        
        //double x = i*0.001;
        double x = a(i);
        double b_value = B(x);
        double derive_b = deriveB(x);
        std::string str = fmt::format("{}, {:.5f}, {}", x, b_value, derive_b);      
        ofile << str;               
        ofile << "\n";
    }    
    ofile.close();
}

int main() {    
    //compute_DD_n_from_NP_solution();
    compute_I_V_curve();
    //save_B("test.txt");
}