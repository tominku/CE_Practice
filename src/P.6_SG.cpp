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

const int N = 301;
//int n_int = 1e10;
//double n_int = 1e16;
double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
// double total_width = 6.0;    
// double t_ox = 0.5;
bool use_normalizer = false;
double thermal = k_B * T / q;

double left_part_width = 1e-8;
double center_part_width = 4e-8;
double deltaX = (left_part_width*2 + center_part_width) / (N-1); // in meter  
double coeff = deltaX*deltaX*q;

double dop_left = 5e25; // in m^3
//double dop_center = 2e23; // in m^3
double dop_center = 2e23; // in m^3
double dop_right = dop_left;
int interface1_i = round(left_part_width/deltaX) + 1;
int interface2_i = round((left_part_width + center_part_width)/deltaX) + 1;
vec one_vector(2*N, fill::ones);
bool compute_only_n = true;

double B(double x)
{
    double result = 0.0;
    if (abs(x) < 0.0252)
    {
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0));
    }
    else if (abs(x) < 0.15)
    {
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0 * (1 - pow(x, 2.0)/40 * (1 - 0.02525225525252525*pow(x, 2.0)))));
    }
    else
    {     
        result = x / (exp(x) - 1);
    }
    return result;
}

double deriveB(double x)
{
    double result = 0.0;
    if (abs(x) < 0.0252)
    {
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0));
    }
    else if (abs(x) < 0.15)
    {
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0 * (1 - pow(x, 2.0)/30 * (1 - 0.0315656565656565656565*pow(x, 2.0)))));
    }
    else
    {        
        result = 1.0/(exp(x)-1) - B(x)*(1.0 / (exp(x) - 1) + 1);
    }
    return result;
}

// residual(phi): the size of r(phi) is N.
void r_and_jacobian(vec &r, mat &jac, vec &phi_n, double bias)
{
    r.fill(0.0);        
    jac.fill(0.0);
    int offset = N;

    r(1) = phi_n(1) - thermal * log(dop_left/n_int);
    r(N) = phi_n(N) - thermal * log(dop_right/n_int) - bias;
    r(offset+1) = phi_n(offset + 1) - dop_left;
    r(offset+N) = phi_n(offset + N) - dop_right;

    jac(1, 1) = 1.0; 
    jac(N, N) = 1.0; 
    jac(offset+1, offset+1) = 1.0; 
    jac(offset+N, offset+N) = 1.0;     

    /*
    r = [r_poisson; r_continuity]
    Jacobian = r w.r.t. phi_n
    */     

    double eps_i_p_0_5 = eps_si;
    double eps_i_m_0_5 = eps_si;                        

    for (int i=(1+1); i<N; i++)
    {                
        // residual for poisson
        r(i) = eps_i_p_0_5*phi_n(i+1) -(eps_i_p_0_5 + eps_i_m_0_5)*phi_n(i) + eps_i_m_0_5*phi_n(i-1);            

        double n_i = phi_n(offset+i);
        if (i < interface1_i)
            r(i) += - coeff*((-dop_left) + n_i); 
        else if (i == interface1_i)
            r(i) += - coeff*(0.5*(-dop_left) + 0.5*(-dop_center) + n_i); 
        else if (i > interface1_i & i < interface2_i)
            r(i) += - coeff*((-dop_center) + n_i); 
        else if (i == interface2_i)
            r(i) += - coeff*(0.5*(-dop_center) + 0.5*(-dop_right) + n_i); 
        else if (i > interface2_i)
            r(i) += - coeff*((-dop_right) + n_i);             

        // poisson w.r.t phis
        jac(i, i+1) = eps_i_p_0_5;
        jac(i, i) = -(eps_i_p_0_5 + eps_i_m_0_5);
        jac(i, i-1) = eps_i_m_0_5;
        
        // poisson w.r.t ns
        jac(i, i+offset) = - coeff;
    }

    for (int i=(N+1+1); i<2*N; i++)
    {                        
        // double n_avg1 = (phi_n(i) + phi_n(i+1)) / 2.0 ;
        // double n_avg2 = (phi_n(i) + phi_n(i-1)) / 2.0 ;    
        // double phi_diff1 = phi_n(i+1-offset) - phi_n(i-offset);
        // double phi_diff2 = phi_n(i-offset) - phi_n(i-1-offset);
        // double n_diff1 = phi_n(i+1) - phi_n(i);
        // double n_diff2 = phi_n(i) - phi_n(i-1);
        
        // residual for continuity        
        r(i) = phi_n(i+1) * B((phi_n(i+1-offset) - phi_n(i-offset)) / thermal) - 
            phi_n(i) * B((phi_n(i-offset) - phi_n(i+1-offset)) / thermal) -
            phi_n(i) * B((phi_n(i-offset) - phi_n(i-1-offset)) / thermal) +
            phi_n(i-1) * B((phi_n(i-1-offset) - phi_n(i-offset)) / thermal);        

        // continuity w.r.t. ns
        jac(i, i+1) = 
            B((phi_n(i+1-offset) - phi_n(i-offset)) / thermal);
        jac(i, i) = 
            - B((phi_n(i-offset) - phi_n(i+1-offset)) / thermal) 
            - B((phi_n(i-offset) - phi_n(i-1-offset)) / thermal);
        jac(i, i-1) = 
            B((phi_n(i-1-offset) - phi_n(i-offset)) / thermal);

        // continuity w.r.t. phis
        jac(i, i+1-offset) = 
            phi_n(i+1)*deriveB((phi_n(i+1-offset) - phi_n(i-offset)) / thermal) +
            phi_n(i)*deriveB((phi_n(i-offset) - phi_n(i+1-offset)) / thermal);
        
        jac(i, i-offset) = -jac(i, i+1-offset) - 
            phi_n(i)*deriveB((phi_n(i-offset) - phi_n(i-1-offset)) / thermal) -
            phi_n(i-1)*deriveB((phi_n(i-1-offset) - phi_n(i-offset)) / thermal);
        
        jac(i, i-1-offset) = 
            phi_n(i)*deriveB((phi_n(i-offset) - phi_n(i-1-offset)) / thermal) +
            phi_n(i-1)*deriveB((phi_n(i-1-offset) - phi_n(i-offset)) / thermal);

        jac(i, i+1-offset) /= thermal;
        jac(i, i-offset) /= thermal;
        jac(i, i-1-offset) /= thermal;
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

void solve_for_phi_n(vec &phi_n_k, double bias)
{        
    vec r(2*N + 1, arma::fill::zeros);
    mat jac(2*N + 1, 2*N + 1, arma::fill::zeros);    

    int num_iters = 15;   

    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);

    for (int k=0; k<num_iters; k++)
    {        
        r_and_jacobian(r, jac, phi_n_k, bias);   
        
        //r.print("r:");        
        //r.save("r.txt", arma::raw_ascii);
        //jac.print("jac:");
        //jac.save("jac.txt", arma::raw_ascii);        
        if (compute_only_n)
        {
            mat jac_part = jac(span(N+1, 2*N), span(N+1, 2*N));
            vec r_part = r(span(N+1, 2*N));
            vec delta_n = arma::solve(jac_part, -r_part);  
            phi_n_k(span(N+1, 2*N)) += delta_n;

            double log_residual = log10(max(abs(r_part)));                                            
            double log_delta = log10(max(abs(delta_n)));                
            log_deltas[k] = log_delta;
            printf("[iter %d]   log_delta_x: %f   log_residual: %f \n", k, log_delta, log_residual);              
        }
        else{            
            vec c_vector(2*N, fill::zeros);        
            c_vector(span(0, N-1)) = thermal * one_vector(span(0, N-1));
            c_vector(span(N, 2*N-1)) = dop_left * one_vector(span(N, 2*N-1));        
            // c_vector(span(0+1, N-1-1)) = thermal * one_vector(span(0, N-1-2));
            // c_vector(span(N+1, 2*N-1-1)) = dop_left * one_vector(span(N, 2*N-1-2));        
            mat C = diagmat(c_vector);  
            //mat C = eye(2*N, 2*N);
            mat jac_scaled = jac(span(1, 2*N), span(1, 2*N)) * C;
            
            colvec r_vector_temp = arma::sum(abs(jac_scaled), 1);
            vec r_vector(2*N, fill::zeros);
            for (int p=0; p<2*N; p++)        
                r_vector(p) = 1 / (r_vector_temp(p) + 1e-10);                
            mat R = diagmat(r_vector);              
            //mat R_eye = eye(2*N, 2*N);
            //R = R_eye;
            jac_scaled = R * jac_scaled;
            vec r_scaled = R * r(span(1, 2*N));

            double cond_jac = arma::cond(jac_scaled);
            printf("[iter %d]   condition number of scaled jac: %f \n", k, cond_jac); 
            
            //jac_scaled.print("jac_scaled: ");
            //jac.print("jac:");
            //jac_scaled.save("jac_scaled.txt", arma::raw_ascii);        
            //save_mat("jac_scaled.txt", jac_scaled);
            vec delta_phi_n = arma::solve(jac_scaled, -r_scaled);        
            //vec delta_phi = arma::solve(jac(span(1, 2*N), span(1, 2*N)), -r(span(1, 2*N)));        
            phi_n_k(span(1, 2*N)) += C * delta_phi_n;                
            
            //phi_i.print("phi_i");
            //jac.print("jac");
            //if (i % 1 == 0)
            //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
            //double log_residual = log10(max(abs(r_scaled)));        
            double log_residual = log10(max(abs(r(span(1, 2*N)))));                    
            //double log_delta = log10(max(abs(C * delta_phi)));                
            vec F = C * delta_phi_n;
            double log_delta = log10(max(abs(F(span(0, N-1)))));                
            //double log_delta = log10(max(abs(F(span(N, 2*N-1)))));                
            log_deltas[k] = log_delta;
            printf("[iter %d]   log_delta_x: %f   log_residual: %f \n", k, log_delta, log_residual);  

            // if (log_residual < - 10)
            //     break;
        }
    }
    
    vec eDensities = phi_n_k(span(N+1, 2*N));
    vec potential = phi_n_k(span(1, N));        

    eDensities = eDensities / 1e6;
    std::string n_file_name = fmt::format("DD_eDensity_{:.2f}.csv", 0.0);
    eDensities.save(n_file_name, csv_ascii);        

    bool do_plot = true;
    if (do_plot)
    {
        if (bias == 0 || bias > 0.9)
        {
            plot_args args;
            args.total_width = 600;
            args.N = N;        
            args.y_label = "Potential (V)";    
            plot(potential, args);

            args.y_label = "eDensity (/cm^3)";  
            args.logscale_y = 10;
            plot(eDensities, args);

            args.y_label = "log (delta phi)"; 
            args.logscale_y = -1;
            plot(log_deltas, args);    
        }
    }
}

void compute_DD_n_from_NP_solution()
{
    compute_only_n = true;
    vec phi_n_k(2*N + 1, arma::fill::zeros);  
    
    phi_n_k(span(1, N)) = thermal * log(dop_left/n_int) * one_vector(span(1, N));
    phi_n_k(span(N+1, 2*N)) = dop_left * one_vector(span(1, N));    

    bool load_initial_solution_from_NP = true;        
    
    double bias = 0;
    if (load_initial_solution_from_NP)
    {
        std::string fn_phi_from_NP = fmt::format("NP_phi_{:.2f}.csv", bias); 
        cout << fn_phi_from_NP << "\n";
        //printf("fn_phi_from_NP: %s \n", fn_phi_from_NP);     
        vec phi_from_NP(N, fill::zeros);
        phi_from_NP.load(fn_phi_from_NP);
                        
        phi_n_k(span(1, N)) = phi_from_NP(span(0, N-1));        
    }
    
    printf("Applying Bias: %f V \n", bias);
    solve_for_phi_n(phi_n_k, bias);    

    vec eDensities = phi_n_k(span(N+1, 2*N));
    vec potential = phi_n_k(span(1, N));        

    eDensities = eDensities / 1e6;
    std::string n_file_name = fmt::format("DD_eDensity_{:.2f}.csv", 0.0);
    eDensities.save(n_file_name, csv_ascii);        

    bool do_plot = true;
    if (do_plot)
    {
        plot_args args;
        args.total_width = 600;
        args.N = N;        
        args.y_label = "Potential (V)";    
        plot(potential, args);

        args.y_label = "eDensity (/cm^3)";  
        args.logscale_y = 10;
        plot(eDensities, args);
    }            
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
        double J_SG = n(i+1)*B((phi(i+1) - phi(i)) / thermal) - n(i)*B((phi(i) - phi(i+1)) / thermal);
        //double J = q * mu * (((n(j+1) + n(j)) / 2.0) * ((phi(j+1) - phi(j)) / deltaX) - thermal*(n(j+1) - n(j))/deltaX);
        double J = J_term1 + J_term2;
        J *= 1e-8;
        current_densities(i) = J;
        printf("Result Current Density J: %f, term1: %f, term2: %f, J_SG: %f \n", J, J_term1, J_term2, J_SG);
    }
    current_densities.save("current_densities.txt", arma::raw_ascii);
}


void compute_I_V_curve()
{
    compute_only_n = false;
    vec phi_n_k(2*N + 1, arma::fill::zeros);  
    
    phi_n_k(span(1, N)) = thermal * log(dop_left/n_int) * one_vector(span(1, N));
    phi_n_k(span(N+1, 2*N)) = dop_left * one_vector(span(1, N));    

    bool load_initial_solution_from_NP = false;    

    int num_biases = 0;
    vec current_densities(num_biases+1, arma::fill::zeros);    
    for (int i=0; i<=(num_biases); ++i)
    {
        double bias = i * 0.05;
        printf("Applying Bias: %f V \n", bias);
        solve_for_phi_n(phi_n_k, bias);
        save_current_densities(phi_n_k);
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