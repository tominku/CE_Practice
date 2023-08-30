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

const int N = 201;
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
double total_width = left_part_width*2 + center_part_width;
double deltaX = (total_width) / (N-1); // in meter  
double coeff = deltaX*deltaX*q / eps_0;

double dop_left = 5e25; // in m^3
double dop_center = 2e23; // in m^3
double dop_avg = (dop_left + dop_center) / 2.0;
double dop_right = dop_left;
int interface1_i = round(left_part_width/deltaX) + 1;
int interface2_i = round((left_part_width + center_part_width)/deltaX) + 1;
vec one_vector(3*N, fill::ones);

double B(double x);
double dB(double x);

// residual(phi): the size of r(phi) is N.
void r_and_jacobian(vec &r, sp_mat &jac, vec &phi_n_p, double bias)
{
    r.fill(0.0);        
    jac.zeros();
    int offset = N;    

    // from B.C. for phi
    double phi1 = thermal * log(dop_left/n_int);
    double phiN = thermal * log(dop_right/n_int) + bias;
    r(1) = phi_n_p(1) - phi1;
    r(N) = phi_n_p(N) - phiN;
    // from B.C. for electron density
    r(offset+1) = phi_n_p(offset + 1) - dop_left;
    r(offset+N) = phi_n_p(offset + N) - dop_right;
    // from B.C. for hole density
    double holeDensity1 = n_int*exp(-phi1/thermal);
    double holeDensityN = n_int*exp(-phiN/thermal);
    r(offset+offset+1) = phi_n_p(offset+offset+1) - holeDensity1;
    r(offset+offset+N) = phi_n_p(offset+offset+N) - holeDensityN;

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

    double eps_i_p_0_5 = eps_si_rel;
    double eps_i_m_0_5 = eps_si_rel;                        

    for (int i=(1+1); i<N; i++)
    {                
        // residual for poisson
        r(i) = eps_i_p_0_5*phi_n_p(i+1) -(eps_i_p_0_5 + eps_i_m_0_5)*phi_n_p(i) + eps_i_m_0_5*phi_n_p(i-1);            

        double n_i = phi_n_p(offset+i);
        double p_i = phi_n_p(offset+offset+i);
        if (i < interface1_i)
            r(i) += - coeff*((-dop_left) + n_i - p_i); 
        else if (i == interface1_i)
            r(i) += - coeff*(0.5*(-dop_left) + 0.5*(-dop_center) + n_i - p_i); 
        else if (i > interface1_i & i < interface2_i)
            r(i) += - coeff*((-dop_center) + n_i - p_i); 
        else if (i == interface2_i)
            r(i) += - coeff*(0.5*(-dop_center) + 0.5*(-dop_right) + n_i - p_i); 
        else if (i > interface2_i)
            r(i) += - coeff*((-dop_right) + n_i - p_i);             

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
        
        r(i) /= dop_avg;

        // continuity w.r.t. ns
        jac(i, i+1) = 
            B((phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal);
        jac(i, i) = 
            - B((phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) 
            - B((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal);
        jac(i, i-1) = 
            B((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        jac(i, i+1) /= dop_avg;             
        jac(i, i) /= dop_avg;             
        jac(i, i-1) /= dop_avg;             

        // continuity w.r.t. phis
        double a = jac(i, i+1-offset) = 
            phi_n_p(i+1)*dB((phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) +
            phi_n_p(i)*dB((phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal);
        
        // jac(i, i-offset) = -jac(i, i+1-offset) - 
        //     phi_n_p(i)*dB((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) -
        //     phi_n_p(i-1)*dB((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);
        
        double b = jac(i, i-1-offset) = 
            phi_n_p(i)*dB((phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) +
            phi_n_p(i-1)*dB((phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        jac(i, i-offset) = - a - b;

        jac(i, i+1-offset) /= thermal*dop_avg;
        jac(i, i-offset) /= thermal*dop_avg;
        jac(i, i-1-offset) /= thermal*dop_avg;

        // residual for hole continuity        
        r(i+offset) = -phi_n_p(i+1+offset) * B(-(phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) + 
            phi_n_p(i+offset) * B(-(phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) +
            phi_n_p(i+offset) * B(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) -
            phi_n_p(i-1+offset) * B(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);        

        r(i+offset) /= dop_avg;

        // hole continuity w.r.t. ps
        jac(i+offset, i+1+offset) = 
            - B(-(phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal);
        jac(i+offset, i+offset) = 
            + B(-(phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal) 
            + B(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal);
        jac(i+offset, i-1+offset) = 
            - B(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        jac(i+offset, i+1+offset) /= dop_avg;         
        jac(i+offset, i+offset) /= dop_avg;            
        jac(i+offset, i-1+offset) /= dop_avg;            

        // hole continuity w.r.t. phis
        a = jac(i+offset, i+1-offset) = 
            phi_n_p(i+1+offset)*dB(-(phi_n_p(i+1-offset) - phi_n_p(i-offset)) / thermal) +
            phi_n_p(i+offset)*dB(-(phi_n_p(i-offset) - phi_n_p(i+1-offset)) / thermal);
        
        b = jac(i+offset, i-1-offset) = 
            phi_n_p(i+offset)*dB(-(phi_n_p(i-offset) - phi_n_p(i-1-offset)) / thermal) +
            phi_n_p(i-1+offset)*dB(-(phi_n_p(i-1-offset) - phi_n_p(i-offset)) / thermal);

        jac(i+offset, i-offset) = - a - b;
                

        jac(i+offset, i+1-offset) /= thermal*dop_avg;
        jac(i+offset, i-offset) /= thermal*dop_avg;
        jac(i+offset, i-1-offset) /= thermal*dop_avg;        
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

void solve_for_phi_n(vec &phi_n_p_k, double bias, sp_mat &C)
{        
    vec r(3*N + 1, arma::fill::zeros);
    sp_mat jac(3*N + 1, 3*N + 1);    
    jac.zeros();

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
                            
        // c_vector(span(0+1, N-1-1)) = thermal * one_vector(span(0, N-1-2));
        // c_vector(span(N+1, 2*N-1-1)) = dop_left * one_vector(span(N, 2*N-1-2));                
        //mat C = eye(2*N, 2*N);
        sp_mat jac_scaled = jac(span(1, 3*N), span(1, 3*N)) * C;
        
        // r_vector_temp = arma::sum(abs(jac_scaled), 1);
        // vec r_vector(3*N, fill::zeros);
        // for (int p=0; p<3*N; p++)        
        //     r_vector(p) = 1 / (r_vector_temp(p) + 1e-10);                
        // mat R = diagmat(r_vector);              
        sp_mat r_vector_temp = arma::sum(abs(jac_scaled), 1);
        vec r_vector(3*N, fill::zeros);
        for (int p=0; p<3*N; p++)        
            r_vector(p) = 1 / (r_vector_temp(p) + 1e-10);                                
        sp_mat R(3*N, 3*N);            
        R.zeros();
        for (int m=0; m<3*N; m++)
            R(m, m) = r_vector(m);        
        //mat R_eye = eye(2*N, 2*N);
        //R = R_eye;
        jac_scaled = R * jac_scaled;
        vec r_scaled = R * r(span(1, 3*N));

        //double cond_jac = arma::cond(jac_scaled);
        //printf("[iter %d]   condition number of scaled jac: %f \n", k, cond_jac); 
        
        //jac_scaled.print("jac_scaled: ");
        //jac.print("jac:");
        //jac_scaled.save("jac_scaled.txt", arma::raw_ascii);        
        //save_mat("jac_scaled.txt", jac_scaled);
        vec delta_phi_n = arma::spsolve(jac_scaled, -r_scaled);        
        //vec delta_phi = arma::solve(jac(span(1, 2*N), span(1, 2*N)), -r(span(1, 2*N)));        
        phi_n_p_k(span(1, 3*N)) += C * delta_phi_n;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        //double log_residual = log10(max(abs(r_scaled)));        
        double log_residual = log10(max(abs(r(span(1, 3*N)))));        
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


void compute_I_V_curve()
{    
    vec phi_n_p_k(3*N + 1, arma::fill::zeros);  
    
    phi_n_p_k(span(1, N)) = thermal * log(dop_left/n_int) * one_vector(span(1, N));
    phi_n_p_k(span(N+1, 2*N)) = dop_left * one_vector(span(1, N));    

    bool load_initial_solution_from_NP = false;    

    sp_mat C(3*N, 3*N);        
    C.zeros();
    for (int m=0; m<N; m++)
        C(m, m) = thermal;
    for (int m=N; m<2*N; m++)
        C(m, m) = abs(dop_left);        
    for (int m=2*N; m<3*N; m++)
        C(m, m) = abs(dop_left);    

    int num_biases = 0;
    vec current_densities(num_biases+1, arma::fill::zeros);    
    for (int i=0; i<=(num_biases); ++i)
    {
        double bias = i * 0.05;
        printf("Applying Bias: %f V \n", bias);
        solve_for_phi_n(phi_n_p_k, bias, C);

        int j = N-2;
        //int j = 34;
        vec phi = phi_n_p_k(span(1, N));
        vec n = phi_n_p_k(span(N+1, 2*N)) * 1e-8;
        double mu = 1417;
        double J = q * mu * (((n(j+1) + n(j)) / 2.0) * ((phi(j+1) - phi(j)) / deltaX) - thermal*(n(j+1) - n(j))/deltaX);
        current_densities(i) = J;
        printf("Result Current Density J: %f \n", J);
    }
    current_densities.save("current_densities_SG_with_hc.txt", arma::raw_ascii);
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
        double derive_b = dB(x);
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

double B(double x)
{
    double result = 0.0;
    
    if (abs(x) < 0.0252)   
        // Bern_P1 = ( 1.0-(x1)/2.0+(x1)^2/12.0*(1.0-(x1)^2/60.0*(1.0-(x1)^2/42.0)) ) ;        
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0));    
    else if (abs(x) < 0.15)
        // Bern_P1 = ( 1.0-(x1)/2.0+(x1)^2/12.0*(1.0-(x1)^2/60.0*(1.0-(x1)^2/42.0*(1-(x1)^2/40*(1-0.02525252525252525252525*(x1)^2)))));
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0 * (1 - pow(x, 2.0)/40 * (1 - 0.02525252525252525252525*pow(x, 2.0)))));
    else
        result = x / (exp(x) - 1);
    return result;
}

double dB(double x)
{
    double result = 0.0;
    if (abs(x) < 0.0252)
        // Deri_Bern_P1_phi1 = (-0.5 + (x1)/6.0*(1.0-(x1)^2/30.0*(1.0-(x1)^2/28.0)) )/thermal;
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0));
    else if (abs(x) < 0.15)
        // Deri_Bern_P1_phi1 = (-0.5 + (x1)/6.0*(1.0-(x1)^2/30.0*(1.0-(x1)^2/28.0*(1-(x1)^2/30*(1-0.03156565656565656565657*(x1)^2)))))/thermal;
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0 * (1 - pow(x, 2.0)/30 * (1 - 0.03156565656565656565657*pow(x, 2.0)))));
    else
        // Deri_Bern_P1_phi1=(1/(exp(x1)-1)-Bern_P1*(1/(exp(x1)-1)+1))/thermal;
        result = 1.0/(exp(x)-1) - B(x)*(1.0 / (exp(x) - 1) + 1);
    return result;
}