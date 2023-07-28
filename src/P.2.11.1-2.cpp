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
double n_int = 1e16;
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

// residual(phi): the size of r(phi) is N.
vec r(vec phi, double boundary_voltage)
{   
    vec r_k(N, arma::fill::zeros);
    // boundary
    r_k(0) = phi(0) - boundary_voltage;
    r_k(N-1) = phi(N-1) - boundary_voltage;

    // oxide
    r_k(span(1, interface1_i-1-1)) = (eps_ox) * (-2*phi(span(1, interface1_i-1-1)) + phi(span(0, interface1_i-1-1-1)) + phi(span(2, interface1_i-1)));
    
    // interface 1
    r_k(interface1_i-1) = -(eps_ox)*phi(interface1_i-1) - (eps_si)*phi(interface1_i-1) + 
        (eps_ox)*phi(interface1_i-1-1) + (eps_si)*phi(interface1_i-1+1) - 0.5 * coeff*dop;
    if (include_nonlinear_terms)
        r_k(interface1_i-1) -= 0.5 * coeff*n_int*exp(phi(interface1_i-1)/thermal);

    // silicon
    r_k(span(si_begin_i-1, si_end_i-1)) = (eps_si) * ( -2*phi(span(si_begin_i-1, si_end_i-1)) +
        phi(span(si_begin_i-2, si_end_i-2)) + phi(span(si_begin_i, si_end_i)) );    
    r_k(span(si_begin_i-1, si_end_i-1)) -= coeff*dop;    
    if (include_nonlinear_terms)
        r_k(span(si_begin_i-1, si_end_i-1)) -= coeff*n_int*exp(phi(span(si_begin_i-1, si_end_i-1))/thermal);

    // interface 2
    r_k(interface2_i-1) = -(eps_si)*phi(interface2_i-1) - (eps_ox)*phi(interface2_i-1) + 
        (eps_si)*phi(interface2_i-1-1) + (eps_ox)*phi(interface2_i-1+1) - 0.5 * coeff*dop;
    if (include_nonlinear_terms)
        r_k(interface2_i-1) -= 0.5 * coeff*n_int*exp(phi(interface2_i-1)/thermal);

    // oxide
    r_k(span(interface2_i-1+1, N-1-1)) = (eps_ox) * (-2*phi(span(interface2_i-1+1, N-1-1)) + 
        phi(span(interface2_i-1, N-1-1-1)) + phi(span(interface2_i-1+1+1, N-1-1+1)));
    
    if (use_normalizer)
        r_k(span(1, N-1-1)) /= eps_0;
    //r_k(span(0, N-1)) /= eps_0;
        

    return r_k;
}

// the jacobian matrix size is N by N
mat jacobian(vec phi)
{
    mat jac(N, N, arma::fill::zeros);    
    
    // boundary
    jac(0, 0) = 1.0;
    jac(N-1, N-1) = 1.0;

    //ox
    for (int i=2; i<=(interface1_i - 1); ++i)    
    {
        jac(i - 1, i + 1 - 1) = eps_ox ;
        jac(i - 1, i - 1) =  -2.0 * eps_ox ;        
        jac(i - 1, i - 1 - 1) = eps_ox ; 
    }    

    // interface 1
    int i = interface1_i;
    jac(i - 1, i + 1 - 1) = eps_si ;
    jac(i - 1, i - 1) =  -eps_ox - eps_si ;        
    if (include_nonlinear_terms)        
        jac(i - 1, i - 1) -= 0.5*coeff*n_int*(1.0/thermal)*exp(phi(i-1)/thermal);
    jac(i - 1, i - 1 - 1) = eps_ox ; 

    // silicon
    for (int i=si_begin_i; i<=si_end_i; ++i)
    {
        jac(i - 1, i + 1 - 1) = eps_si ;
        jac(i - 1, i - 1) =  -2.0 * eps_si ;
        if (include_nonlinear_terms)        
            jac(i - 1, i - 1) -= coeff*n_int*(1.0/thermal)*exp(phi(i-1)/thermal);
        jac(i - 1, i - 1 - 1) = eps_si ; 
    }

    // interface 2
    i = interface2_i;
    jac(i - 1, i + 1 - 1) = eps_ox ;
    jac(i - 1, i - 1) =  -eps_ox - eps_si ; 
    if (include_nonlinear_terms)        
        jac(i - 1, i - 1) -= 0.5*coeff*n_int*(1.0/thermal)*exp(phi(i-1)/thermal);
    jac(i - 1, i - 1 - 1) = eps_si ; 

    // oxide
    for (int i=(interface2_i + 1); i<=(N-1); ++i)    
    {
        jac(i - 1, i + 1 - 1) = eps_ox ;
        jac(i - 1, i - 1) =  -2.0 * eps_ox ;        
        jac(i - 1, i - 1 - 1) = eps_ox ; 
    }

    if (use_normalizer)
        jac(span(1, N-1-1), span(1, N-1-1)) /= eps_0;

    return jac;
}

double integrate_n_over_si(vec n)
{    
    double integrated = 0.5*n(interface1_i-1)*deltaX + 0.5*n(interface2_i-1)*deltaX;
    integrated += sum(n(span(si_begin_i-1, si_end_i-1)) * deltaX);
    integrated /= 1.0e4; // m^-2 => cm^-2
    printf("interface 1 value: %f \n", n(interface1_i-1));
    printf("interface 2 value: %f", n(interface2_i-1));
    printf("interface 1 index: %d", interface1_i-1);
    printf("interface 2 index: %d", interface2_i-1);
    return integrated;
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
        log_residuals[i] = log_residual;
        printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), log_residual);  
        // if (log_residual < - 15)
        //     break;
    }

    plot_args args;
    //args.total_width = 6.0;
    args.N = num_iters;    
    args.y_label = "log(max residual)";    
    if (plot_error)
        plot(log_residuals, args);

    //phi_i.print("found solution (phi):");    
    vec n(N, arma::fill::zeros);
    n(span(interface1_i-1, interface2_i-1)) = n_int * exp(q * phi_i(span(interface1_i-1, interface2_i-1)) / (k_B * T));
    std::pair<vec, vec> result(phi_i, n);
    return result;
}

// void compare_with_linear_solution()
// {
//     include_nonlinear_terms = false;
//     vec phi_wo_n_p = solve_phi();

//     include_nonlinear_terms = true;
//     vec phi_w_n_p = solve_phi();

//     mat phis(N, 2, arma::fill::zeros);
//     phis(arma::span::all, 0) = phi_wo_n_p;
//     phis(arma::span::all, 1) = phi_w_n_p;
        
//     // Potential
//     plot_args args;
//     args.total_width = 6.0;
//     args.N = N;    
//     args.y_label = "Potential (V)";
//     args.labels.push_back("without nonlinear");
//     args.labels.push_back("with nonlinear");
//     plot(phis, args);
// }


int main() {    
    //printf("nl term: %f", (q/(k_B*T)));
    
    include_nonlinear_terms = true;
    double start_voltage = 0.33374;
    double experiment_gap = 0.01;
    int num_experiments = 101;
    vec boundary_voltages(num_experiments);
    for (int i=0; i<num_experiments; ++i)
    {
        boundary_voltages(i) = start_voltage + experiment_gap * i;
    }

    plot_args args;
    args.total_width = 6.0;
    args.N = N;    
    args.y_label = "Potential (V)";    
    mat phis(N, num_experiments, arma::fill::zeros);
    char buf[100];  
    vec integrated_ns(num_experiments);
    
    plot_args args2;        
    args2.x_label = "Gate Voltage (V)";
    args2.y_label = "Integrated n (cm^{-2})";            

    vec phi_0(N, arma::fill::zeros);
    
    for (int i=0; i<num_experiments; ++i)
    {
        double boundary_voltage = boundary_voltages(i);
        bool plot_error = false;
        if (i % 10 == 0)
            plot_error = true;
        std::pair<vec, vec> result = solve_phi(phi_0, boundary_voltage, plot_error);    
        vec phi = result.first;
        vec n = result.second;
        phi_0 = phi;

        if (i==100)
        {
            plot(phi, args);
            args.y_label = "Electron Density (/cm^{3})";
            vec n_cm3 = n / 1e6;
            plot(n_cm3, args);            
        }
        //phis(arma::span::all, i) = phi;
        double integrated_n_over_si = integrate_n_over_si(n);            
        integrated_ns[i] = integrated_n_over_si;            
        
        if (i==100)
        {            
            //vec gate_voltages = arma::linspace(0, boundary_voltages(num_experiments-1) - start_voltage, num_experiments);
            //plot(gate_voltages, integrated_ns, args2);
        }
        //phis(arma::span::all, i) = n;
        phis(arma::span::all, i) = phi;
        sprintf(buf, "BC %f", boundary_voltage);
        //args.labels.push_back(std::string::);
        std::string label_text = fmt::format("V_g: {:.1f} V", boundary_voltage - start_voltage);        
        args.labels.push_back(label_text);        
    }  
    args.y_label = "Potential (V)";
    plot(phis, args);      
    
    vec gate_voltages = arma::linspace(0, boundary_voltages(num_experiments-1) - start_voltage, num_experiments);
    //args2.logscale_y = 10;
    
    plot(gate_voltages, integrated_ns, args2);    
   
    // Potential
    // plot_args args;
    // args.total_width = 6.0;
    // args.N = N;    
    // args.y_label = "Potential (V)";    
    // plot(phi, args);
}