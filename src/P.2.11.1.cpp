#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

const int N = 61;
double deltaX = 0.1e-9; // in meter  
double dop = 1e18 * 1e6; // in meter    
int n_int = 1e10 * 1e6;;
double T = 300;
int boundary_i_first = 6;
int boundary_i_second = 56;

// f(phi) or residual(phi)
vec f(vec phi)
{
    int start_i = 1;
    int last_i = N;    
    vec phi_i_minus_1 = phi(span(0, last_i - 1 - 1));
    vec phi_i_plus_1 = phi(span(1, last_i - 1));
    vec r_i = (eps_si/deltaX) * (phi_i_plus_1 - 2*phi + phi_i_minus_1);
    int idx1 = boundary_i_first + 1 - 1; 
    int idx2 = boundary_i_first - 1 - 1; 
    r_i(span(idx1, idx2)) += ( deltaX*q*dop - deltaX*q*n_int*exp(q*phi(span(idx1, idx2))/(k_B*T)) );
    return r_i;
}

mat jacobian(vec phi)
{
                
}

int main() {

    int N = 61;
    double total_width = 6.0;    
    double t_ox = 0.5;
    double t_si = 5;

    // matrix construction
    pairlist width_eps_list;        
    width_eps_list.push_back(std::pair<double, double>(t_ox, 3.9));
    width_eps_list.push_back(std::pair<double, double>(t_si, 11.7));
    width_eps_list.push_back(std::pair<double, double>(t_ox, 3.9));

    std::pair<double, double> end_potentials(0.33374, 0.33374);

    std::pair<mat, vec> A_b = construct_A_b_poisson(total_width, N, width_eps_list, end_potentials);   

    mat A = A_b.first;
    vec b = A_b.second;
    
    double deltaX = 0.1 * 1e-9; // in meter    
    double N_acceptor = 1e18 * 1e6; // in meter     

    for (int i=1; i<=N; ++i)
    {        
        double c = deltaX * deltaX * q * N_acceptor / eps_0;
        if (i > 1 && i < 6)
            b(i-1) = 0.0; 
        else if (i == 6) // ox-si boundary
            b(i-1) = 0.5 * c; 
        else if (i > 6 and i < 56)
            b(i-1) = c;
        else if (i == 56) // si-ox boundary
            b(i-1) = 0.5 * c;
        else if (i > 56 && i < N)
            b(i-1) = 0.0;
    }

    vec sol_vec = arma::solve(A, b);      

    // Potential
    plot_args args;
    args.total_width = total_width;
    args.N = N;    
    args.y_label = "Potential (V)";
    plot(sol_vec, args);

    // Electron Density
    double T = 300;
    double n_int = 1e10;
    double segment_width = total_width / (N - 1); // in nanometer
    int ox_si_boundary_k = int(t_ox / segment_width);
    int si_ox_boundary_k = int((t_ox + t_si) / segment_width);
    vec ed(N, arma::fill::zeros);
    ed(span(ox_si_boundary_k, si_ox_boundary_k)) = n_int * exp(q * sol_vec(span(ox_si_boundary_k, si_ox_boundary_k)) / (k_B * T));
    args.y_label = "Electron Density (/cm^3)";
    plot(ed, args);
}