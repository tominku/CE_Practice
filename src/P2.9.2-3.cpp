#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

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

    for (int k=1; k<60; ++k)
    {
        int i = k + 1;
        double c = deltaX * deltaX * q * N_acceptor / eps_0;
        if (i < 6)
            b(i) = 0.0; 
        else if (i == 6)
            b(i) = 0.5 * c; 
        else if (i > 6 and i < 56)
            b(i) = c;
        else if (i == 56)
            b(i) = 0.5 * c;
        else if (i > 56)
            b(i) = 0.0;
    }
    b(b.n_elem - 1) = 0.33374;

    vec sol_vec = arma::solve(A, b);  

    plot_args args;
    args.total_width = total_width;
    args.N = N;    

    // Potential
    args.y_label = "Potential (V)";
    plot(sol_vec, args);

    // Electron Density
    double T = 300;
    double n_int = 1e10;
    double segment_width = total_width / (N - 1);
    int ox_si_boundary_k = int(t_ox / segment_width);
    int si_ox_boundary_k = int((t_ox + t_si) / segment_width);
    vec ed(N, arma::fill::zeros);
    ed(span(ox_si_boundary_k, si_ox_boundary_k)) = n_int * exp(q * sol_vec(span(ox_si_boundary_k, si_ox_boundary_k)) / (k_B * T));
    args.y_label = "Electron Density (/cm^3)";
    plot(ed, args);

    // Hole Density
    vec hd(N, arma::fill::zeros);
    hd(span(ox_si_boundary_k, si_ox_boundary_k)) = n_int * exp(- q * sol_vec(span(ox_si_boundary_k, si_ox_boundary_k)) / (k_B * T));
    args.y_label = "Hole Density (/cm^3)";
    plot(hd, args);
}