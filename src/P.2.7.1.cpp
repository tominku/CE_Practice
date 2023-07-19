#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

int main() {
 
    int N = 61;
    double total_width = 6.0;    

    // matrix construction
    pairlist width_eps_list;    
    width_eps_list.push_back(std::pair<double, double>(0.5, 3.9));
    width_eps_list.push_back(std::pair<double, double>(5, 11.7));
    width_eps_list.push_back(std::pair<double, double>(0.5, 3.9));    

    std::pair<double, double> end_potentials(0.0, 0.0);

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

    // plot
    plot_args args;
    args.total_width = total_width;
    args.N = N;      
    args.x_label = "Position (nm)";
    args.y_label = "Potential (V)";   
    vec sol_vec = arma::solve(A, b);  
    plot(sol_vec, args);
}