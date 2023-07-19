#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

int main() {

    int N = 51;
    double total_width = 5.0;    

    // matrix construction
    pairlist width_eps_list;    
    width_eps_list.push_back(std::pair<double, double>(2.5, 11.7));
    width_eps_list.push_back(std::pair<double, double>(2.5, 3.9));
    
    std::pair<double, double> end_potentials(0.0, 1.0);

    std::pair<mat, vec> A_b = construct_A_b_poisson(total_width, N, width_eps_list, end_potentials);   

    mat A = A_b.first;
    vec b = A_b.second;

    //A.print("A:");
    plot_args args;
    args.total_width = total_width;
    args.N = N;    
    args.y_label = "Potential (V)";
    vec sol_vec = arma::solve(A, b);      
    plot(sol_vec, args);
        
}