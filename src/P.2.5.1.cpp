#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

int main() {

    int N = 5;
    double total_width = 5.0;    

    // matrix construction
    pairlist width_eps_list;    
    width_eps_list.push_back(std::pair<double, double>(total_width, 1.0));    
    
    std::pair<double, double> end_potentials(0.0, 1.0);

    std::pair<mat, vec> A_b = construct_A_b_poisson(total_width, N, width_eps_list, end_potentials);   

    mat A = A_b.first;
    vec b = A_b.second;

    //A.print("A:");

    vec sol_vec = arma::solve(A, b);  
    plot(N, sol_vec);
}