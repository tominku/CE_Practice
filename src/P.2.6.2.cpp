#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

int main() {

    int N = 5;
    double total_width = 5.0;
    int last_index = N - 1; 
    double e1 = 11.7;
    double e2 = 3.9;
    mat A(N, N, arma::fill::zeros);

    // matrix construction
    A(0, 0) = 1;
    A(last_index, last_index) = 1;

    // matrix construction
    pairlist width_eps_list;    
    width_eps_list.push_back(std::pair<double, double>(2.5, 0.01));
    width_eps_list.push_back(std::pair<double, double>(2.5, 0.02));
    
    std::pair<mat, vec> A_b = construct_A_b_poisson(total_width, N, width_eps_list);   

    A(1, span(0, 2)) = {e1, -2*e1, e1};
    A(2, span(1, 3)) = {e1, -e2 - e1, e2};
    A(3, span(2, 4)) = {e2, -2*e2, e2};

    vec b(N, arma::fill::zeros);
    b(last_index) = 1;

    A.print("A:");

    vec sol_vec = arma::solve(A, b);  
    plot(N, sol_vec);
}