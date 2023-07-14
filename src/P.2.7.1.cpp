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

    // width_eps_list.push_back(std::pair<double, double>(1.5, 3.9));
    // width_eps_list.push_back(std::pair<double, double>(3, 11.7));
    // width_eps_list.push_back(std::pair<double, double>(1.5, 3.9));

    std::pair<double, double> end_potentials(0.0, 0.0);

    std::pair<mat, vec> A_b = construct_A_b_poisson(total_width, N, width_eps_list, end_potentials);   

    mat A = A_b.first;
    vec b = A_b.second;
    
    double deltaX = 0.1 * 1e-9; // in meter
    double q = 1.602192e-19;
    double N_acceptor = 1e18 * 1e6; // 
    double eps_0 = 8.854 * 1e-12; // in meter

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

    // matrix scaling
    // double eps_0 = 8.854 * 1e-12; // in meter
    // A.rows(1, N-2) = A.rows(1, N-2) / eps_0;
    // b(span(1, N-2)) = b(span(1, N-2)) / eps_0;
    
    //A.print("A:");

    vec sol_vec = arma::solve(A, b);  
    plot(total_width, N, sol_vec);
}