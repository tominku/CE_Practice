#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

int main() {

    double n_int = 1e10;
    double dop = 1e18;
    double T = 300; // (K)
    double phi = asinh(dop / (2*n_int)) * k_B * T / q;

    printf("phi: %f \n", phi);
}