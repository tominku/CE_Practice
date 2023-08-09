#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include<cmath>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

int main() {    

    vec v(5, fill::randu);
    mat D = diagmat(v);   
    D.print("D:");


}