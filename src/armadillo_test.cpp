#include <armadillo> 
#include <iostream> 
#define FMT_HEADER_ONLY
#include <fmt/format.h>

using namespace arma;

int main() 
{    
    vec c_vector(3, fill::zeros);        
    c_vector(0) = 1.0;                
    c_vector(1) = 2.0;                
    c_vector(2) = 3.0;                
    
    sp_mat C(3, 3);
    C.zeros();
    C.print(); 

    for (int m=0; m<3; ++m)
        C(m, m) = -9;    
    sp_mat D = arma::sum(abs(C), 1);
    
    D.print(); 

    // sp_mat D(3, 3);
    // sp_mat E = C*D;
}