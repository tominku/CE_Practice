#include <armadillo> 
#include <iostream> 
using namespace arma; 
using namespace std; 

int main() {

    mat A(2, 2, arma::fill::zeros);

    A = { { 2, -1 },
          { -1, 2 } };
            
    A.print("A:");

    cx_vec eigval;
    cx_mat eigvec;

    eig_gen(eigval, eigvec, A);

    eigval.print("eigen values:");
    eigvec.print("eigen vectors:");
}