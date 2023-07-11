#include <armadillo> 
#include <iostream> 
using namespace arma; 
using namespace std; 

int main() {

    mat A(2, 2, arma::fill::zeros);

    A = { { 2, -1 },
          { -1, 2 } };
            
    A.print("A:");

    cx_vec eigvals;
    cx_mat eigvecs;

    eig_gen(eigvals, eigvecs, A);

    eigvals.print("eigen values:");
    eigvecs.print("eigen vectors:");
}