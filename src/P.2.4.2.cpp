#include <armadillo> 
#include <iostream> 
using namespace arma; 
using namespace std; 

int main() {

    int N = 5;
    int mat_thickness = N-2;
    int last_index = mat_thickness - 1; 
    mat A(mat_thickness, mat_thickness, arma::fill::zeros);

    // matrix construction
    A(0, arma::span(0, 1)) = {-2, 1};
    A(last_index, arma::span(last_index-1, last_index)) = {1, -2};

    // matrix construction
    for (int i=1; i<last_index; i++)
    {
        A(i, arma::span(i-1, i+1)) = {1, -2, 1};
    }

    //A.print("A:");

    // eigen analysis
    cx_vec eigvals;
    cx_mat eigvecs;
    eig_gen(eigvals, eigvecs, A);
    eigvals.print("eigen values:");
    eigvecs.print("eigen vectors:");
    //eigvecs.col(0)
    cx_vec smallest_eigvec = eigvecs.col(last_index);
    smallest_eigvec.print("Smallest Eigen Vector");

    //A.print("A:");
}