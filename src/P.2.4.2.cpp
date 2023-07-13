#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
using namespace arma; 
using namespace std; 
using namespace sciplot;
typedef std::vector<double> stdvec;

int main() {

    double a = 5.0;
    int N = 50; 
    mat A(N-2, N-2, arma::fill::zeros);

    // matrix construction
    A(0, arma::span(0, 1)) = {-2, 1};
    int last_index = N - 2 - 1;
    A(last_index, arma::span(last_index-1, last_index)) = {1, -2};

    // matrix construction
    for (int k=1; k < last_index; k++)
    {
        A(k, arma::span(k-1, k+1)) = {1, -2, 1};
    }

    //A.print("A:");

    // eigen analysis
    cx_vec cx_eigvals;
    cx_mat cx_eigvecs;
    eig_gen(cx_eigvals, cx_eigvecs, A);
    mat eigvecs = real(cx_eigvecs);
    vec eigvals = real(cx_eigvals);
    vec k_deltaX_squared = -eigvals;
    double deltaX = a / double(N -1);
    double deltaX_squared = deltaX * deltaX;
    vec k_squared = (1.0 / deltaX_squared) * k_deltaX_squared;
    vec k = sqrt(k_squared);
    k.print("k");
    //vec n = k * a / double(3.14);
    
    //n.print("k");
    //eigvals.print("eigen values:");
    //eigvecs.print("eigen vectors:");
    
    uvec indices = arma::sort_index(eigvals, "ascend"); // ascending order
    //indices.print("indices");
    int n = 3;
    int target_index = indices[last_index - (n - 1)];
    //int target_index = indices[2];
    //printf("target index: %d \n", target_index);

    vec target_eigvec = eigvecs.col(target_index);
    double target_eigval = eigvals(target_index);
    //target_eigvec.print("Target Eigenvector");

    //stdvec solution_vec(mat_thickness)
    
    vec solution_vec(N, arma::fill::zeros);
    solution_vec(span(1, N-2)) = target_eigvec;
    stdvec std_solution_vec = conv_to<stdvec>::from(solution_vec);
    //sciplot::Vec solution_vec_sciplot(solution_vec)
    
    //A.print("A:");

    // Create a Plot object
    Plot2D plot;

    plot.xlabel("Position(nm)");
    plot.ylabel("Wavefunction");

    // Set the x and y ranges
    //plot.xrange(0.0, 5);
    //plot.yrange(-3, 3);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    plot.grid().show();

    Vec x = linspace(0.0, 5, N);
    plot.drawPoints(x, std_solution_vec).pointType(6);
    plot.drawCurve(x, std_solution_vec);
    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show();    
}