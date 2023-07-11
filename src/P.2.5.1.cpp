#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
using namespace arma; 
using namespace std; 
using namespace sciplot;
typedef std::vector<double> stdvec;

int main() {

    int N = 50;
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

    A.print("A:");

    // eigen analysis
    cx_vec cx_eigvals;
    cx_mat cx_eigvecs;
    eig_gen(cx_eigvals, cx_eigvecs, A);
    mat eigvecs = real(cx_eigvecs);
    vec eigvals = real(cx_eigvals);
    eigvals.print("eigen values:");
    eigvecs.print("eigen vectors:");
    
    //eigvecs.col(0)
    //vec smallest_eigvec = eigvecs.col(last_index);
    //vec smallest_eigvec = eigvecs.col(last_index-);
    vec smallest_eigvec = eigvecs.col(2);
    double smallest_eigval = eigvals(last_index);
    smallest_eigvec.print("Smallest Eigenvector");

    //stdvec solution_vec(mat_thickness)
    stdvec solution_vec = conv_to<stdvec>::from(smallest_eigvec);
    //sciplot::Vec solution_vec_sciplot(solution_vec)
    
    //A.print("A:");

    // Create a Plot object
    Plot2D plot;

    // Set the x and y labels
    plot.xlabel("x");
    plot.ylabel("y");
    //plot.size(600, 600);

    // Set the x and y ranges
    //plot.xrange(0.0, 5);
    //plot.yrange(-3, 3);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);

    Vec x = linspace(0.0, 5, mat_thickness);
    plot.drawPoints(x, solution_vec).pointType(6);
    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show();    
}