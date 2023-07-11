#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
using namespace arma; 
using namespace std; 
using namespace sciplot;
typedef std::vector<double> stdvec;

int main() {

    int N = 50;
    int last_index = N - 1; 
    mat A(N, N, arma::fill::zeros);

    // matrix construction
    A(0, 0) = 1;
    A(last_index, last_index) = 1;

    // matrix construction
    for (int i=1; i<last_index; i++)
    {
        A(i, arma::span(i-1, i+1)) = {1, -2, 1};
    }

    vec b(N, arma::fill::zeros);
    b(last_index) = 1;

    A.print("A:");

    vec sol_vec = arma::solve(A, b);

    //stdvec solution_vec(mat_thickness)
    
    stdvec solution_vec = conv_to<stdvec>::from(sol_vec);
    
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

    Vec x = linspace(0.0, 5, N);
    plot.drawPoints(x, solution_vec).pointType(6);
    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show();    
}