#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include <math.h>
using namespace arma; 
using namespace std; 
using namespace sciplot;
typedef std::vector<double> stdvec;
typedef std::vector < std::pair<double, double> > pairlist;

const double q = 1.602192e-19; // (C)
const double k_B = 1.380662e-23; // (J/K)
double eps_0 = 8.854187817 * 1e-12; // (F/m)

// A function that matches node index i to permitivity (eps).
// i starts from 1.
double i_to_eps(double i, double total_width, int N, pairlist& width_eps_list)
{   
    double delta_x = total_width * 1.0 / double(N-1);
    double physical_position = delta_x * (i - 1);
    
    pairlist::iterator it;   
    double accumulated_width = 0.0;
    double found_eps = -1.0;
    for (it = width_eps_list.begin(); it != width_eps_list.end(); ++it) 
    {
        double width = (*it).first;  
        double eps = (*it).second;
        //std::cout << "*it " << " = " << (*it).first << " : " << (*it).second << std::endl;
        accumulated_width += width;
        if (physical_position <= accumulated_width)
        {
            found_eps = eps;
            break;
        }
                      
    }        
    return found_eps;
}

std::pair<mat, vec> construct_A_b_poisson(double total_width, int N, 
    pairlist& width_eps_list, std::pair<double, double> end_potentials)
{    
    mat A(N, N, arma::fill::zeros);
    int last_index = N - 1; 
    
    // matrix construction
    A(0, 0) = 1;
    A(last_index, last_index) = 1;

    // matrix construction
    for (int k=1; k<last_index; k++)
    {
        double i = double(k+1);
        rowvec row_elements(3);
        row_elements(0) = i_to_eps(i - 0.5, total_width, N, width_eps_list);
        row_elements(1) = - ( i_to_eps(i - 0.5, total_width, N, width_eps_list) + i_to_eps(i + 0.5, total_width, N, width_eps_list) );
        row_elements(2) = i_to_eps(i + 0.5, total_width, N, width_eps_list);
        A(k, span(k-1, k+1)) = row_elements;
    }

    vec b(N, arma::fill::zeros);    
    b(0) = end_potentials.first;
    b(last_index) = end_potentials.second;       

    return std::pair<mat, vec>(A, b);    
}

void plot(double total_width, int N, vec& y)
{
    stdvec solution_vec = conv_to<stdvec>::from(y);

    Plot2D plot;
    plot.xlabel("Position (nm)");
    plot.ylabel("Potential (V)");

    // Set the x and y ranges
    //plot.xrange(0.0, 5);
    //plot.yrange(-3, 3);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    plot.grid().show();
    
    //sciplot::linspace(0.0, 6.0, N)
    sciplot::Vec x = sciplot::linspace(0.0, total_width, N);
    plot.drawCurve(x, solution_vec).label("Potential");
    plot.drawPoints(x, solution_vec).pointType(6).label("Potential");
    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show();  
}