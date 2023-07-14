#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

double f(double x)
{
    return x*x - 1;
}

double df_over_dx(double x)
{
    return 2*x;
}

void plot(stdvec &xs, stdvec &residuals);

int main() {

    double x_0 = 2;    
    int num_iters = 20;
    stdvec xs;
    stdvec residuals;
    double x_i = x_0;
    for (int i=0; i<num_iters; i++)
    {
        double residual = f(x_i);
        xs.push_back(x_i);        
        residuals.push_back(residual);
        double delta_x = (1 / df_over_dx(x_i) ) * ( - residual );
        x_i += delta_x;        
        printf("x_i: %f \n", x_i);
    }

    plot(xs, residuals);
}

void plot(stdvec &xs, stdvec &residuals)
{
    Plot2D plot;

    plot.xlabel("x_i");
    plot.ylabel("residual: f(x)");

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    plot.grid().show();

    plot.drawPoints(xs, residuals).pointType(6).pointSize(2);
    plot.drawCurve(xs, residuals);
    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show(); 
}