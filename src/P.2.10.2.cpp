#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
using namespace arma; 

const double n_int = 1e10;    
const double T = 300; // (K)
const double V_T = k_B * T / q;

double f(double phi, double dop)
{            
    double f_value = n_int * ( exp(phi / V_T) - exp(-phi / V_T) ) - dop;
    return f_value;
}

double df_over_dx(double phi)
{
    double derivative_value = n_int * ( (1.0 / V_T)*exp(phi / V_T) - (-1.0 / V_T)*exp(-phi / V_T) );
    return derivative_value;
}

void plot(stdvec &doping_list, stdvec &phis);

int main() {

    double x_0 = 3;    
    double dop = 1e18;
    int num_iters = 500;
    stdvec xs;
    stdvec residuals;

    int num_experiments = 9;
    vec phis(num_experiments);
    vec doping_list(num_experiments);
    doping_list[0] = 1e10;
    for (int i=1; i<doping_list.n_elem; ++i)
    {
        doping_list[i] = doping_list[0] * pow(10, i); 
    }

    for (int k=0; k<doping_list.n_elem; ++k)
    {
        double x_i = x_0;
        double dop = doping_list[k];
        for (int i=0; i<num_iters; i++)
        {
            double residual = f(x_i, dop);
            xs.push_back(x_i);        
            residuals.push_back(residual);
            double delta_x = (1 / df_over_dx(x_i) ) * ( - residual );
            x_i += delta_x;                    
        }
        phis[k] = x_i;
        printf("doping: 1e%d, x_sol: %f \n", 10+k, x_i);
    }

    stdvec std_doping_list = conv_to<stdvec>::from(doping_list);
    stdvec std_phis = conv_to<stdvec>::from(phis);
    plot(std_doping_list, std_phis);
}


void plot(stdvec &doping_list, stdvec &phis)
{
    Plot2D plot;

    plot.xlabel("Doping Density (/cm^3)");
    plot.ylabel("Electrostatic Potential(V)");

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    plot.grid().show();

    plot.xtics().logscale(10);

    plot.drawPoints(doping_list, phis).pointType(6).pointSize(2);
    plot.drawCurve(doping_list, phis);
    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show(); 
}