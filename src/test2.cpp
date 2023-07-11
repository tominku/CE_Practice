#include <sciplot/sciplot.hpp>
//include "sciplot/sciplot.hpp"

using namespace sciplot;

int main(int argc, char** argv)
{
    // Create a vector with values from 0 to pi divived into 200 uniform intervals for the x-axis
    Vec x = linspace(0.0, PI, 50);

    // Create a Plot object
    Plot2D plot;

    // Set the x and y labels
    plot.xlabel("x");
    plot.ylabel("y");
    //plot.size(600, 600);

    // Set the x and y ranges
    plot.xrange(0.0, PI);
    plot.yrange(0.0, 1.0);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);

    // Plot sin(i*x) from i = 1 to i = 6
    plot.drawDots(x, std::sin(1.0 * x)).label("sin(x)");
    plot.drawCurve(x, std::sin(2.0 * x)).label("sin(x)");
    // plot.drawCurve(x, std::sin(2.0 * x)).label("sin(2x)");
    // plot.drawCurve(x, std::sin(3.0 * x)).label("sin(3x)");
    // plot.drawCurve(x, std::sin(4.0 * x)).label("sin(4x)");
    // plot.drawCurve(x, std::sin(5.0 * x)).label("sin(5x)");
    // plot.drawCurve(x, std::sin(6.0 * x)).label("sin(6x)");

    // Create figure to hold plot
    Figure fig = {{plot}};
    // Create canvas to hold figure
    Canvas canvas = {{fig}};
    canvas.size(800, 400);

    // Show the plot in a pop-up window
    canvas.show();

    // Save the plot to a PDF file
    //canvas.save("example-sine-functions.pdf");
}