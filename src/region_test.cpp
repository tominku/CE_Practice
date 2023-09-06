#include <iostream> 
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

struct Region
{    
    std::string id;
    double doping;
    double eps;   

    Region(string id_, double doping_, double eps_)
    {
        id = id_;
        doping = doping_;
        eps = eps_;
    }
};
Region nwell_left = {"nwell_left", 1e23, eps_si};
Region nwell_right = {"nwell_right", 1e23, eps_si};
Region bulk = {"bulk", 1e21, eps_si};
Region regions[] = {bulk, nwell_left, nwell_right};

int main() {    
    for (int i=0; i<3; ++i)
    {
        Region region = regions[i];
        std::string debug_string = fmt::format("{}", region.doping);
        std::cout << region.id << ", " << region.doping << endl;
        //std::cout << "size of regions is:" << sizeof(regions) << endl;
    }    

    Region testRegion("testRegion", 2, 3);
    std::string debug_string = fmt::format("{}", testRegion.doping);
    std::cout << testRegion.id << ", " << testRegion.doping << endl;
}