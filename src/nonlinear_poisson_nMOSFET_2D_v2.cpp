#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"
#include <cassert> 

#define FMT_HEADER_ONLY
#include <fmt/format.h>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int Nx = 101;
const int Ny = 251;
const int N = Nx * Ny;
//double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
double thermal = k_B * T / q;

// double bulk_width =  5e-7;
// double bulk_height = 5e-7;
//double bulk_width =  5e-7;
double bulk_width =  8e-7;
double bulk_height = 1990e-9;
double ox_height = 10e-9;
double nwell_width = 1e-7;
double nwell_height = 1e-7;
double total_width = bulk_width;
double total_height = bulk_height + ox_height;
double deltaX = total_width / (Nx-1); // in meter  
double deltaY = (total_height) / (Ny-1); // in meter  
double ox_boundary_potential = 0.333703995136;
//double ox_boundary_potential = 0;

#define ijTok(i, j) (Nx*(j-1) + i)
#define phi_at(i, j, phi_name, phi_center_name) ((j >= 1 && j <= Ny && i >= 1 && i <= Nx) ? phi_name(ijTok(i, j)) : 0)
#define index_exist(i, j) ((j >= 1 && j <= Ny && i >= 1 && i <= Nx) ? true : false)

struct Coord
{    
    double x; double y;
    Coord(double _x, double _y)
    {
        x = _x;
        y = _y;
    }
};

struct Region
{    
    std::string id; double doping; double eps;   
    int x_begin; int x_end;
    int y_begin; int y_end;
};

const int min_x_index = 1;
const int max_x_index = (total_width/deltaX) + 1;
const int min_y_index = 1;
const int max_y_index = (total_height/deltaX) + 1;

Region bulk_region = {"bulk_region", -5e21, eps_si, 0, round(bulk_width/deltaX), 0, round(bulk_height/deltaY)};
Region ox_region = {"ox_region", 0, eps_ox, 0, round(bulk_width/deltaX), round(bulk_height/deltaY), round(total_height/deltaY)};
Region nwell_left_region = {"nwell_left_region", 5e23, eps_si, 
    0, round(nwell_width/deltaX), round((bulk_height-nwell_height)/deltaY), round(bulk_height/deltaY)};
Region nwell_right_region = {"nwell_right_region", 5e23, eps_si,
    Nx-1-round(nwell_width/deltaX), Nx-1, round((bulk_height-nwell_height)/deltaY), round(bulk_height/deltaY)};
Region regions[] = {bulk_region, ox_region, nwell_left_region, nwell_right_region};
int num_regions = 4;

Region contact1 = {"contact1", 0, 0, 
    0, 0, 
    round((bulk_height-(nwell_height/2))/deltaY), round(bulk_height/deltaY)};
Region contact2 = {"contact2", 0, 0,
    round(bulk_width/deltaX), round(bulk_width/deltaX), 
    round((bulk_height-(nwell_height/2))/deltaY), round(bulk_height/deltaY)};
Region contact_ox = {"contact_ox", 0, 0, 
    round(nwell_width/deltaX), Nx-1-round(nwell_width/deltaX), 
    round(total_height/deltaY), round(total_height/deltaY)};    

Region contacts[] = {contact1, contact2, contact_ox};
int num_contacts = 3;

bool belongs_to(double i, double j, Region &region)
{    
    if (i >= (double)region.x_begin && i <= (double)region.x_end && 
        j >= (double)region.y_begin && j <= (double)region.y_end)
        return true;
    else
        return false;
}

const string subject_name = "MOSFET_2D_NP";

#define INCLUDE_VFLUX true

double compute_eq_phi(double doping_density)
{
    /* 
        N^+_{dop} == n_int * 2 * sinh(phi/thermal)        
        sinh(phi/thermal) = N^+_{dop} / (n_int * 2)
        phi/thermal = asinh( N^+_{dop} / (n_int * 2) )
        phi = thermal *  asinh( N^+_{dop} / (n_int * 2) )
    */    
    double phi = thermal *  asinh( doping_density / (n_int * 2) );    
    
    return phi;        
}

bool is_contact_node(Coord coord)
{
    for (int c=0; c<num_contacts; c++)
    {
        Region contact = contacts[c];
        if (belongs_to(coord.x, coord.y, contact))
            return true;
    }
    return false;
}

void r_and_jacobian(vec &r, sp_mat &jac, vec &phi, double bias)
{
    r.fill(0.0);        
    jac.zeros();             
    
    // Handle contacts
    for (int c=0; c<num_contacts; c++)
    {
        Region contact = contacts[c];
        for (int j=contact.y_begin; j<=contact.y_end; j++)
        {
            for (int i=contact.x_begin; i<=contact.x_end; i++)
            {
                for (int p=0; p<num_regions; p++)
                {
                    Region region = regions[p];
                    if (belongs_to(i, j, region))
                    {
                        int k = ijTok(i+1, j+1);
                        double doping = region.doping;
                        if (doping != 0)
                        {                                           
                            r(k) = phi(k) - compute_eq_phi(region.doping); jac(k, k) = 1.0;                         
                        }
                        else                        
                        {
                            r(k) = phi(k) - ox_boundary_potential; jac(k, k) = 1.0;                        
                        }
                    }                    
                }
            }
        }
    } 

    double ion_term = 0.0;
    double eps_ipj = eps_si;
    double eps_imj = eps_si;
    double eps_ijp = eps_si;
    double eps_ijm = eps_si;   
    std::vector<Region> current_regions;           
    std::map<string, Coord *> epsID_to_coord;     
    std::map<string, std::pair<double, uint>> epsID_to_eps;   

    for (int j=1; j<=Ny; j++)    
    {
        for (int i=1; i<=Nx; i++)        
        {   
            Coord coord(i-1, j-1);
            if (is_contact_node(coord))
                continue;

            int k = ijTok(i, j);
            double phi_ij = phi(k);                                    

            epsID_to_coord.clear();
            Coord coord_ipj(i-1 + 0.5, j-1);
            Coord coord_imj(i-1 - 0.5, j-1);
            Coord coord_ijp(i-1, j-1 + 0.5);
            Coord coord_ijm(i-1, j-1 - 0.5);
            epsID_to_coord["eps_ipj"] = &coord_ipj;
            epsID_to_coord["eps_imj"] = &coord_imj;
            epsID_to_coord["eps_ijp"] = &coord_ijp;
            epsID_to_coord["eps_ijm"] = &coord_ijm;

            epsID_to_eps["eps_ipj"] = std::pair<double, uint>(0, 0);
            epsID_to_eps["eps_imj"] = std::pair<double, uint>(0, 0);
            epsID_to_eps["eps_ijp"] = std::pair<double, uint>(0, 0);
            epsID_to_eps["eps_ijm"] = std::pair<double, uint>(0, 0);

            current_regions.clear();
            for (int p=0; p<num_regions; p++)
            {
                Region region = regions[p];
                if (belongs_to(coord.x, coord.y, region))
                {
                    current_regions.push_back(region);
                    map<string, Coord *>::iterator it;           
                    for (it = epsID_to_coord.begin(); it != epsID_to_coord.end(); ++it)
                    {
                        Coord *fluxCoord = it->second;
                        if (belongs_to(fluxCoord->x, fluxCoord->y, region))   
                        {                     
                            std::pair<double, uint> eps_info = epsID_to_eps[it->first];
                            double eps_sum = eps_info.first;
                            uint num_overlaps = eps_info.second;
                            epsID_to_eps[it->first] = std::pair<double, uint>(eps_sum + region.eps, num_overlaps+1);
                        }
                    }
                }
            }
            if (epsID_to_eps["eps_ipj"].second > 0)
                eps_ipj = epsID_to_eps["eps_ipj"].first / epsID_to_eps["eps_ipj"].second;            
            if (epsID_to_eps["eps_imj"].second > 0)
                eps_imj = epsID_to_eps["eps_imj"].first / epsID_to_eps["eps_imj"].second;
            if (epsID_to_eps["eps_ijp"].second > 0)
                eps_ijp = epsID_to_eps["eps_ijp"].first / epsID_to_eps["eps_ijp"].second;
            if (epsID_to_eps["eps_ijm"].second > 0)
                eps_ijm = epsID_to_eps["eps_ijm"].first / epsID_to_eps["eps_ijm"].second;               

            Region doping_region = current_regions.back();
            ion_term = doping_region.doping;                          

            double phi_ipj = phi_at(i+1, j, phi, phi_ij);                           
            double phi_imj = phi_at(i-1, j, phi, phi_ij);                           
            double phi_ijp = phi_at(i, j+1, phi, phi_ij);                           
            double phi_ijm = phi_at(i, j-1, phi, phi_ij);                           

            double phi_diff_ipi = phi_ipj - phi_ij;
            double phi_diff_iim = phi_ij - phi_imj;
            double phi_diff_jpj = phi_ijp - phi_ij;
            double phi_diff_jjm = phi_ij - phi_ijm;

            // if (!INCLUDE_VFLUX)
            //     deltaY = 1;
            double s_ipj = deltaY;
            double s_imj = - deltaY;
            double s_ijp = deltaX;
            double s_ijm = - deltaX;
            if (!INCLUDE_VFLUX)            
                s_ijp = s_ijm = 0;                            
            
            double V = deltaX*deltaY;

            if (j == min_y_index)            
                s_ijm = 0; s_ipj *= 0.5; s_imj *= 0.5; V *= 0.5;                        
            if (j == max_y_index)                              
                s_ijp = 0; s_ipj *= 0.5; s_imj *= 0.5; V *= 0.5;                        
            if (i == min_x_index)            
                s_imj = 0; s_ijp *= 0.5; s_ijm *= 0.5; V *= 0.5;                                        
            if (i == max_x_index)            
                s_ipj = 0; s_ijp *= 0.5; s_ijm *= 0.5; V *= 0.5; 
            
            double D_ipj = -eps_ipj * phi_diff_ipi / deltaX;
            double D_imj = -eps_imj * phi_diff_iim / deltaX;                                        

            // Residual for the Poisson Equation
            if (INCLUDE_VFLUX)
            {
                double D_ijp = -eps_ijp * phi_diff_jpj / deltaY;
                double D_ijm = -eps_ijm * phi_diff_jjm / deltaY;        
                r(k) = s_ipj*D_ipj + s_imj*D_imj + s_ijp*D_ijp + s_ijm*D_ijm;            
            }
            else
                r(k) = s_ipj*D_ipj + s_imj*D_imj;

            r(k) -= V*q*(ion_term - n_int*( exp(phi_ij/thermal) - exp(-phi_ij/thermal) ));            
            r(k) /= eps_0;
                        
            // Jacobian for the nonlinear Poisson Equation
            if (INCLUDE_VFLUX)
            {
                jac(k, k) = s_ipj*eps_ipj/deltaX - s_imj*eps_imj/deltaX +
                    s_ijp*eps_ijp/deltaY - s_ijm*eps_ijm/deltaY;            
            }
            else
                jac(k, k) = s_ipj*eps_ipj/deltaX - s_imj*eps_imj/deltaX;                
            
            jac(k, k) += V*q*n_int*(1.0/thermal)*( exp(phi_ij/thermal) + exp(-phi_ij/thermal) );

            jac(k, k) /= eps_0;
            
            if (index_exist(i+1, j))
            {
                jac(k, ijTok(i+1, j)) = - s_ipj*eps_ipj / deltaX; 
                jac(k, ijTok(i+1, j)) /= eps_0;                       
            }
            if (index_exist(i-1, j))
            {
                jac(k, ijTok(i-1, j)) = s_imj*eps_imj / deltaX;               
                jac(k, ijTok(i-1, j)) /= eps_0;
            }
            
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))
                {
                    jac(k, ijTok(i, j+1)) = - s_ijp*eps_ijp / deltaY;
                    jac(k, ijTok(i, j+1)) /= eps_0;
                }
                if (index_exist(i, j-1))                        
                {
                    jac(k, ijTok(i, j-1)) = s_ijm*eps_ijm / deltaY;                                                                      
                    jac(k, ijTok(i, j-1)) /= eps_0;
                }
            }                                    
        }      
    }
}

vec solve_phi(double boundary_potential, vec &phi_0)
{                
    int num_iters = 25;    
    printf("boundary voltage: %f \n", boundary_potential);
    auto start = high_resolution_clock::now();
    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);
        
    vec r(N + 1, arma::fill::zeros);
    sp_mat jac(N + 1, N + 1);
    jac = jac.zeros();    
    
    vec phi_k(N + 1, arma::fill::zeros);     
    phi_k = phi_0;

    for (int k=0; k<num_iters; k++)
    {        
        r_and_jacobian(r, jac, phi_k, boundary_potential);   
        //jac.print("jac:");
        //printf("test");
        // xs.row(i) = x_i.t();         
        // residuals.row(i) = residual.t();     
        //residual.print("residual: "); 
        sp_mat jac_part = jac(span(1, N), span(1, N));  

        superlu_opts opts;
        opts.allow_ugly  = true;
        opts.equilibrate = true;
        //opts.refine = superlu_opts::REF_DOUBLE;

        vec delta_phi_k = arma::spsolve(jac_part, -r(span(1, N)));
        //phi_i(span(1, N - 1 - 1)) += delta_phi_i;                
        phi_k(span(1, N)) += delta_phi_k;                
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        double log_residual = log10(max(abs(r)));        
        double log_delta = log10(max(abs(delta_phi_k)));        
        log_residuals[k] = log_residual;
        log_deltas[k] = log_delta;
        //double cond_jac = arma::cond(jac);
        printf("[iter %d]   log detal_x: %f   log residual: %f \n", k, log_delta, log_residual);  
        
        // if (log_delta < - 10)
        //     break;
    }

    auto stop = high_resolution_clock::now();
    // Subtract stop and start timepoints and
    // cast it to required unit. Predefined units
    // are nanoseconds, microseconds, milliseconds,
    // seconds, minutes, hours. Use duration_cast()
    // function.
    auto duration = duration_cast<milliseconds>(stop - start);        
    cout << "duration: " << duration.count() << endl;

    std::string convergence_file_name = fmt::format("{}_conv_{:.2f}.csv", subject_name, boundary_potential);
    log_deltas.save(convergence_file_name, csv_ascii);            
        
    return phi_k;
    //plot(potentials, args);
    
    // if (plot_error)
    //     plot(log_deltas, args);
        //plot(log_residuals, args);

    //phi_i.print("found solution (phi):");    
    // vec n(N, arma::fill::zeros);
    // n(span(interface_i-1, interface2_i-1)) = n_int * exp(q * phi_i(span(interface_i-1, interface2_i-1)) / (k_B * T));
    // std::pair<vec, vec> result(phi_i, n);
    // return result;
}

void save_current_densities(vec &phi_n)
{
    vec phi = phi_n(span(1, N));
    vec n = phi_n(span(N+1, 2*N));
    vec current_densities(N+2, arma::fill::zeros);
    for (int i=2; i<=N-2; i++)
    {            
        double mu = 1417;
        double J_term1 = -q * mu * ((n(i+1) + n(i)) / 2.0) * ((phi(i+1) - phi(i)) / deltaX);
        double J_term2 = q * mu * thermal*(n(i+1) - n(i))/deltaX;        
        //double J = q * mu * (((n(j+1) + n(j)) / 2.0) * ((phi(j+1) - phi(j)) / deltaX) - thermal*(n(j+1) - n(j))/deltaX);
        double J = J_term1 + J_term2;
        J *= 1e-8;
        current_densities(i) = J;
        printf("Result Current Density J: %f, term1: %f, term2: %f \n", J, J_term1, J_term2);
    }
    //current_densities.save("current_densities.txt", arma::raw_ascii);
}


void fill_initial(vec &phi, string method)
{        
    std::vector<Region> current_regions;    
    for (int j=1; j<=Ny; j++)
    {
        for (int i=1; i<=Nx; i++)
        { 
            int k = ijTok(i, j);            
            current_regions.clear();
            for (int p=0; p<num_regions; p++)
            {
                Region region = regions[p];
                if (belongs_to(i-1, j-1, region))
                    current_regions.push_back(region);
            }
            Region doping_region = current_regions.back();
            double doping = doping_region.doping;            
            if (doping != 0)
            {
                double eq_phi = compute_eq_phi(doping);
                phi(k) = eq_phi;
            }
            else
            {                
                phi(k) = 0;
            }
        }
     }      
}


int main() {    

    double start_potential = 0;    

    double phi_test =compute_eq_phi(-5e21);
    printf("hole density: %f \n", phi_test);

    vec one_vector(N+1, arma::fill::ones);
    vec phi_0(N+1, arma::fill::zeros);
    fill_initial(phi_0, "uniform");
    //fill_initial(phi_0, "random");
    //fill_initial(phi_0, "linear");
    //for (int i=0; i<10; i++)
    {        
        int i = 0;
        vec phi = solve_phi(start_potential + (0.1*i), phi_0); 
        phi_0 = phi;   
        
        std::string log = fmt::format("BD {:.2f} V \n", start_potential + (0.1*i));            
        cout << log;        
        
        vec n(N+1, arma::fill::zeros);
        n(span(1, N)) = n_int * exp(phi(span(1, N)) / thermal);
        n /= 1e6;        
        vec eDensity = n(span(1, N));        
        std::string n_file_name = fmt::format("{}_eDensity_{:.2f}.csv", subject_name, (0.1*i));
        eDensity.save(n_file_name, csv_ascii);        

        vec h(N+1, arma::fill::zeros);
        h(span(1, N)) = n_int * exp(- phi(span(1, N)) / thermal);
        h /= 1e6;        
        vec holeDensity = h(span(1, N));        

        std::string h_file_name = fmt::format("{}_holeDensity_{:.2f}.csv", subject_name, (0.1*i));
        holeDensity.save(h_file_name, csv_ascii);        
        
        vec phi_for_plot = phi(span(1, N));
        std::string phi_file_name = fmt::format("{}_phi_{:.2f}.csv", subject_name, (0.1*i));
        phi_for_plot.save(phi_file_name, csv_ascii);                        

        vec phi_n(2*N+1, arma::fill::zeros);
        phi_n(span(1, N)) = phi(span(1, N));
        phi_n(span(N+1, 2*N)) = eDensity;        

        //save_current_densities(phi_n);
    }
}
