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
//const int Ny = 251;
const int Ny = 21;
const int N = Nx * Ny;
//double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double T = 300;    
double thermal = k_B * T / q;

// double bulk_width =  5e-7;
// double bulk_height = 5e-7;
//double bulk_width =  5e-7;
double bulk_width =  4e-7;
// double bulk_height = 1990e-9;
// double ox_height = 10e-9;
double nwell_width = 2e-7;
double nwell_height = 80e-9;
double total_width = bulk_width;
double bulk_height = nwell_height;
double total_height = bulk_height;
double deltaX = total_width / (Nx-1); // in meter  
double deltaY = (total_height) / (Ny-1); // in meter  
double ox_boundary_potential = 0.333703995136;

#define ijTok(i, j) (Nx*(j-1) + i)
#define index_exist(i, j) ((j >= 1 && j <= Ny && i >= 1 && i <= Nx) ? true : false)
#define phi_at(i, j, var_name) (index_exist(i, j) ? var_name(ijTok(i, j)) : 0)
#define n_at(i, j, var_name) (index_exist(i, j) ? var_name(N + ijTok(i, j)) : 0)
#define p_at(i, j, var_name) (index_exist(i, j) ? var_name(2*N + ijTok(i, j)) : 0)

vec one_vector(3*N, fill::ones);
double B(double x);
double dB(double x);

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
const int max_y_index = (total_height/deltaY) + 1;

Region bulk_region = {"bulk_region", -5e21, eps_si, 0, round(bulk_width/deltaX), 0, round(bulk_height/deltaY)};
Region nwell_left_region = {"nwell_left_region", 5e23, eps_si, 
    0, round(nwell_width/deltaX), 0, round(bulk_height/deltaY)};
Region pwell_right_region = {"pwell_right_region", -2e23, eps_si,
    Nx-1-round(nwell_width/deltaX), Nx-1, 0, round(bulk_height/deltaY)};
Region regions[] = {bulk_region, nwell_left_region, pwell_right_region};
//Region regions[] = {nwell_left_region, pwell_right_region};
int num_regions = 3;

Region contact_n = {"contact_n", 0, 0, 
    0, 0, 
    0, round(bulk_height/deltaY)};
Region contact_p = {"contact_p", 0, 0,
    round(bulk_width/deltaX), round(bulk_width/deltaX), 
    0, round(bulk_height/deltaY)};

Region contacts[] = {contact_n, contact_p};
int num_contacts = 2;

bool belongs_to(double i, double j, Region &region)
{    
    if (i >= (double)region.x_begin && i <= (double)region.x_end && 
        j >= (double)region.y_begin && j <= (double)region.y_end)
        return true;
    else
        return false;
}

const string subject_name = "DD_PN_2D_v4";

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

void r_and_jacobian(vec &r, sp_mat &jac, vec &phi_n_p, std::map<std::string, double> &contactID_to_bias)
{
    r.fill(0.0);        
    jac.zeros();             
    
    // Handle contacts
    for (int c=0; c<num_contacts; c++)
    {
        Region contact = contacts[c];
        for (int j_=contact.y_begin; j_<=contact.y_end; j_++)
        {
            for (int i_=contact.x_begin; i_<=contact.x_end; i_++)
            {
                int i = i_ + 1;
                int j = j_ + 1;                
                std::string contact_id = contact.id;
                
                double bias = 0;
                if (contactID_to_bias.find(contact_id) != contactID_to_bias.end()) 
                    bias = contactID_to_bias[contact.id];                

                for (int p=0; p<num_regions; p++)
                {
                    Region region = regions[p];
                    if (belongs_to(i_, j_, region))
                    {
                        int k = ijTok(i, j);
                        double doping = region.doping;
                        if (doping != 0)
                        {                                                                               
                            r(k) = phi_at(i, j, phi_n_p) - compute_eq_phi(doping) - bias;        
                            if (doping > 0)
                            {
                                r(N + k) = n_at(i, j, phi_n_p) - doping;        
                                r(2*N + k) = p_at(i, j, phi_n_p) - (n_int*n_int/doping);
                            }
                            else
                            {
                                r(N + k) = n_at(i, j, phi_n_p) - fabs(n_int*n_int/doping);        
                                r(2*N + k) = p_at(i, j, phi_n_p) - fabs(doping);
                            }
                            jac(k, k) = 1.0;  
                            jac(N + k, N + k) = 1.0; 
                            jac(2*N + k, 2*N + k) = 1.0;                                      
                        }
                        else // metal-oxide contact                 
                        {
                            //r(k) = phi_at(i, j, phi_n_p) - ox_boundary_potential - gate_bias; 
                            r(k) = phi_at(i, j, phi_n_p) - ox_boundary_potential - bias; 
                            r(N + k) = n_at(i, j, phi_n_p) - 0;        
                            r(2*N + k) = p_at(i, j, phi_n_p) - 0;
                            jac(k, k) = 1.0;  
                            jac(N + k, N + k) = 1.0; 
                            jac(2*N + k, 2*N + k) = 1.0;                                      
                        }
                    }                    
                }
            }
        }
    } 

    double ion_term = 0.0;
    double eps_ipj = 0;
    double eps_imj = 0;
    double eps_ijp = 0;
    double eps_ijm = 0;   
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
            //double phi_ij = phi(k);      
            double phi_ij = phi_at(i, j, phi_n_p);  
            double n_ij = n_at(i, j, phi_n_p);
            double p_ij = p_at(i, j, phi_n_p);                                                         

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

            double phi_ipj = phi_at(i+1, j, phi_n_p);                           
            double phi_imj = phi_at(i-1, j, phi_n_p);                           
            double phi_ijp = phi_at(i, j+1, phi_n_p);                           
            double phi_ijm = phi_at(i, j-1, phi_n_p);                           

            double n_ipj = n_at(i+1, j, phi_n_p);                           
            double n_imj = n_at(i-1, j, phi_n_p);                           
            double n_ijp = n_at(i, j+1, phi_n_p);                           
            double n_ijm = n_at(i, j-1, phi_n_p);                                       

            double p_ipj = p_at(i+1, j, phi_n_p);                           
            double p_imj = p_at(i-1, j, phi_n_p);                           
            double p_ijp = p_at(i, j+1, phi_n_p);                           
            double p_ijm = p_at(i, j-1, phi_n_p);                           

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
            {
                s_ijm = 0; s_ipj *= 0.5; s_imj *= 0.5; V *= 0.5;                        
            }
            if (j == max_y_index)                              
            {
                s_ijp = 0; s_ipj *= 0.5; s_imj *= 0.5; V *= 0.5;                        
            }
            if (i == min_x_index)     
            {                                   
                s_imj = 0; s_ijp *= 0.5; s_ijm *= 0.5; V *= 0.5;                                                                
            }
            if (i == max_x_index)             
            {
                s_ipj = 0; s_ijp *= 0.5; s_ijm *= 0.5; V *= 0.5;                             
            }
            
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

            //r(k) -= V*q*(ion_term - n_int*( exp(phi_ij/thermal) - exp(-phi_ij/thermal) ));            
            r(k) -= V*q*(ion_term - n_ij + p_ij);   
            r(k) /= eps_0;
                        
            // Jacobian for the nonlinear Poisson Equation
            if (INCLUDE_VFLUX)
            {
                jac(k, k) = s_ipj*eps_ipj/deltaX - s_imj*eps_imj/deltaX +
                    s_ijp*eps_ijp/deltaY - s_ijm*eps_ijm/deltaY;            
            }
            else
                jac(k, k) = s_ipj*eps_ipj/deltaX - s_imj*eps_imj/deltaX;                
            
            //jac(k, k) += V*q*n_int*(1.0/thermal)*( exp(phi_ij/thermal) + exp(-phi_ij/thermal) );

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

            jac(k, N + ijTok(i, j)) =  q*V; // r w.r.t. n                        
            jac(k, 2*N + ijTok(i, j)) = - q*V; // r w.r.t. p    
            jac(k, N + ijTok(i, j)) /=  eps_0;
            jac(k, 2*N + ijTok(i, j)) /=  eps_0;              

            // Residual for the SG (n)     
            double Jn_ipj = n_ipj*B(phi_diff_ipi/thermal) - n_ij*B(-phi_diff_ipi/thermal);
            double Jn_imj = n_ij*B(phi_diff_iim/thermal) - n_imj*B(-phi_diff_iim/thermal);                        
            if (INCLUDE_VFLUX)
            {
                double Jn_ijp = n_ijp*B(phi_diff_jpj/thermal) - n_ij*B(-phi_diff_jpj/thermal);
                double Jn_ijm = n_ij*B(phi_diff_jjm/thermal) - n_ijm*B(-phi_diff_jjm/thermal);
                r(N + k) = s_ipj*Jn_ipj/deltaX + s_imj*Jn_imj/deltaX + s_ijp*Jn_ijp/deltaY + s_ijm*Jn_ijm/deltaY;
            }            
            else
                r(N + k) = s_ipj*Jn_ipj/deltaX + s_imj*Jn_imj/deltaX;

            // Jacobian for the SG (n)
            // w.r.t. phis
            double a, b, c, d;
            a = b = c = d = 0;
            if (index_exist(i+1, j))
                jac(N + k, ijTok(i+1, j)) = a = s_ipj*(n_ij*dB(-phi_diff_ipi/thermal) + n_ipj*dB(phi_diff_ipi/thermal)) / (thermal*deltaX);
            if (index_exist(i-1, j))
                jac(N + k, ijTok(i-1, j)) = b = s_imj*(-n_imj*dB(-phi_diff_iim/thermal) - n_ij*dB(phi_diff_iim/thermal)) / (thermal*deltaX);
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))                        
                    jac(N + k, ijTok(i, j+1)) = c = s_ijp*(n_ij*dB(-phi_diff_jpj/thermal) + n_ijp*dB(phi_diff_jpj/thermal)) / (thermal*deltaY);
                if (index_exist(i, j-1))                        
                    jac(N + k, ijTok(i, j-1)) = d = s_ijm*(-n_ijm*dB(-phi_diff_jjm/thermal) - n_ij*dB(phi_diff_jjm/thermal)) / (thermal*deltaY);
                jac(N + k, ijTok(i, j)) = -a-b-c-d;
            }
            else
                jac(N + k, ijTok(i, j)) = -a-b;
            // w.r.t. ns
            if (index_exist(i+1, j))
                jac(N + k, N + ijTok(i+1, j)) = s_ipj*B(phi_diff_ipi/thermal) / deltaX;
            if (index_exist(i-1, j))                
                jac(N + k, N + ijTok(i-1, j)) = -s_imj*B(-phi_diff_iim/thermal) / deltaX;
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))                        
                    jac(N + k, N + ijTok(i, j+1)) = s_ijp*B(phi_diff_jpj/thermal) / deltaY;
                if (index_exist(i, j-1))                        
                    jac(N + k, N + ijTok(i, j-1)) = -s_ijm*B(-phi_diff_jjm/thermal) / deltaY;
                jac(N + k, N + ijTok(i, j)) = - s_ipj*B(-phi_diff_ipi/thermal)/deltaX + s_imj*B(phi_diff_iim/thermal)/deltaX - s_ijp*B(-phi_diff_jpj/thermal)/deltaY + s_ijm*B(phi_diff_jjm/thermal)/deltaY;
            }
            else
                jac(N + k, N + ijTok(i, j)) = - s_ipj*B(-phi_diff_ipi/thermal)/deltaX + s_imj*B(phi_diff_iim/thermal)/deltaX;        

            // Residual for the SG (p)    
            double Jp_ipj = -p_ipj*B(-phi_diff_ipi/thermal) + p_ij*B(phi_diff_ipi/thermal);
            double Jp_imj = -p_ij*B(-phi_diff_iim/thermal) + p_imj*B(phi_diff_iim/thermal);                        
            if (INCLUDE_VFLUX)
            {
                double Jp_ijp = -p_ijp*B(-phi_diff_jpj/thermal) + p_ij*B(phi_diff_jpj/thermal);
                double Jp_ijm = -p_ij*B(-phi_diff_jjm/thermal) + p_ijm*B(phi_diff_jjm/thermal);                
                r(2*N + k) = s_ipj*Jp_ipj/deltaX + s_imj*Jp_imj/deltaX + s_ijp*Jp_ijp/deltaY + s_ijm*Jp_ijm/deltaY;                    
            }
            else
                r(2*N + k) = s_ipj*Jp_ipj/deltaX + s_imj*Jp_imj/deltaX;

            // Jacobian for the SG (p)
            // w.r.t. phis
            a = b = c = d = 0;
            if (index_exist(i+1, j))
                jac(2*N + k, ijTok(i+1, j)) = a = s_ipj*(p_ij*dB(phi_diff_ipi/thermal) + p_ipj*dB(-phi_diff_ipi/thermal)) / (thermal*deltaX);                      
            if (index_exist(i-1, j))
                jac(2*N + k, ijTok(i-1, j)) = b = s_imj*(- p_imj*dB(phi_diff_iim/thermal) - p_ij*dB(-phi_diff_iim/thermal)) / (thermal*deltaX);
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))                        
                    jac(2*N + k, ijTok(i, j+1)) = c = s_ijp*(p_ij*dB(phi_diff_jpj/thermal) + p_ijp*dB(-phi_diff_jpj/thermal)) / (thermal*deltaY);
                if (index_exist(i, j-1))                        
                    jac(2*N + k, ijTok(i, j-1)) = d = s_ijm*(- p_ijm*dB(phi_diff_jjm/thermal) - p_ij*dB(-phi_diff_jjm/thermal)) / (thermal*deltaY);
                jac(2*N + k, ijTok(i, j)) = -a-b-c-d;
            }
            else
                jac(2*N + k, ijTok(i, j)) = -a-b;
            // w.r.t. ps
            if (index_exist(i+1, j))
                jac(2*N + k, 2*N + ijTok(i+1, j)) = -s_ipj*B(-phi_diff_ipi/thermal) / deltaX;            
            if (index_exist(i-1, j))                
                jac(2*N + k, 2*N + ijTok(i-1, j)) = s_imj*B(phi_diff_iim/thermal) / deltaX;   
            if (INCLUDE_VFLUX)
            {         
                if (index_exist(i, j+1))                        
                    jac(2*N + k, 2*N + ijTok(i, j+1)) = -s_ijp*B(-phi_diff_jpj/thermal) / deltaY;
                if (index_exist(i, j-1))                        
                    jac(2*N + k, 2*N + ijTok(i, j-1)) = s_ijm*B(phi_diff_jjm/thermal) / deltaY;
                jac(2*N + k, 2*N + ijTok(i, j)) = s_ipj*B(phi_diff_ipi/thermal)/deltaX - s_imj*B(-phi_diff_iim/thermal)/deltaX + s_ijp*B(phi_diff_jpj/thermal)/deltaY - s_ijm*B(-phi_diff_jjm/thermal)/deltaY;
            }
            else
                jac(2*N + k, 2*N + ijTok(i, j)) = s_ipj*B(phi_diff_ipi/thermal)/deltaX - s_imj*B(-phi_diff_iim/thermal)/deltaX;
        }      
    }
}

vec solve_phi(std::map<string, double> contactID_to_bias, vec &phi_n_p_0, sp_mat &C)
{                
    int num_iters = 20;        
    auto start = high_resolution_clock::now();
    vec log_residuals(num_iters, arma::fill::zeros);
    vec log_deltas(num_iters, arma::fill::zeros);
        
    vec r(3*N + 1, arma::fill::zeros);
    sp_mat jac(3*N + 1, 3*N + 1);
    jac = jac.zeros();    
    
    vec phi_n_p_k(3*N + 1, arma::fill::zeros);     
    phi_n_p_k = phi_n_p_0;    

    for (int k=0; k<num_iters; k++)
    {        
        r_and_jacobian(r, jac, phi_n_p_k, contactID_to_bias);   
                                      
        sp_mat jac_scaled = jac(span(1, 3*N), span(1, 3*N)) * C;        

        sp_mat r_vector_temp = arma::sum(abs(jac_scaled), 1);
        vec r_vector(3*N, fill::zeros);
        for (int p=0; p<3*N; p++)        
            r_vector(p) = 1 / (r_vector_temp(p) + 1e-10);                        
        
        sp_mat R(3*N, 3*N);            
        R.zeros();
        for (int m=0; m<3*N; m++)
            R(m, m) = r_vector(m);

        //mat R_eye = eye(2*N, 2*N);
        //R = R_eye;
        jac_scaled = R * jac_scaled;
        vec r_scaled = R * r(span(1, 3*N));

        vec delta_phi_n_p = arma::spsolve(jac_scaled, -r_scaled);        
        //vec delta_phi = arma::solve(jac(span(1, 2*N), span(1, 2*N)), -r(span(1, 2*N)));        
        vec update_vector = C * delta_phi_n_p;
        phi_n_p_k(span(1, 3*N)) += update_vector;                        

        // superlu_opts opts;
        // opts.allow_ugly  = true;
        // opts.equilibrate = true;
        //opts.refine = superlu_opts::REF_DOUBLE;
        
        //phi_i.print("phi_i");
        //jac.print("jac");
        //if (i % 1 == 0)
        //printf("[iter %d]   detal_x: %f   residual: %f\n", i, max(abs(delta_phi_i)), max(abs(residual)));  
        double log_residual = log10(max(abs(r)));          
        double log_delta = log10(max(abs(update_vector(span(0, N-1)))));        
        log_residuals[k] = log_residual;
        log_deltas[k] = log_delta;

        printf("[iter %d]   log detal_x: %f   log residual: %f \n", k, log_delta, log_residual);  
        
        if (log_delta < - 10)
            break;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);        
    cout << "duration: " << duration.count() << endl;

    // std::string convergence_file_name = fmt::format("{}_conv_{:.2f}.csv", subject_name, gate_bias);
    // log_deltas.save(convergence_file_name, csv_ascii);            
        
    return phi_n_p_k;
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

vec get_current_density(vec &phi_n_p)
{    
    double J_SG = 0;    
    std::vector<std::pair<int, int>> coord_list;
    coord_list.push_back(std::pair<int, int>(2 + 1, (bulk_height - nwell_height/2) / deltaY + 1));
    coord_list.push_back(std::pair<int, int>((bulk_width) / deltaX - 2, (bulk_height - nwell_height/2) / deltaY + 1));
    // coord_list.push_back(std::pair<int, int>(3 + 1, (bulk_height - nwell_height/3) / deltaY + 1));
    //coord_list.push_back(std::pair<int, int>((bulk_width) / deltaX - 20 + 1, (bulk_height - nwell_height/3) / deltaY + 1));
    // int i = 3 + 1;
    // int j = (bulk_height - nwell_height/3) / deltaY + 1;
    int num_investigation_points = coord_list.size();
    vec Js(num_investigation_points, arma::fill::zeros);
    for (int p=0; p<num_investigation_points; ++p)
    {         
        std::pair<int, int> coord = coord_list[p];
        int i = coord.first;
        int j = coord.second;
        double mu_n = 1417;
        double mu_h = 470.5;
        double phi_ij = phi_at(i, j, phi_n_p);                           
        double n_ij = n_at(i, j, phi_n_p);                           
        double p_ij = p_at(i, j, phi_n_p);                           

        double phi_ipj = phi_at(i+1, j, phi_n_p);                           
        double phi_imj = phi_at(i-1, j, phi_n_p);                           
        double phi_ijp = phi_at(i, j+1, phi_n_p);                           
        double phi_ijm = phi_at(i, j-1, phi_n_p);                           

        double n_ipj = n_at(i+1, j, phi_n_p);                           
        double n_imj = n_at(i-1, j, phi_n_p);                           
        double n_ijp = n_at(i, j+1, phi_n_p);                           
        double n_ijm = n_at(i, j-1, phi_n_p);                                       

        double p_ipj = p_at(i+1, j, phi_n_p);                           
        double p_imj = p_at(i-1, j, phi_n_p);                           
        double p_ijp = p_at(i, j+1, phi_n_p);                           
        double p_ijm = p_at(i, j-1, phi_n_p);         
        // double J_term1 = -q * mu_n * ((n(i+1) + n(i)) / 2.0) * ((phi(i+1) - phi(i)) / deltaX);
        // double J_term2 = q * mu_n * thermal*(n(i+1) - n(i))/deltaX;
        //double J = q * mu_n * (((n(j+1) + n(j)) / 2.0) * ((phi(j+1) - phi(j)) / deltaX) - thermal*(n(j+1) - n(j))/deltaX);
        //double J = J_term1 + J_term2;
        double phi_diff_ipi = phi_ipj - phi_ij;
        double J_n_SG = n_ipj*B(phi_diff_ipi / thermal) - n_ij*B((-phi_diff_ipi) / thermal);
        double J_h_SG = -p_ipj*B(-(phi_diff_ipi) / thermal) + p_ij*B(phi_diff_ipi / thermal);
        //double J = q * mu_n * (((n(j+1) + n(j)) / 2.0) * ((phi(j+1) - phi(j)) / deltaX) - thermal*(n(j+1) - n(j))/deltaX);
        J_n_SG *= 1e-8;
        J_h_SG *= 1e-8;
        J_SG = mu_n*J_n_SG - mu_h*J_h_SG;
        J_SG *= q * thermal / deltaX;
        Js(p) = J_SG;
        //J_SG *= 1e6;
        
        //current_densities(i) = J_SG;
        //printf("Result Current Density J: %f (J_n_SG: %f, J_h_SG: %f) \n", J_SG, J_n_SG, J_h_SG);
        //printf("Result Current Density J: %f, term1: %f, term2: %f, J_SG: %f \n", J, J_term1, J_term2, J_SG);
    }
    //current_densities.save("current_densities.txt", arma::raw_ascii);

    return Js;
}


int main() {    

    std::map<std::string, double> contactID_to_bias;    
    
    string setting = fmt::format("deltaX: {}, deltaY: {}", deltaX, deltaY); 
    cout << setting << "\n";        
    double bias = 0.0;    
    vec phi_n_p_0(3*N+1, arma::fill::zeros);    
    bool load_initial_solution_from_NP = true;    
    if (load_initial_solution_from_NP)
    {
        string load_subject_name = "NP_PN_2D_v4";
        std::string file_name = fmt::format("{}_phi_{:.2f}.csv", load_subject_name, 0.0); 
        cout << file_name << "\n";        
        vec phi_from_NP(N, fill::zeros);
        phi_from_NP.load(file_name);

        file_name = fmt::format("{}_eDensity_{:.2f}.csv", load_subject_name, 0.0); 
        cout << file_name << "\n";        
        vec eDensity_from_NP(N, fill::zeros); 
        eDensity_from_NP.load(file_name);        

        file_name = fmt::format("{}_holeDensity_{:.2f}.csv", load_subject_name, 0.0); 
        cout << file_name << "\n";        
        vec holeDensity_from_NP(N, fill::zeros);
        holeDensity_from_NP.load(file_name);           
                        
        phi_n_p_0(span(1, N)) = phi_from_NP(span(0, N-1));        
        phi_n_p_0(span(N+1, 2*N)) = eDensity_from_NP(span(0, N-1));        
        phi_n_p_0(span(2*N+1, 3*N)) = holeDensity_from_NP(span(0, N-1));        
    }
    
    sp_mat C(3*N, 3*N);        
    C.zeros();
    for (int m=0; m<N; m++)
        C(m, m) = thermal;
    for (int m=N; m<2*N; m++)
        C(m, m) = nwell_left_region.doping;
    for (int m=2*N; m<3*N; m++)
        C(m, m) = nwell_left_region.doping;  

    for (int i=0; i<41; i++)    
    {           
        //double bias = 0.1*8;        
        double bias = 0.02*i;        
        contactID_to_bias["contact_p"] = bias;
        vec result = solve_phi(contactID_to_bias, phi_n_p_0, C); 
        phi_n_p_0 = result;           
        vec Js = get_current_density(phi_n_p_0);
        printf("J1: %f \n", Js[0]);
        printf("J2: %f \n", Js[1]);
        
        std::string log = fmt::format("Bias {:.2f} V \n", bias);            
        cout << log;            
                
        vec phi = phi_n_p_0(span(1, N));                
        vec eDensity = phi_n_p_0(span(N+1, 2*N));        
        eDensity /= 1e6;
        vec holeDensity = phi_n_p_0(span(2*N+1, 3*N));        
        holeDensity /= 1e6;
        
        std::string phi_file_name = fmt::format("{}_phi_B_{:.2f}.csv", subject_name, bias);
        phi.save(phi_file_name, csv_ascii);                

        std::string n_file_name = fmt::format("{}_eDensity_B_{:.2f}.csv", subject_name, bias);
        eDensity.save(n_file_name, csv_ascii);                

        std::string h_file_name = fmt::format("{}_holeDensity_B_{:.2f}.csv", subject_name, bias);
        holeDensity.save(h_file_name, csv_ascii);                                     

        std::string j_file_name = fmt::format("{}_J_B_{:.2f}.csv", subject_name, bias);
        Js.save(j_file_name, csv_ascii);                                                         
    }
}

double B(double x)
{
    double result = 0.0;
    
    if (abs(x) < 0.0252)   
        // Bern_P1 = ( 1.0-(x1)/2.0+(x1)^2/12.0*(1.0-(x1)^2/60.0*(1.0-(x1)^2/42.0)) ) ;        
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0));    
    else if (abs(x) < 0.15)
        // Bern_P1 = ( 1.0-(x1)/2.0+(x1)^2/12.0*(1.0-(x1)^2/60.0*(1.0-(x1)^2/42.0*(1-(x1)^2/40*(1-0.02525252525252525252525*(x1)^2)))));
        result = 1.0 - x/2.0 + pow(x, 2.0)/12.0 * (1.0 - pow(x, 2.0)/60.0 * (1.0 - pow(x, 2.0)/42.0 * (1 - pow(x, 2.0)/40 * (1 - 0.02525252525252525252525*pow(x, 2.0)))));
    else
        result = x / (exp(x) - 1);
    return result;
}

double dB(double x)
{
    double result = 0.0;
    if (abs(x) < 0.0252)
        // Deri_Bern_P1_phi1 = (-0.5 + (x1)/6.0*(1.0-(x1)^2/30.0*(1.0-(x1)^2/28.0)) )/thermal;
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0));
    else if (abs(x) < 0.15)
        // Deri_Bern_P1_phi1 = (-0.5 + (x1)/6.0*(1.0-(x1)^2/30.0*(1.0-(x1)^2/28.0*(1-(x1)^2/30*(1-0.03156565656565656565657*(x1)^2)))))/thermal;
        result = -0.5 + x/6.0 * (1.0 - pow(x, 2.0)/30.0 * (1.0 - pow(x, 2.0)/28.0 * (1 - pow(x, 2.0)/30 * (1 - 0.03156565656565656565657*pow(x, 2.0)))));
    else
        // Deri_Bern_P1_phi1=(1/(exp(x1)-1)-Bern_P1*(1/(exp(x1)-1)+1))/thermal;
        result = 1.0/(exp(x)-1) - B(x)*(1.0 / (exp(x) - 1) + 1);
    return result;
}