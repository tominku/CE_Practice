#include <armadillo> 
#include <iostream> 
#include <sciplot/sciplot.hpp>
#include "util.cpp"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

// #include <fmt/core.h>
// #include <fmt/format.h>
using namespace arma; 

const int Nx = 101;
const int Ny = 21;
const int N = Nx * Ny;
//double n_int = 1.075*1e16; // need to check, constant.cc, permitivity, k_T, epsilon, q, compare 
double n_int = 1.0*1e16;
double T = 300;    
double thermal = k_B * T / q;

double left_part_width = 2e-7;
double total_width = left_part_width*2;
double deltaX = total_width / (Nx-1); // in meter  
double total_height = 0.8e-7;
double deltaY = (total_height) / (Ny-1); // in meter  

#define ijTok(i, j) (Nx*(j-1) + i)
//#define eps_ipj(i, j) ((i+0.5) < Ny ? eps_si : 0)
//#define eps_imj(i, j) ((i-0.5) > 1 ? eps_si : 0)
#define eps_ipj(i, j) eps_si
#define eps_imj(i, j) eps_si
#define eps_ijp(i, j) eps_si
#define eps_ijm(i, j) eps_si
//#define var_at(i, j, var_name, var_center_name) ((j >= 1 && j <= Ny) ? var_name(ijTok(i, j)) : var_center_name)
#define index_exist(i, j) ((j >= 1 && j <= Ny && i >= 1 && i <= Nx) ? true : false)
#define phi_at(i, j, var_name) (index_exist(i, j) ? var_name(ijTok(i, j)) : 0)
#define n_at(i, j, var_name) (index_exist(i, j) ? var_name(N + ijTok(i, j)) : 0)
#define p_at(i, j, var_name) (index_exist(i, j) ? var_name(2*N + ijTok(i, j)) : 0)
#define INCLUDE_VFLUX false

const string subject_name = "PN_2D_SG";

double dop_left = 5e24; // in m^3
// double dop_center = 2e23; // in m^3
//double dop_left = 5e23; // in m^3
double dop_right = -2e24;
//double dop_right = -2e23;

int interface_i = round(left_part_width/deltaX) + 1;
vec one_vector(3*N, fill::ones);

double B(double x);
double dB(double x);

double compute_eq_phi(double doping_density)
{
    double phi = 0;
    if (doping_density > 0)
        phi = thermal * log(doping_density/n_int);
    else            
        phi = - thermal * log(abs(doping_density)/n_int);
    
    return phi;        
}

void r_and_jacobian(vec &r, sp_mat &jac, vec &phi_n_p, double bias)
{
    r.fill(0.0);        
    jac.zeros();           
    
    // set boundary condition
    int i = 0;
    for (int j=1; j<=Ny; ++j)
    {
        // left boundary
        i = 1;      
        int k = ijTok(i, j);
        r(k) = phi_at(i, j, phi_n_p) - compute_eq_phi(dop_left);        
        r(N + k) = n_at(i, j, phi_n_p) - abs(dop_left);        
        r(2*N + k) = p_at(i, j, phi_n_p) - abs(n_int*n_int/dop_left);        
        jac(k, k) = 1.0; 
        jac(N + k, N + k) = 1.0; 
        jac(2*N + k, 2*N + k) = 1.0; 

        // right boundary      
        i = Nx;            
        k = ijTok(i, j);
        r(k) = phi_at(i, j, phi_n_p) - compute_eq_phi(dop_right) - bias;
        r(N + k) = n_at(i, j, phi_n_p) - abs(n_int*n_int/dop_right);
        r(2*N + k) = p_at(i, j, phi_n_p) - abs(dop_right);        
        jac(k, k) = 1.0; 
        jac(N + k, N + k) = 1.0; 
        jac(2*N + k, 2*N + k) = 1.0;                             
    }        
    
    double ion_term = 0.0;

    for (int j=1; j<=Ny; j++)    
    {
        for (int i=(1+1); i<Nx; i++)        
        {               
            int k = ijTok(i, j);
            double phi_ij = phi_at(i, j, phi_n_p);
            double n_ij = n_at(i, j, phi_n_p);
            double p_ij = p_at(i, j, phi_n_p);                 

            if (i < interface_i)                                                    
                ion_term = dop_left;                                                       
            else if (i == interface_i)            
                ion_term = 0.5*(dop_left) + 0.5*(dop_right);                                              
            else if (i > interface_i)            
                ion_term = dop_right;                                                             

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
            // double s_ijp = 0;
            // double s_ijm = 0;
            double V = deltaX*deltaY;

            if (j == 1)      
            {      
                s_ipj *= 0.5;
                s_imj *= 0.5;
                s_ijm = 0;
                V *= 0.5;            
            }
            else if(j == Ny)                  
            {
                s_ipj *= 0.5;
                s_imj *= 0.5;
                s_ijp = 0;
                V *= 0.5;            
            }
            
            double D_ipj = -eps_ipj(i,j) * phi_diff_ipi / deltaX;
            double D_imj = -eps_imj(i,j) * phi_diff_iim / deltaX;                                        

            // Residual for the Poisson Equation
            if (INCLUDE_VFLUX)
            {
                double D_ijp = -eps_ijp(i,j) * phi_diff_jpj / deltaY;
                double D_ijm = -eps_ijm(i,j) * phi_diff_jjm / deltaY;        
                r(k) = s_ipj*D_ipj + s_imj*D_imj + s_ijp*D_ijp + s_ijm*D_ijm;            
            }
            else
                r(k) = s_ipj*D_ipj + s_imj*D_imj;

            r(k) -= V*q*(ion_term - n_ij + p_ij);             
            r(k) /= eps_0;
            
            //r(k) *= deltaX;
            // Jacobian for the Poisson Equation
            if (INCLUDE_VFLUX)
            {
                jac(k, k) = s_ipj*eps_ipj(i, j)/deltaX - s_imj*eps_imj(i, j)/deltaX +
                    s_ijp*eps_ijp(i, j)/deltaY - s_ijm*eps_ijm(i, j)/deltaY;            
            }
            else
                jac(k, k) = s_ipj*eps_ipj(i, j)/deltaX - s_imj*eps_imj(i, j)/deltaX;                
            jac(k, k) /= eps_0;
            
            jac(k, ijTok(i+1, j)) = - s_ipj*eps_ipj(i, j) / deltaX;                        
            jac(k, ijTok(i-1, j)) = s_imj*eps_imj(i, j) / deltaX;   
            jac(k, ijTok(i+1, j)) /= eps_0;
            jac(k, ijTok(i-1, j)) /= eps_0;
            
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))
                {
                    jac(k, ijTok(i, j+1)) = - s_ijp*eps_ijp(i, j) / deltaY;
                    jac(k, ijTok(i, j+1)) /= eps_0;
                }
                if (index_exist(i, j-1))                        
                {
                    jac(k, ijTok(i, j-1)) = s_ijm*eps_ijm(i, j) / deltaY;                                                                      
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
                r(N + k) = s_ipj*Jn_ipj + s_imj*Jn_imj + s_ijp*Jn_ijp + s_ijm*Jn_ijm;
            }            
            else
                r(N + k) = s_ipj*Jn_ipj + s_imj*Jn_imj;

            // Jacobian for the SG (n)
            // w.r.t. phis
            double a, b, c, d;
            jac(N + k, ijTok(i+1, j)) = a = s_ipj*(n_ij*dB(-phi_diff_ipi/thermal) + n_ipj*dB(phi_diff_ipi/thermal)) / thermal;
            jac(N + k, ijTok(i-1, j)) = b = s_imj*(-n_imj*dB(-phi_diff_iim/thermal) - n_ij*dB(phi_diff_iim/thermal)) / thermal;
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))                        
                    jac(N + k, ijTok(i, j+1)) = c = s_ijp*(n_ij*dB(-phi_diff_jpj/thermal) + n_ijp*dB(phi_diff_jpj/thermal)) / thermal;
                if (index_exist(i, j-1))                        
                    jac(N + k, ijTok(i, j-1)) = d = s_ijm*(-n_ijm*dB(-phi_diff_jjm/thermal) - n_ij*dB(phi_diff_jjm/thermal)) / thermal;
                jac(N + k, ijTok(i, j)) = -a-b-c-d;
            }
            else
                jac(N + k, ijTok(i, j)) = -a-b;
            // w.r.t. ns
            jac(N + k, N + ijTok(i+1, j)) = s_ipj*B(phi_diff_ipi/thermal);
            jac(N + k, N + ijTok(i-1, j)) = -s_imj*B(-phi_diff_iim/thermal);
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))                        
                    jac(N + k, N + ijTok(i, j+1)) = s_ijp*B(phi_diff_jpj/thermal);
                if (index_exist(i, j-1))                        
                    jac(N + k, N + ijTok(i, j-1)) = -s_ijm*B(-phi_diff_jjm/thermal);
                jac(N + k, N + ijTok(i, j)) = - s_ipj*B(-phi_diff_ipi/thermal) + s_imj*B(phi_diff_iim/thermal) - s_ijp*B(-phi_diff_jpj/thermal) + s_ijm*B(phi_diff_jjm/thermal);
            }
            else
                jac(N + k, N + ijTok(i, j)) = - s_ipj*B(-phi_diff_ipi/thermal) + s_imj*B(phi_diff_iim/thermal);        

            // Residual for the SG (p)    
            double Jp_ipj = -p_ipj*B(-phi_diff_ipi/thermal) + p_ij*B(phi_diff_ipi/thermal);
            double Jp_imj = -p_ij*B(-phi_diff_iim/thermal) + p_imj*B(phi_diff_iim/thermal);                        
            if (INCLUDE_VFLUX)
            {
                double Jp_ijp = -p_ijp*B(-phi_diff_jpj/thermal) + p_ij*B(phi_diff_jpj/thermal);
                double Jp_ijm = -p_ij*B(-phi_diff_jjm/thermal) + p_ijm*B(phi_diff_jjm/thermal);                
                r(2*N + k) = s_ipj*Jp_ipj + s_imj*Jp_imj + s_ijp*Jp_ijp + s_ijm*Jp_ijm;                    
            }
            else
                r(2*N + k) = s_ipj*Jp_ipj + s_imj*Jp_imj;

            // Jacobian for the SG (p)
            // w.r.t. phis
            jac(2*N + k, ijTok(i+1, j)) = a = s_ipj*(p_ij*dB(phi_diff_ipi/thermal) + p_ipj*dB(-phi_diff_ipi/thermal)) / thermal;                      
            jac(2*N + k, ijTok(i-1, j)) = b = s_imj*(- p_imj*dB(phi_diff_iim/thermal) - p_ij*dB(-phi_diff_iim/thermal)) / thermal;
            if (INCLUDE_VFLUX)
            {
                if (index_exist(i, j+1))                        
                    jac(2*N + k, ijTok(i, j+1)) = c = s_ijp*(p_ij*dB(phi_diff_jpj/thermal) + p_ijp*dB(-phi_diff_jpj/thermal)) / thermal;
                if (index_exist(i, j-1))                        
                    jac(2*N + k, ijTok(i, j-1)) = d = s_ijm*(- p_ijm*dB(phi_diff_jjm/thermal) - p_ij*dB(-phi_diff_jjm/thermal)) / thermal;
                jac(2*N + k, ijTok(i, j)) = -a-b-c-d;
            }
            else
                jac(2*N + k, ijTok(i, j)) = -a-b;
            // w.r.t. ps
            jac(2*N + k, 2*N + ijTok(i+1, j)) = -s_ipj*B(-phi_diff_ipi/thermal);            
            jac(2*N + k, 2*N + ijTok(i-1, j)) = s_imj*B(phi_diff_iim/thermal);   
            if (INCLUDE_VFLUX)
            {         
                if (index_exist(i, j+1))                        
                    jac(2*N + k, 2*N + ijTok(i, j+1)) = -s_ijp*B(-phi_diff_jpj/thermal);
                if (index_exist(i, j-1))                        
                    jac(2*N + k, 2*N + ijTok(i, j-1)) = s_ijm*B(phi_diff_jjm/thermal);
                jac(2*N + k, 2*N + ijTok(i, j)) = s_ipj*B(phi_diff_ipi/thermal) - s_imj*B(-phi_diff_iim/thermal) + s_ijp*B(phi_diff_jpj/thermal) - s_ijm*B(-phi_diff_jjm/thermal);
            }
            else
                jac(2*N + k, 2*N + ijTok(i, j)) = s_ipj*B(phi_diff_ipi/thermal) - s_imj*B(-phi_diff_iim/thermal);
        }      
    }
}

vec solve_phi(double boundary_potential, vec &phi_n_p_0, sp_mat &C)
{                
    int num_iters = 30;    
    printf("boundary voltage: %f \n", boundary_potential);
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
        r_and_jacobian(r, jac, phi_n_p_k, boundary_potential);   
                                      
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

    std::string convergence_file_name = fmt::format("{}_conv_{:.2f}.csv", subject_name, boundary_potential);
    log_deltas.save(convergence_file_name, csv_ascii);            
        
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


void fill_initial(vec &phi, string method)
{        
    double phi_1 = compute_eq_phi(dop_left);
    double phi_Nx = compute_eq_phi(dop_right);
    for (int j=1; j<=Ny; j++)
    {
        for (int i=1; i<=Nx; i++)
        { 
            int k = ijTok(i, j);            
            if (i==1)  
                phi(k) = phi_1;
            else if (i==Nx)                            
                phi(k) = phi_Nx;
            else
            {
                if (method.compare("uniform") == 0)
                {
                    if (i <= interface_i)                
                        phi(k) = phi_1;
                    else
                        phi(k) = phi_Nx;                            
                }
                else if (method.compare("linear") == 0)
                {
                    phi(k) = phi_1 + (phi_Nx - phi_1) * ((i-1)/(Nx-1));
                }
                else if (method.compare("random") == 0)
                {
                    double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                    phi(k) = r;
                }
            }
        }
     }      
}


int main() {    

    double bias = -0.4;    
    vec phi_n_p_0(3*N+1, arma::fill::zeros);
    //fill_initial(phi_n_p_0, "uniform");
    //fill_initial(phi_n_p_0, "random");
    //fill_initial(phi_n_p_0, "linear");
    bool load_initial_solution_from_NP = true;    
    if (load_initial_solution_from_NP)
    {
        string subject_name = "PN_2D_NP";
        std::string file_name = fmt::format("{}_phi_{:.2f}.csv", subject_name, 0.0); 
        cout << file_name << "\n";        
        vec phi_from_NP(N, fill::zeros);
        phi_from_NP.load(file_name);

        file_name = fmt::format("{}_eDensity_{:.2f}.csv", subject_name, 0.0); 
        cout << file_name << "\n";        
        vec eDensity_from_NP(N, fill::zeros);
        eDensity_from_NP.load(file_name);        

        file_name = fmt::format("{}_holeDensity_{:.2f}.csv", subject_name, 0.0); 
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
        C(m, m) = abs(dop_left);        
    for (int m=2*N; m<3*N; m++)
        C(m, m) = abs(dop_left);                

    //for (int i=0; i<10; i++)
    {        
        int i = 0;
        vec result = solve_phi(bias + (0.1*i), phi_n_p_0, C); 
        phi_n_p_0 = result;   
        
        std::string log = fmt::format("BD {:.2f} V \n", bias + (0.1*i));            
        cout << log;        
                   
        vec phi = phi_n_p_0(span(1, N));                
        vec eDensity = phi_n_p_0(span(N+1, 2*N));        
        eDensity /= 1e6;
        vec holeDensity = phi_n_p_0(span(2*N+1, 3*N));        
        holeDensity /= 1e6;
        
        std::string phi_file_name = fmt::format("{}_phi_{:.2f}.csv", subject_name, bias+(0.1*i));
        phi.save(phi_file_name, csv_ascii);                

        std::string n_file_name = fmt::format("{}_eDensity_{:.2f}.csv", subject_name, bias+(0.1*i));
        eDensity.save(n_file_name, csv_ascii);                

        std::string h_file_name = fmt::format("{}_holeDensity_{:.2f}.csv", subject_name, bias+(0.1*i));
        holeDensity.save(h_file_name, csv_ascii);                                     

        //save_current_densities(phi_n);
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