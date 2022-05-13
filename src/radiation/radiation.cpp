//
// Created by owen on 3/19/22.
//
#include "radiation.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include "environment/runEnvironment.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

ablate::radiation::RadiationSolver::RadiationSolver(std::string solverId, std::shared_ptr<domain::Region> region, int rayNumber,
                                                    std::shared_ptr<parameters::Parameters> options)
    : CellSolver(std::move(solverId), std::move(region),  std::move(options)) {
    raynumber = rayNumber;
    nTheta = raynumber; //The DEFAULT number of angles to solve with, should be given by user input probably?
    nPhi = 2 * raynumber; //The DEFAULT number of angles to solve with, should be given by user input
}

ablate::radiation::RadiationSolver::~RadiationSolver() {
}

void ablate::radiation::RadiationSolver::Setup() {  // allows initialization after the subdomain and dm is established
    ablate::solver::CellSolver::Setup();
    dim = subDomain->GetDimensions();  // Number of dimensions already defined in the setup

}
void ablate::radiation::RadiationSolver::Initialize() {
    // this->reallySolveParallelPlates();
    this->RayInit();  // Runs the ray initialization finding cell indices
    /*this->RayTrace(0);
    for (int i = 1; i < 20; i++) {
        this->RayProduct(0, i);
    }*/

}

/// Declaring function for the initialization to call, draws each ray vector and gets all of the cells associated with it (sorted by distance and starting at the boundary working in)
void ablate::radiation::RadiationSolver::RayInit() {
    //ablate::solver::CellSolver::Setup();

    double theta;  // represents the actual current angle (inclination)
    double phi;    // represents the actual current angle (rotation)

    /// Locally get a range of cells that are included in this subdomain at this time step for the ray initialization
    // March over each cell in this region to create the stencil
    // IS cellIS;
    // PetscInt cStart, cEnd;  // Keep these as local variables, the cell indices in the radiation initializer should be updated with every radiation initialization. The variables here are owned by the
                            // DM
    stencilSet.clear(); //Clear the existing set because a new set of cells will be created below

    solver::Range cellRange;
    GetCellRange(cellRange);
    // std::set<PetscInt> stencilSet; //Moved to a class variable
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // keep a list of cells in the stencil
        stencilSet.insert(cell);
    }

    /// Create a nested vector which can store cell locations based on origin cell, theta, phi, and space step
    std::vector<PetscInt> rayPhis;                                // Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
    std::vector<std::vector<PetscInt>> rayThetas(nPhi, rayPhis);  // Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
    std::vector<std::vector<std::vector<PetscInt>>> rayCells(
        nTheta, rayThetas);                    // Make vector to store this dimensional row //Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
    rays.resize(stencilSet.size(), rayCells);  // Indices: Cell, angle (theta), angle(phi), space steps

    /// Get setup things for the position vector of the current cell index
    const PetscScalar* cellGeomArray;  // Declare the variables that will contain the geometry of the cells
    PetscReal minCellRadius;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, &minCellRadius);  // Obtain the geometric information about the cells in the DM?
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    PetscFVCellGeom* cellGeom;

    PetscInt ncells = 0;
    for (auto iCell : stencilSet) {
        // double percentComplete = 100.0 * double(ncells) / double(stencilSet.size());
        // PetscPrintf(PETSC_COMM_WORLD, "Radiation Initializer Percent Complete: %f\n", percentComplete);

        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  // Reads the cell location from the current cell

        /// Set the spatial step size to the minimum cell radius
        h = minCellRadius;

        for (int ntheta = 1; ntheta < nTheta; ntheta++) {  // for every angle theta
            // precalculate sin and cosine of the angle theta because it is used frequently?
            for (int nphi = 0; nphi < nPhi; nphi++) {  // for every angle phi
                PetscReal magnitude = h;
                // Should represent the distance from the origin cell to the boundary. How to get this? By marching out of the domain! (and checking whether we are still inside)
                int nsteps = 0;  // Number of spatial steps that  the ray has taken towards the origin
                bool boundary = false;  // Keeps track of whether the ray has intersected the boundary at this point or not
                while (!boundary) {
                    /// Draw a point on which to grab the cells of the DM
                    Vec intersect;                                   // Intersect should point to the boundary, and then be pushed back to the origin, getting each cell
                    theta = ((double)ntheta / (double)nTheta) * pi;  // converts the present angle number into a real angle
                    phi = ((double)nphi / (double)nPhi) * 2.0 * pi;  // converts the present angle number into a real angle
                    PetscInt i[3] = {0, 1, 2};
                    //[dim] is not known so compiler doesn't like

                    PetscReal direction[3] = {
                        (magnitude * sin(theta) * cos(phi)) + cellGeom->centroid[0],  // x component conversion from spherical coordinates, adding the position of the current cell
                        (magnitude * sin(theta) * sin(phi)) + cellGeom->centroid[1],  // y component conversion from spherical coordinates, adding the position of the current cell
                        (magnitude * cos(theta)) + cellGeom->centroid[2]};            // z component conversion from spherical coordinates, adding the position of the current cell
                    //(Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction

                    /// This block creates the vector pointing to the cell whose index will be stored during the current loop

                    VecCreate(PETSC_COMM_WORLD, &intersect);  // Instantiates the vector
                    VecSetBlockSize(intersect, dim) >> checkError;
                    VecSetSizes(intersect, PETSC_DECIDE, dim);  // Set size
                    VecSetFromOptions(intersect);
                    VecSetValues(intersect, dim, i, direction, INSERT_VALUES);  // Actually input the values of the vector (There are 'dim' values to input)

                    /// Loop through points to try to get a list (vector) of cells that are sitting on that line
                    // Use std vector to store points (build vector outside of while loop?) (or use a set prevents index duplication in the list?)
                    PetscSF cellSF = nullptr;  // PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
                    DMLocatePoints(cellDM, intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  // Locate the points in v in the mesh and return a PetscSF of the containing cells
                    /// An array that maps each point to its containing cell can be obtained with the below
                    // We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
                    PetscInt nFound;
                    const PetscInt* point = nullptr;
                    const PetscSFNode* cell = NULL;
                    PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  // Using this to get the petsc int cell number from the struct (SF)



                    /// IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
                    if (nFound > -1) {
                        /// This function returns multiple values if multiple points are input to it
                        if (cell[0].index >= 0) {
                            if (stencilSet.count(cell[0].index) != 0) { //Make sure that whatever cell is returned is in the stencil set (and not outside of the radiation domain)
                                /// Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
                                rays[ncells][ntheta][nphi].push_back(cell[0].index);
                                // rayPhis.push_back(cell[0].index);
                                // PetscPrintf(PETSC_COMM_WORLD, "Intersect x: %f, y: %f, z: %f Value: %i\n", direction[0], direction[1], direction[2], cell[0].index);
                                // PetscPrintf(PETSC_COMM_WORLD, "Cell: %i, nTheta: %i, Theta: %g, nPhi: %i, Phi: %g, Value: %i\n", iCell, ntheta, theta, nphi, phi, cell[p].index);
                            } else {
                                boundary = true;
                            }
                        } else {
                            boundary = true;
                        }
                    } else {
                        boundary = true;
                    }

                    /// Increase the step size of the ray toward the boundary by one more minimum cell radius
                    magnitude += h;  // Increase the magnitude of the ray tracing vector by one space step
                    nsteps++;        // Increase the number of steps taken, informs how many indices the vector has

                    // Cleanup
                    PetscSFDestroy(&cellSF);
                    VecDestroy(&intersect);
                }
            }
        }
        ncells++;
    }
    PetscPrintf(PETSC_COMM_WORLD, "Finished!\n");
}

void ablate::radiation::RadiationSolver::RayTrace(PetscReal time) {  /// Gets the total intensity/radiative gain at a single cell

    /// Get setup things for the position vector of the current cell index
    const PetscScalar* cellGeomArray;  // Declare the variables that will contain the geometry of the cells
    PetscReal minCellRadius;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, &minCellRadius);  // Obtain the geometric information about the cells in the DM?
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    PetscFVCellGeom* cellGeom;

    /// For ABLATE implementation, get temperature based on this function
    const auto& temperatureField = subDomain->GetField("temperature");
    PetscScalar* temperatureArray = nullptr;
    subDomain->GetFieldLocalVector(temperatureField, time, &vis, &loctemp, &vdm);
    VecGetArray(loctemp, &temperatureArray);                                                     // Get the array that lives inside the vector

    /// Declare the basic information
    // std::vector<PetscReal> radGain(stencilSet.size(), 0); //Represents the total final radiation intensity (for every cell, index as index)
    PetscReal temperature;  // The temperature at any given location
    PetscReal dTheta = pi / nTheta;
    PetscReal dPhi = (2 * pi) / nPhi;
    double kappa = 1;  // Absorptivity coefficient, property of each cell
    double theta;

    /// Get the file path for the output
    std::filesystem::path radOutput = environment::RunEnvironment::Get().GetOutputDirectory() / "originalMethod.txt";
    std::ofstream stream(radOutput);

    std::vector<std::vector<PetscReal>> locations;  // 2 Dimensional vector which stores the locations of the cell centers
    PetscInt ncells = 0;

    for (auto iCell : stencilSet) {  // loop through subdomain cell indices
        /// Print location of the current cell
        for (int i = 0; i < dim; i++) {  // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
            DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  // Reads the cell location from the point in question
            stream << cellGeom->centroid[i];
            //Print the value of the centroid to a 3 (4) column txt file of x,y,z, (ray number ?). (Ultimately these values will need to be stored in program or used)
            stream << " ";
        }
        PetscReal intensity = 0;
        for (int ntheta = 0; ntheta < nTheta; ntheta++) {    // for every angle theta
            theta = ((double)ntheta / (double)nTheta) * pi;  // converts the present angle number into a real angle
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /// Each ray is born here. They begin at the far field temperature.
                /// Initial ray intensity should be set based on which boundary it is coming from.

                /// If the ray originates from the walls, then set the initial ray intensity to the wall temperature
                PetscReal rayIntensity;                      // = FlameIntensity(1, refTemp); //Initialize the ray intensity as the far fie0ld flame intensity
                if (theta > pi / 2) {                        // If sitting on the bottom boundary (everything on the lower half of the angles)
                    rayIntensity = FlameIntensity(1, 1300);  // Set the initial ray intensity to the bottom wall intensity
                } else if (theta < pi / 2) {                 // If sitting on the top boundary
                    rayIntensity = FlameIntensity(1, 700);   // Set the initial ray intensity to the top wall intensity
                }

                int numPoints = static_cast<int>(rays[ncells][ntheta][nphi].size());
                for (int n = (numPoints - 1); n >= 0; n--) {  /// Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                    std::vector<PetscReal> loc(3, 0);
                    DMPlexPointLocalRead(cellDM, rays[ncells][ntheta][nphi][n], cellGeomArray, &cellGeom);  // Reads the cell location from the point in question
                    for (int i = 0; i < dim; i++) {  // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
                        loc[i] = cellGeom->centroid[i];  // Get the location of the current cell and export it to the rest of the function
                    }

                    /// Define the absorptivity and temperature in this section
                    // DMPlexPointLocalRef(subDomain->GetDM(), rays[ncells][ntheta][nphi][n], temperatureArray, &temperature);  // Gets the temperature from the cell index specified

                    ///Get the local temperature at every point along the ray. For the analytical solution, the temperature at each point will be based on the spatial location of the cell.
                    if(loc[2] <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                        temperature = -6.349E6*loc[2]*loc[2] + 2000.0;
                    }else{
                        temperature = -1.179E7*loc[2]*loc[2] + 2000.0;
                    }

                    // TODO: Input absorptivity (kappa) values from model here.

                    /// The ray intensity changes as a function of the environment at this point
                    rayIntensity = FlameIntensity(1 - exp(-kappa * h), temperature) + rayIntensity * exp(-kappa * h);
                    // PetscPrintf(PETSC_COMM_WORLD, "Intensity: %f\n", rayIntensity);
                }
                /// The rays end here, their intensity is added to the total intensity of the cell
                intensity += rayIntensity * sin(theta) * dTheta * dPhi;  // Gives the partial impact of the ray on the total sphere. The sin(theta) is a result of the polar coordinate discretization
            }
            // PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        }

        //Total energy gain of the current cell depends on absorptivity at the current cell
        PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        /// Print the radiative gain at each cell
        stream << intensity;
        stream << "\n";

        ncells++;
        // Cleanup
        VecRestoreArray(loctemp, &temperatureArray);
    }
    stream.close();
}

PetscReal ablate::radiation::RadiationSolver::ReallySolveParallelPlates(PetscReal z) {  ///Analytical solution of a special verification case.
    /// Define variables and basic information
    PetscReal G;
    PetscReal IT = FlameIntensity(1, 700);   // Intensity of rays originating from the top plate
    PetscReal IB = FlameIntensity(1, 1300);  // Set the initial ray intensity to the bottom wall intensity //Intensity of rays originating from the bottom plate
    PetscReal kappa = 1;                     // Kappa is not spatially dependant in this special case
    PetscReal zBottom = -0.0105;             // Prescribe the top and bottom heights for the domain
    PetscReal zTop = 0.0105;

    PetscReal temperature;
    PetscReal Ibz;

    PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    const PetscReal sbc = 5.6696e-8;
    PetscReal nZp = 1000;
    Ibz = 0;

    std::vector<PetscReal> Iplus;
    std::vector<PetscReal> Iminus;

    for (double nzp = 1; nzp < (nZp - 1); nzp++) {             /// Plus integral goes from bottom to Z
        PetscReal zp = zBottom + (nzp / nZp) * (z - zBottom);  // Calculate the z height
        /// Get the temperature
        if (zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
            temperature = -6.349E6 * zp * zp + 2000.0;
        } else {
            temperature = -1.179E7 * zp * zp + 2000.0;
        }
        /// Get the black body intensity here
        Ibz = FlameIntensity(1, temperature);
        Iplus.push_back(Ibz * EInteg(1, kappa * (z - zp)));
    }
    for (double nzp = 1; nzp < (nZp - 1); nzp++) {    /// Minus integral goes from z to top
        PetscReal zp = z + (nzp / nZp) * (zTop - z);  // Calculate the zp height
        /// Get the temperature
        if (zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
            temperature = -6.349E6 * zp * zp + 2000.0;
        } else {
            temperature = -1.179E7 * zp * zp + 2000.0;
        }
        /// Get the black body intensity here
        Ibz = FlameIntensity(1, temperature);
        Iminus.push_back(Ibz * EInteg(1, kappa * (zp - z)));
    }

    PetscReal term1 = IB * EInteg(2, kappa * (z - zBottom));
    PetscReal term2 = IT * EInteg(2, kappa * (zTop - z));
    PetscReal term3 = CSimp(zBottom, z, Iplus);
    PetscReal term4 = CSimp(z, zTop, Iminus);

    G = 2 * pi * (term1 + term2 + term3 + term4);

    ///Now compute the losses at the given input point (this is in order to match the output that is given by the ComputeRHSFunction)
    if (z <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
        temperature = -6.349E6 * z * z + 2000.0;
    } else {
        temperature = -1.179E7 * z * z + 2000.0;
    }
    PetscReal losses = 4 * sbc * temperature * temperature * temperature * temperature;
    PetscReal radTotal =  -kappa * (losses - G);


    return radTotal;
}

PetscReal ablate::radiation::RadiationSolver::EInteg(int order, double x) {
    if (x == 0 && order != 1) return 1 / (order - 1);  // Simple solution in this case, exit
    std::vector<PetscReal> En;
    double N = 100;
    for (double n = 1; n < N; n++) {
        double mu = n / N;
        if (order == 1) {
            En.push_back(exp(-x / mu) / mu);
        }
        if (order == 2) {
            En.push_back(exp(-x / mu));
        }
        if (order == 3) {
            En.push_back(exp(-x / mu) * mu);
        }
    }
    PetscReal final = CSimp(0, 1, En);
    return final;
}

PetscReal ablate::radiation::RadiationSolver::FlameIntensity(double epsilon, double temperature) {  /// Gets the flame intensity based on temperature and emissivity
    const PetscReal sbc = 5.6696e-8;                                                                // Stefan-Boltzman Constant (J/K)
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
}

PetscReal ablate::radiation::RadiationSolver::CSimp(PetscReal a, PetscReal b, std::vector<double> f) {
    /// b-a represents the size of the total domain that is being integrated over
    // PetscReal b = H; //End
    // PetscReal a = 0; //Beginning
    PetscReal I;
    PetscReal n = static_cast<double>(f.size());  // The number of elements in the vector that is being integrated over
    int margin = 0;

    // PetscReal h = (b-a)/n; //Step size
    PetscReal f_sum = 0;  // Initialize the sum of all middle elements

    if (a != b) {
        /// Loop through every point except the first and last
        for (int i = margin; i < (n - margin); i++) {
            if (i % 2 == 0) {
                f[i] = 2 * f[i];  // Weight lightly on the border
            } else {
                f[i] = 4 * f[i];  // Weight heavily in the center
            }
            f_sum += f[i];  // Add this value to the total every time
        }
        // I = ((b - a) / (3 * n)) * (f_sum);  // The ends make it upset, just don't include them :3
        I = ((b - a) / (3 * n)) * (f[0] + f_sum + f[n]);  // Compute the total final integral
    } else {
        I = 0;
    }
    return I;
}

PetscErrorCode ablate::radiation::RadiationSolver::RayProduct(PetscReal time, PetscInt segSteps) {
    /// Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable

    /// Get setup things for the position vector of the current cell index
    const PetscScalar* cellGeomArray;  // Declare the variables that will contain the geometry of the cells
    PetscReal minCellRadius;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, &minCellRadius);  // Obtain the geometric information about the cells in the DM?
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    PetscFVCellGeom* cellGeom;

    //const auto& eulerFieldInfo = subDomain->GetField("euler");

    /// Get the temperature of the current cell (in order to compute the losses)
    // For ABLATE implementation, get temperature based on this function
    const auto& temperatureField = subDomain->GetField("temperature");
    PetscScalar* temperatureArray = nullptr;
    subDomain->GetFieldLocalVector(temperatureField, time, &vis, &loctemp, &vdm);
    VecGetArray(loctemp, &temperatureArray);

    /// Declare the basic information
    // std::vector<PetscReal> radGain(stencilSet.size(), 0); //Represents the total final radiation intensity (for every cell, index as index)
    PetscReal temperature;  // The temperature at any given location
    PetscReal dTheta = pi / nTheta;
    PetscReal dPhi = (2 * pi) / nPhi;
    double kappa = 1;  // Absorptivity coefficient, property of each cell
    double theta;

    ///Things for this function in particular
    PetscReal Io;
    PetscReal Ib;

    /// Get the file path for the output
    std::filesystem::path radOutput = environment::RunEnvironment::Get().GetOutputDirectory() / std::to_string(segSteps);
    std::ofstream stream(radOutput);

    std::vector<std::vector<PetscReal>> locations;  // 2 Dimensional vector which stores the locations of the cell centers
    PetscInt ncells = 0;

    for (auto iCell : stencilSet) {  // loop through subdomain cell indices
        /// Print location of the current cell
        for (int i = 0; i < dim; i++) {  // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
            DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  // Reads the cell location from the point in question
            stream << cellGeom->centroid[i];
            //Print the value of the centroid to a 3 (4) column txt file of x,y,z, (ray number ?). (Ultimately these values will need to be stored in program or used)
            stream << " ";
        }
        PetscReal intensity = 0;
        for (int ntheta = 0; ntheta < nTheta; ntheta++) {    // for every angle theta
            theta = ((double)ntheta / (double)nTheta) * pi;  // converts the present angle number into a real angle
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /// Each ray is born here. They begin at the far field temperature.
                /// Initial ray intensity should be set based on which boundary it is coming from.
                /// If the ray originates from the walls, then set the initial ray intensity to the wall temperature, etc.
                PetscReal rayIntensity;
                PetscReal rayIntensitySegment = 1; //So that it can be multiplied and become the first value
                int numPoints = static_cast<int>(rays[ncells][ntheta][nphi].size());
                for (int n = (numPoints - 1); n >= 0; n--) {  /// Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                    std::vector<PetscReal> loc(3, 0);
                    for (int i = 0; i < dim; i++) {  // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
                                                                                                                            //   PetscFVCellGeom* cellGeom;
                        DMPlexPointLocalRead(cellDM, rays[ncells][ntheta][nphi][n], cellGeomArray, &cellGeom);  // Reads the cell location from the point in question

                        loc[i] = cellGeom->centroid[i];  // Get the location of the current cell and export it to the rest of the function
                    }

                    /// Define the absorptivity and temperature in this section
                    DMPlexPointLocalRef(vdm, rays[ncells][ntheta][nphi][n], temperatureArray, &temperature);  // Gets the temperature from the cell index specified

                    ///Get the local temperature at every point along the ray. For the analytical solution, the temperature at each point will be based on the spatial location of the cell.
                    if(loc[2] <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                        temperature = -6.349E6*loc[2]*loc[2] + 2000.0;
                    }else{
                        temperature = -1.179E7*loc[2]*loc[2] + 2000.0;
                    }

                    if (n == (numPoints - 1)) {
                        rayIntensity = FlameIntensity(1, temperature);  // Set the initial ray intensity to the boundary intensity
                        Io = rayIntensity; //Set the originating ray intensity
                    }else {
                        /// The ray intensity changes as a function of the environment at this point
                        rayIntensity = FlameIntensity(1 - exp(-kappa * h), temperature) + rayIntensity * exp(-kappa * h);
                    }
                    //TODO: The rays should be split into segments and evaluated in parts.
                    if (n % segSteps == 0) { //If the step number is at the end of a segment of 5 steps
                        // TODO: This is where the ray segments are step-multiplied together (PI product thing)
                        rayIntensitySegment *= rayIntensity; //Multiply the ray intensity again.
                    }
                }
                //TODO: Finally, the ray is given its end intensity based on the guess of Ib. (The rest of that equation)
                rayIntensity = Ib - (Ib - Io)*rayIntensitySegment;
                /// The rays end here, their intensity is added to the total intensity of the cell
                intensity += rayIntensity * sin(theta) * dTheta * dPhi;  // Gives the partial impact of the ray on the total sphere. The sin(theta) is a result of the polar coordinate discretization
            }
            // PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        }
        /// Gets the temperature from the cell index specified
        DMPlexPointLocalRef(vdm, iCell, temperatureArray, &temperature);

        /// Put the irradiation into the right hand side function (net radiation)
        //PetscScalar* rhsValues;
        //DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        //rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += kappa * (4 * sbc * *temperature * *temperature * *temperature * *temperature - intensity);

        //Total energy gain of the current cell depends on absorptivity at the current cell
        PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        /// Print the radiative gain at each cell
        stream << intensity;
        stream << "\n";
        ncells++;

    }
    // Cleanup
    //VecRestoreArrayRead(rhs, &rhsArray);
    VecRestoreArray(loctemp, &temperatureArray);
    //VecRestoreArray(cellGeomVec, &cellGeomArray);
    stream.close();
    PetscFunctionReturn(0);
}

/// End of the added radiation stuff
PetscErrorCode ablate::radiation::RadiationSolver::ComputeRHSFunction(PetscReal time, Vec locXVec,
                                                                      Vec rhs) {  // main interface for integrating in time Inputs: local vector, x vector (current solution), local f vector
    PetscFunctionBeginUser;                                                           // gets fields out of the main vector

    /// Get setup things for the position vector of the current cell index
    /* const auto& errorField = subDomain->GetField("error");
    PetscScalar* errorArray = nullptr;
    const PetscScalar* cellGeomArray;  // Declare the variables that will contain the geometry of the cells
    subDomain->GetFieldLocalVector(errorField, time, &eis, &errors, &edm);
    VecGetArray(errors,&errorArray);
    PetscReal minCellRadius;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, &minCellRadius);  // Obtain the geometric information about the cells in the DM?
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    PetscFVCellGeom* cellGeom; */

    /// Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);

    const auto& eulerFieldInfo = subDomain->GetField("euler");

    /// Get the temperature field
    // For ABLATE implementation, get temperature based on this function
    // For ABLATE implementation, get temperature based on this function
    const auto& temperatureField = subDomain->GetField("temperature");
    PetscScalar* temperatureArray = nullptr;
    subDomain->GetFieldLocalVector(temperatureField, time, &vis, &loctemp, &vdm);
    VecGetArray(loctemp, &temperatureArray);

    /// Declare the basic information
    // std::vector<PetscReal> radGain(stencilSet.size(), 0); //Represents the total final radiation intensity (for every cell, index as index)
    PetscReal* temperature;  // The temperature at any given location
    PetscReal dTheta = pi / nTheta;
    PetscReal dPhi = (2 * pi) / nPhi;
    double kappa = 1;  // Absorptivity coefficient, property of each cell
    double theta;

    std::vector<std::vector<PetscReal>> locations;  // 2 Dimensional vector which stores the locations of the cell centers
    PetscInt ncells = 0;

    for (auto iCell : stencilSet) {  /// loop through subdomain cell indices
        PetscReal intensity = 0;
        for (int ntheta = 0; ntheta < nTheta; ntheta++) {    // for every angle theta
            theta = ((double)ntheta / (double)nTheta) * pi;  // converts the present angle number into a real angle
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /// Each ray is born here. They begin at the far field temperature.
                /// Initial ray intensity should be set based on which boundary it is coming from.
                /// If the ray originates from the walls, then set the initial ray intensity to the wall temperature, etc.
                PetscReal rayIntensity;
                int numPoints = static_cast<int>(rays[ncells][ntheta][nphi].size());

                for (int n = (numPoints - 1); n >= 0; n--) {  /// Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                    /// Define the absorptivity and temperature in this section
                    // For ABLATE implementation, get temperature based on this function                     // Get the array that lives inside the vector
                    DMPlexPointLocalRef(vdm, rays[ncells][ntheta][nphi][n], temperatureArray, &temperature);  // Gets the temperature from the cell index specified

                    // TODO: Input absorptivity (kappa) values from model here.

                    if (n == (numPoints - 1)) {
                        rayIntensity = FlameIntensity(1, *temperature);  // Set the initial ray intensity to the boundary intensity
                        // TODO: Make intensity boundary conditions be set from boundary label
                        /// For debugging purposes VVV
                        /*if (theta > pi / 2) {                        // If sitting on the bottom boundary (everything on the lower half of the angles)
                            rayIntensity = FlameIntensity(1, 1300);  // Set the initial ray intensity to the bottom wall intensity
                        } else if (theta < pi / 2) {                 // If sitting on the top boundary
                            rayIntensity = FlameIntensity(1, 700);   // Set the initial ray intensity to the top wall intensity
                        }*/
                    } else {
                        /// The ray intensity changes as a function of the environment at this point
                        rayIntensity = FlameIntensity(1 - exp(-kappa * h), *temperature) + rayIntensity * exp(-kappa * h);
                        // PetscPrintf(PETSC_COMM_WORLD, "Intensity: %f\n", rayIntensity);
                    }
                }
                /// The rays end here, their intensity is added to the total intensity of the cell
                intensity += rayIntensity * sin(theta) * dTheta * dPhi;  // Gives the partial impact of the ray on the total sphere. The sin(theta) is a result of the polar coordinate discretization
            }
        }
        /// Gets the temperature from the cell index specified
        DMPlexPointLocalRef(vdm, iCell, temperatureArray, &temperature);

        /// Put the irradiation into the right hand side function
        PetscScalar* rhsValues;
        DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        PetscReal losses = 4 * sbc * *temperature * *temperature * *temperature * *temperature;
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += -kappa * (losses - intensity);
        // PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);

        /* // PetscReal actualResult = -kappa * (losses - losses - intensity);
        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);
        PetscScalar analyticalResult = ablate::radiation::RadiationSolver::ReallySolveParallelPlates(cellGeom->centroid[dim-1]);
        // PetscScalar* errorValues;
        // DMPlexPointLocalRef(edm, iCell, errorArray, &errorValues);
        // double error = (analyticalResult);
        // errorValues[0] = error; */

        //Total energy gain of the current cell depends on absorptivity at the current cell
        // PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        ncells++;

    }
    // Cleanup
    VecRestoreArrayRead(rhs, &rhsArray);
    // VecRestoreArray(errors, &errorArray);
    VecRestoreArray(loctemp, &temperatureArray);
    // subDomain->RestoreFieldLocalVector(errorField, &eis, &errors, &edm);
    subDomain->RestoreFieldLocalVector(temperatureField, &vis, &loctemp, &vdm);
    //VecView(rhs, PETSC_VIEWER_STDOUT_WORLD);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::RadiationSolver, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"));