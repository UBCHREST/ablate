#include "radiation.hpp"
#include <set>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "utilities/mathUtilities.hpp"

ablate::radiation::Radiation::Radiation(std::string solverId, std::shared_ptr<domain::Region> region, const PetscInt raynumber, std::shared_ptr<parameters::Parameters> options,
                                        std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)), raynumber(raynumber), radiationModel(std::move(radiationModelIn)), log(std::move(log)) {
    nTheta = raynumber;    //!< The number of angles to solve with, given by user input
    nPhi = 2 * raynumber;  //!< The number of angles to solve with, given by user input
}

ablate::radiation::Radiation::~Radiation() {}

void ablate::radiation::Radiation::Setup() { /** allows initialization after the subdomain and dm is established*/
    ablate::solver::CellSolver::Setup();
    dim = subDomain->GetDimensions();  //!< Number of dimensions already defined in the setup
}

void ablate::radiation::Radiation::Initialize() {
    /** Begins radiation properties model
     * Runs the ray initialization, finding cell indices
     * Initialize the log if provided
     */
    // Store the required data for the low level c functions
    absorptivityFunction = radiationModel->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, subDomain->GetFields());

    if (log) {
        log->Initialize(subDomain->GetComm());
    }
    this->RayInit();
}

void ablate::radiation::Radiation::RayInit() {
    /** Initialization to call, draws each ray vector and gets all of the cells associated with it
     * (sorted by distance and starting at the boundary working in)
     * */

    double theta;  //!< represents the actual current angle (inclination)
    double phi;    //!< represents the actual current angle (rotation)

    /**Locally get a range of cells that are included in this subdomain at this time step for the ray initialization
     * */
    solver::Range cellRange;
    GetCellRange(cellRange);

    /** Create a nested vector which can store cell locations based on origin cell, theta, phi, and space step
     * Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
     * Make vector to store this dimensional row
     * Indices: Cell, angle (theta), angle(phi), space steps
     * */
    std::vector<PetscInt> rayPhis;
    std::vector<std::vector<PetscInt>> rayThetas(nPhi, rayPhis);
    std::vector<std::vector<std::vector<PetscInt>>> rayCells(nTheta, rayThetas);
    rays.resize((cellRange.end - cellRange.start), rayCells);

    /** Get setup things for the position vector of the current cell index
     * Declare the variables that will contain the geometry of the cells
     * Obtain the geometric information about the cells in the DM
     * */
    const PetscScalar* cellGeomArray;
    PetscReal minCellRadius;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM);
    DMPlexGetGeometryFVM(cellDM, nullptr, nullptr, &minCellRadius);
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    PetscFVCellGeom* cellGeom;

    PetscInt ncells = 0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;
        /** Provide progress updates if the log is enabled.
         * Initializer described how many of the cell ray locations have been stored.
         */
        if (log) {
            double percentComplete = 100.0 * double(ncells) / double((cellRange.end - cellRange.start));
            log->Printf("Radiation Initializer Percent Complete: %.1f\n", percentComplete);
        }

        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  //!< Reads the cell location from the current cell

        /** Set the spatial step size to the minimum cell radius */
        h = minCellRadius;

        /** for every angle theta
         * for every angle phi
         */
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /** Should represent the distance from the origin cell to the boundary. How to get this? By marching out of the domain! (and checking whether we are still inside)
                 * Number of spatial steps that  the ray has taken towards the origin
                 * Keeps track of whether the ray has intersected the boundary at this point or not
                 */
                PetscReal magnitude = h;
                int nsteps = 0;
                bool boundary = false;
                while (!boundary) {
                    /** Draw a point on which to grab the cells of the DM
                     * Intersect should point to the boundary, and then be pushed back to the origin, getting each cell
                     * converts the present angle number into a real angle
                     * */
                    Vec intersect;
                    theta = ((double)ntheta / (double)nTheta) * pi;
                    phi = ((double)nphi / (double)nPhi) * 2.0 * pi;
                    PetscInt i[3] = {0, 1, 2};

                    /** x component conversion from spherical coordinates, adding the position of the current cell
                     * y component conversion from spherical coordinates, adding the position of the current cell
                     * z component conversion from spherical coordinates, adding the position of the current cell
                     * (Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction
                     * */
                    PetscReal direction[3] = {
                        (magnitude * sin(theta) * cos(phi)) + cellGeom->centroid[0],  // x component conversion from spherical coordinates, adding the position of the current cell
                        (magnitude * sin(theta) * sin(phi)) + cellGeom->centroid[1],  // y component conversion from spherical coordinates, adding the position of the current cell
                        (magnitude * cos(theta)) + cellGeom->centroid[2]};            // z component conversion from spherical coordinates, adding the position of the current cell
                    //(Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction

                    /** This block creates the vector pointing to the cell whose index will be stored during the current loop*/
                    VecCreate(PETSC_COMM_WORLD, &intersect);  //!< Instantiates the vector
                    VecSetBlockSize(intersect, dim) >> checkError;
                    VecSetSizes(intersect, PETSC_DECIDE, dim);  //!< Set size
                    VecSetFromOptions(intersect);
                    VecSetValues(intersect, dim, i, direction, INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)

                    /** Loop through points to try to get a list (vector) of cells that are sitting on that line*/
                    PetscSF cellSF = nullptr;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
                    DMLocatePoints(cellDM, intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  //!< Locate the points in v in the mesh and return a PetscSF of the containing cells

                    /** An array that maps each point to its containing cell can be obtained with the below
                     * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
                     * */

                    PetscInt nFound;
                    const PetscInt* point = nullptr;
                    const PetscSFNode* cell = NULL;
                    PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  //!< Using this to get the petsc int cell number from the struct (SF)

                    /** IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
                     * This function returns multiple values if multiple points are input to it
                     * Make sure that whatever cell is returned is in the stencil set (and not outside of the radiation domain)
                     * Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
                     * The boundary has been reached if any of these conditions don't hold
                     * */
                    if (nFound > -1 && cell[0].index >= 0 && subDomain->InRegion(cell[0].index)) {
                        rays[ncells][ntheta][nphi].push_back(cell[0].index);
                    } else {
                        boundary = true;  //!< The boundary has been reached if any of these conditions don't hold
                    }

                    /** Increase the step size of the ray toward the boundary by one more minimum cell radius
                     * Increase the magnitude of the ray tracing vector by one space step
                     * Increase the number of steps taken, informs how many indices the vector has
                     * */
                    magnitude += h;
                    nsteps++;

                    /** Cleanup*/
                    PetscSFDestroy(&cellSF);
                    VecDestroy(&intersect);
                }
            }
        }
        ncells++;  //!< Increase the number of cells that have been finished.
    }
    /** Cleanup*/
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray);
    RestoreRange(cellRange);
}

PetscErrorCode ablate::radiation::Radiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBeginUser;
    /** Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
     * These DM objects belong to the temperature field which is used to calculate radiation transport
     */
    DM vdm;
    Vec loctemp;
    IS vis;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable*/
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);

    /** Get the array of the solution vector*/
    const PetscScalar* solArray;
    VecGetArrayRead(solVec, &solArray);

    const auto& eulerFieldInfo = subDomain->GetField("euler");
    auto dm = subDomain->GetDM();  //!< Get the main DM for the solution vector

    /** Get the temperature field
     * For ABLATE implementation, get temperature based on this function
     */
    const auto& temperatureField = subDomain->GetField("temperature");
    PetscScalar* temperatureArray = nullptr;
    subDomain->GetFieldLocalVector(temperatureField, time, &vis, &loctemp, &vdm);
    VecGetArray(loctemp, &temperatureArray);

    /** Declare the basic information*/
    PetscReal* sol;          //!< The solution value at any given location
    PetscReal* temperature;  //!< The temperature at any given location
    PetscReal dTheta = pi / nTheta;
    PetscReal dPhi = (2 * pi) / nPhi;
    double kappa = 1;  //!< Absorptivity coefficient, property of each cell
    double theta;

    std::vector<std::vector<PetscReal>> locations;  //!< 2 Dimensional vector which stores the locations of the cell centers
    PetscInt ncells = 0;

    auto absorptivityFunctionContext = absorptivityFunction.context.get();

    /** loop through subdomain cell indices
     * for every angle theta
     * converts the present angle number into a real angle
     * */
    solver::Range cellRange;
    GetCellRange(cellRange);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;
        PetscReal intensity = 0;
        for (int ntheta = 1; ntheta < nTheta; ntheta++) {    // for every angle theta
            theta = ((double)ntheta / (double)nTheta) * pi;  // converts the present angle number into a real angle
            for (int nphi = 0; nphi < nPhi; nphi++) {
                /** Each ray is born here. They begin at the far field temperature.
                    Initial ray intensity should be set based on which boundary it is coming from.
                    If the ray originates from the walls, then set the initial ray intensity to the wall temperature, etc.
                 */
                PetscReal rayIntensity = 0.0;
                int numPoints = static_cast<int>(rays[ncells][ntheta][nphi].size());

                if (numPoints > 0) {
                    for (int n = (numPoints - 1); n >= 0; n--) {
                        /** Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                            Define the absorptivity and temperature in this section
                            For ABLATE implementation, get temperature based on this function
                            Get the array that lives inside the vector
                            Gets the temperature from the cell index specified
                        */
                        DMPlexPointLocalRead(vdm, rays[ncells][ntheta][nphi][n], temperatureArray, &temperature);
                        DMPlexPointLocalRead(dm, rays[ncells][ntheta][nphi][n], solArray, &sol);
                        /** Input absorptivity (kappa) values from model here.*/
                        absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);

                        if (n == (numPoints - 1)) {                          /** If this is the beginning of the ray, set this as the initial intensity.*/
                            rayIntensity = FlameIntensity(1, *temperature);  //!< Set the initial ray intensity to the boundary intensity
                            /// Make intensity boundary conditions be set from boundary label
                        } else {
                            /** The ray intensity changes as a function of the environment at this point*/
                            rayIntensity = FlameIntensity(1 - exp(-kappa * h), *temperature) + rayIntensity * exp(-kappa * h);
                        }
                    }
                }
                /** The rays end here, their intensity is added to the total intensity of the cell
                 * Gives the partial impact of the ray on the total sphere.
                 * The sin(theta) is a result of the polar coordinate discretization
                 * */
                intensity += rayIntensity * sin(theta) * dTheta * dPhi;
            }
        }
        /** Provide progress updates if the log is enabled.
         * Radiative gain can be useful for debugging purposes.
         */
        if (log) {
            log->Printf("Radiative Gain: %g\n", intensity);
        }

        /** Gets the temperature from the cell index specified*/
        DMPlexPointLocalRef(vdm, iCell, temperatureArray, &temperature);

        /** Put the irradiation into the right hand side function*/
        PetscScalar* rhsValues;
        DMPlexPointLocalFieldRead(subDomain->GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues);
        PetscReal losses = 4 * sbc * *temperature * *temperature * *temperature * *temperature;
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += -kappa * (losses - intensity);

        /** Add to the number of cells that has been counted*/
        ncells++;
    }
    /** Cleanup*/
    VecRestoreArrayRead(rhs, &rhsArray);
    VecRestoreArray(loctemp, &temperatureArray);
    subDomain->RestoreFieldLocalVector(temperatureField, &vis, &loctemp, &vdm);
    RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

PetscReal ablate::radiation::Radiation::FlameIntensity(double epsilon, double temperature) { /** Gets the flame intensity based on temperature and emissivity*/
    const PetscReal sbc = 5.6696e-8;                                                         //!< Stefan-Boltzman Constant (J/K)
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiation::Radiation, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));