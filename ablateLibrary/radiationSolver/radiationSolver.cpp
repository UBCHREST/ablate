//
// Created by owen on 3/19/22.
//
#include "radiationSolver.hpp"
#include <set>
#include <utility>
#include "radiationProcess.hpp"
#include "utilities/mathUtilities.hpp"
#include <fstream>
#include <iostream>
#include "environment/runEnvironment.hpp"

ablate::radiationSolver::RadiationSolver::RadiationSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary, std::shared_ptr<parameters::Parameters> options)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)), fieldBoundary(std::move(fieldBoundary)) {}

ablate::radiationSolver::RadiationSolver::~RadiationSolver() {
    if (gradientCalculator) {
        PetscFVDestroy(&gradientCalculator); //We probably don't need anything related to gradients here? Unless absorptivity stepping somehow uses it
    }
}

static void AddNeighborsToStencil(std::set<PetscInt>& stencilSet, DMLabel boundaryLabel, PetscInt boundaryValue, PetscInt depth, DM dm, PetscInt cell) {
    const PetscInt maxDepth = 2;

    // Check to see if this cell is already in the list
    if (stencilSet.count(cell)) {
        return;
    }

    // Add to the list
    stencilSet.insert(cell);

    // If not at max depth, check another layer
    if (depth < maxDepth) {
        PetscInt numberFaces;
        const PetscInt* cellFaces;
        DMPlexGetConeSize(dm, cell, &numberFaces) >> ablate::checkError;
        DMPlexGetCone(dm, cell, &cellFaces) >> ablate::checkError;

        // For each connected face
        for (PetscInt f = 0; f < numberFaces; f++) {
            PetscInt face = cellFaces[f];

            // Don't allow the search back over the boundary Label
            PetscInt faceValue;
            DMLabelGetValue(boundaryLabel, face, &faceValue) >> ablate::checkError;
            if (faceValue == boundaryValue) {
                continue;
            }

            // check any neighbors
            PetscInt numberNeighborCells;
            const PetscInt* neighborCells;
            DMPlexGetSupportSize(dm, face, &numberNeighborCells) >> ablate::checkError;
            DMPlexGetSupport(dm, face, &neighborCells) >> ablate::checkError;
            for (PetscInt n = 0; n < numberNeighborCells; n++) {
                AddNeighborsToStencil(stencilSet, boundaryLabel, boundaryValue, depth + 1, dm, neighborCells[n]);
            }
        }
    }
}

void ablate::radiationSolver::RadiationSolver::Setup() { //allows initialization after the subdomain and dm is established

    // Set up the gradient calculator
    PetscFVCreate(PETSC_COMM_SELF, &gradientCalculator) >> checkError;
    // Set least squares as the default type
    PetscFVSetType(gradientCalculator, PETSCFVLEASTSQUARES) >> checkError;
    // Set any other required options
    PetscObjectSetOptions((PetscObject)gradientCalculator, petscOptions) >> checkError;
    PetscFVSetFromOptions(gradientCalculator) >> checkError;
    PetscFVSetNumComponents(gradientCalculator, 1) >> checkError;
    PetscFVSetSpatialDimension(gradientCalculator, subDomain->GetDimensions()) >> checkError;

    // Make sure that his fvm supports gradients
    PetscBool supportsGrads;
    PetscFVGetComputeGradients(gradientCalculator, &supportsGrads) >> checkError;
    if (!supportsGrads) {
        throw std::invalid_argument("The RadiationSolver requires a PetscFVM that supports gradients.");
    }

    // Get the geometry for the mesh TODO: Figure out if anything in the setup is useful/necessary for ray tracing processes
    Vec faceGeomVec, cellGeomVec;
    DMPlexGetGeometryFVM(subDomain->GetDM(), &faceGeomVec, &cellGeomVec, nullptr) >> checkError; //Return precomputed geometric data
    //DM faceDM, cellDM; //Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
    VecGetDM(faceGeomVec, &faceDM) >> checkError; //Gets the DM defining the data layout of the vector
    VecGetDM(cellGeomVec, &cellDM) >> checkError; //Gets the DM defining the data layout of the vector
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
    dim = subDomain->GetDimensions(); //Number of dimensions already defined in the setup

    // Get the labels //Fair enough
    DMLabel boundaryLabel;
    PetscInt boundaryValue = fieldBoundary->GetValue(); //The boundary value is represented by a single index?
    DMGetLabel(subDomain->GetDM(), fieldBoundary->GetName().c_str(), &boundaryLabel) >> checkError;

    // check to see if there is a ghost label
    DMLabel ghostLabel; //What is a ghost label?
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;

    // Keep track of the current maxFaces
    PetscInt maxFaces = 0;

    // March over each cell in this region to create the stencil
    IS cellIS;
    PetscInt cStart, cEnd; //Keep these as local variables, the cell indicies in the radiation initializer should be updated with every radiation initialization. The variables here are owned by the DM
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // make sure we are not working on a ghost cell
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, cell, &ghost);
        }
        if (ghost >= 0) {
            // put in nan, should not be used
            gradientStencils.emplace_back(
                GradientStencil{.geometry = {.normal = {NAN, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}}, .stencil = {}, .gradientWeights = {}, .stencilSize = 0});
            continue;
        }

        // keep a list of cells in the stencil
        std::set<PetscInt> stencilSet{cell};

        // March over each face
        PetscInt numberFaces;
        const PetscInt* cellFaces;
        DMPlexGetConeSize(subDomain->GetDM(), cell, &numberFaces) >> checkError;
        DMPlexGetCone(subDomain->GetDM(), cell, &cellFaces) >> checkError; //Return the points on the in-edges for this point in the DAG

        // Create a new BoundaryFVFaceGeom
        BoundaryFVFaceGeom geom{.normal = {0.0, 0.0, 0.0}, .areas = {0.0, 0.0, 0.0}, .centroid = {0.0, 0.0, 0.0}};

        // Set the face centroid to be equal to the face for gradient calc
        PetscFVCellGeom* cellGeom;
        DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cellGeom);
        PetscArraycpy(geom.centroid, cellGeom->centroid, dim) >> checkError; //Copy the array?

        // Perform some error checking
        if (numberFaces < 1) {
            throw std::runtime_error("Isolated cell " + std::to_string(cell) + " cannot be used in RadiationSolver.");
        }

        // For each connected face
        for (PetscInt f = 0; f < numberFaces; f++) {
            PetscInt face = cellFaces[f];

            // check to see if this face is in the boundary region TODO: Knowing whether a face is on the boundary! Ray tracing starts at boundaries.
            PetscInt faceValue; //Index of the current face being checked
            DMLabelGetValue(boundaryLabel, face, &faceValue) >> checkError;
            if (faceValue != boundaryValue) { //Maybe the ray tracer will need to sort the cells based on the distance from the origin, maybe it will do this passively.
                continue;
            }

            // Get the connected cells
            PetscInt numberNeighborCells; //This is interesting, probably not useful here
            const PetscInt* neighborCells;
            DMPlexGetSupportSize(subDomain->GetDM(), face, &numberNeighborCells) >> checkError;
            DMPlexGetSupport(subDomain->GetDM(), face, &neighborCells) >> checkError;

            for (PetscInt n = 0; n < numberNeighborCells; n++) {
                AddNeighborsToStencil(stencilSet, boundaryLabel, boundaryValue, 1, subDomain->GetDM(), neighborCells[n]);
            }

            // Add this geometry to the BoundaryFVFaceGeom
            PetscFVFaceGeom* fg;
            DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;

            // The normal should be pointing away from the other phase into the boundary solver domain.  The current fg support points from cell[0] -> cell[1]
            // If the neighborCells[0] is in the boundary (this cell), flip the normal
            if (neighborCells[0] == cell) {
                for (PetscInt d = 0; d < dim; d++) {
                    geom.normal[d] -= fg->normal[d];
                    geom.areas[d] -= fg->normal[d];
                }
            } else {
                for (PetscInt d = 0; d < dim; d++) {
                    geom.normal[d] += fg->normal[d];
                    geom.areas[d] += fg->normal[d];
                }
            }
        }

        // compute the normal
        utilities::MathUtilities::NormVector(dim, geom.normal);

        // remove the boundary cell from the stencil
        stencilSet.erase(cell);

        // Compute the weights for the stencil
        std::vector<PetscInt> stencil(stencilSet.begin(), stencilSet.end());
        std::vector<PetscScalar> gradientWeights(stencil.size() * dim, 0.0);
        std::vector<PetscScalar> dx(stencil.size() * dim);
        std::vector<PetscScalar> volume(stencil.size());

        // Use a Reciprocal distance interpolate for the distribution weights.  This can be abstracted away in the future.
        std::vector<PetscScalar> distributionWeights(stencil.size(), 0.0);
        PetscScalar distributionWeightSum = 0.0;
        for (std::size_t n = 0; n < stencil.size(); n++) {
            PetscFVCellGeom* cg;
            DMPlexPointLocalRead(cellDM, stencil[n], cellGeomArray, &cg);
            for (PetscInt d = 0; d < dim; ++d) {
                dx[n * dim + d] = cg->centroid[d] - geom.centroid[d];
                distributionWeights[n] += PetscSqr(cg->centroid[d] - geom.centroid[d]);
            }
            distributionWeights[n] = 1.0 / PetscSqrtScalar(distributionWeights[n]);
            distributionWeightSum += distributionWeights[n];

            // store the volume
            volume[n] = cg->volume;
        }

        // normalize the distributionWeights
        utilities::MathUtilities::ScaleVector(distributionWeights.size(), distributionWeights.data(), 1.0 / distributionWeightSum);

        // Compute gradients
        if ((PetscInt)stencil.size() > maxFaces) {
            maxFaces = (PetscInt)stencil.size();
            PetscFVLeastSquaresSetMaxFaces(gradientCalculator, maxFaces) >> checkError;
        }
        PetscFVComputeGradient(gradientCalculator, (PetscInt)stencil.size(), &dx[0], &gradientWeights[0]) >> checkError;

        // Store the stencil
        gradientStencils.emplace_back(GradientStencil{
            .geometry = geom, .stencil = stencil, .gradientWeights = gradientWeights, .stencilSize = (PetscInt)stencil.size(), .distributionWeights = distributionWeights, .volumes = volume});

        maximumStencilSize = PetscMax(maximumStencilSize, (PetscInt)stencil.size());
    }
    RestoreRange(cellIS, cStart, cEnd, cells);

    // clean up the geom
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
}
void ablate::radiationSolver::RadiationSolver::Initialize() {
    RegisterPreStage([](auto ts, auto& solver, auto stageTime) { //Adds function to be called before each flow stage
        Vec locXVec;
        DMGetLocalVector(solver.GetSubDomain().GetDM(), &locXVec) >> checkError;
        DMGlobalToLocal(solver.GetSubDomain().GetDM(), solver.GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

        // Get the time from the ts
        PetscReal time; //Radiation solver does not directly integrate in time. However, this may still be useful.
        TSGetTime(ts, &time) >> checkError;

        auto& cellSolver = dynamic_cast<CellSolver&>(solver);
        cellSolver.UpdateAuxFields(time, locXVec, solver.GetSubDomain().GetAuxVector()); ///This is the function that updates aux fields!

        DMRestoreLocalVector(solver.GetSubDomain().GetDM(), &locXVec) >> checkError;
    });

    //this->reallySolveParallelPlates();
    std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rays = this->rayInit(); //Runs the ray initialization finding cell indices
    this->rayTrace(rays);
}

///Starting new radiation stuff here
//rayInit() will collect all needed cell locations based on the ray that they are associated with, making the ray cells easy  to step through during run time
//rayTrace() will

///Things this code needs from ABLATE
//Location of the current (ray passing through) cell should be known //This is almost done?
//Get location of domain boundary based on intersection with vector (Just the furthest cell from the ray origin for a given ray) //Don't know how this is going to work yet? Check if a cell is at the boundary?
//Temperature and absorptivity(any property) of arbitrary point based on location (Using aux fields most likely?) //No idea yet, probably not crazy hard (Don't think absorptivity model exists yet)

///Declaring function for the initialization to call, draws each ray vector and gets all of the cells associated with it (sorted by distance and starting at the boundary working in)
std::vector<std::vector<std::vector<std::vector<PetscInt>>>> ablate::radiationSolver::RadiationSolver::rayInit() {
    double theta;                                           // represents the actual current angle (inclination)
    double phi;                                             // represents the actual current angle (rotation)

    ///Locally get a range of cells that are included in this subdomain at this time step for the ray initialization
    // March over each cell in this region to create the stencil
    IS cellIS;
    PetscInt cStart, cEnd; //Keep these as local variables, the cell indices in the radiation initializer should be updated with every radiation initialization. The variables here are owned by the DM
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);
    std::set<PetscInt> stencilSet;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // keep a list of cells in the stencil
        stencilSet.insert(cell);
    }

    /// Create a matrix which can store cell locations based on origin cell, theta, and phi
    std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rays; //Indices: Cell, angle (theta), angle(phi), space ... Can use the pushback command to append

    PetscInt ncells = 0;                                                                //PARALLEL THINGS -------------------------------------------------------
    for(int iCell: stencilSet) {               //loop through subdomain cell indices    //Should each of the cells be owned by a different process? (For parallel)
        double percentComplete = 100.0*double(ncells)/double(stencilSet.size());        //MPI Scatter or Broadcast could distribute the cell stencil information?
        PetscPrintf(PETSC_COMM_WORLD, "Radiation Initializer Percent Complete: %f\n", percentComplete);       //Distribute the cells in the stencil based on rank of the process?
                                                                                        //Incorporate buffers at the end of every cell process?
        std::vector<std::vector<std::vector<PetscInt>>> rayCells; //Make vector to store this dimensional row
        //std::vector<std::vector<std::vector<PetscInt>>> rayCells(stencilSet.size(), 0); //Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible
        //TODO: May need to adjust by moving all of the vector declarations up here and nesting them to start with. This shouldn't be an issue really.

        ///Get the position vector of the current cell index
        const PetscScalar* cellGeomArray; //Declare the variables that will contain the geometry of the cells
        Vec cellGeomVec;
        PetscReal minCellRadius;
        DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, &minCellRadius); //Obtain the geometric information about the cells in the DM?
        VecGetArrayRead(cellGeomVec, &cellGeomArray);
        PetscFVCellGeom* cellGeom;
        DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  // Reads the cell location from the current cell

        ///Set the spatial step size to the minimum cell radius
        //h = minCellRadius;

        for (int ntheta = 0; ntheta < nTheta; ntheta++) {  // for every angle theta
            std::vector<std::vector<PetscInt>> rayThetas; //Make vector to store this dimensional row
            //std::vector<std::vector<PetscInt>> rayThetas(nTheta, 0); //Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible

            // precalculate sin and cosine of the angle theta because it is used frequently?
            for (int nphi = 0; nphi < nPhi; nphi++) {  // for every angle phi

                std::vector<PetscInt> rayPhis; //Make vector to store this dimensional row
                //std::vector<std::vector<PetscInt>> rayPhis(nPhi, 0); //Preallocate the sub-vectors in order to avoid dynamic sizing as much as possible

                PetscReal magnitude = h; //Should represent the distance from the origin cell to the boundary. How to get this? By marching out of the domain! (and checking whether we are still inside)
                int nsteps =  0; //Number of spatial steps that  the ray has taken towards the origin
                bool boundary = false; //Keeps track of whether the ray has intersected the boundary at this point or not
                while (!boundary) {
                    /// Draw a point on which to grab the cells of the DM
                    Vec intersect;                                                // Intersect should point to the boundary, and then be pushed back to the origin, getting each cell
                    theta = ((double)ntheta / (double)nTheta) * pi;             // converts the present angle number into a real angle
                    phi = ((double)nphi / (double)nPhi) * 2.0 * pi;          // converts the present angle number into a real angle

                    //TODO: Alter direction vector so the number of components matches number of dimensions (apply appropriate fixes)
                    PetscInt i[3] = {0, 1, 2};; //[dim] is not known so compiler doesn't like
                    switch(dim) { ///Accounting for the uncertainty in number of dimensions here
                        case 1 :
                            theta = pi/2.0; //Set the value so that the intersect vector sits only in a plane
                            nTheta = 1; //Set nTheta so that only one iteration of the theta angle runs

                            //i = {0}; //Petsc needs the indices of the values that are being input to the vector
                            break;
                        case 2 :
                            theta = pi/2.0; //Set the value so that the intersect vector sits only in a plane
                            nTheta = 1; //Set nTheta so that only one iteration of the theta angle runs

                            //i = {0, 1}; //Petsc needs the indices of the values that are being input to the vector
                            break;
                        case 3 :
                            //i = {0, 1, 2}; //Petsc needs the indices of the values that are being input to the vector
                            break;
                    }

                    PetscReal direction[3] = {(magnitude * sin(theta) * cos(phi)) + cellGeom->centroid[0],  // x component conversion from spherical coordinates, adding the position of the current cell
                                              (magnitude * sin(theta) * sin(phi)) + cellGeom->centroid[1],  // y component conversion from spherical coordinates, adding the position of the current cell
                                              (magnitude * cos(theta)) + cellGeom->centroid[2]};            // z component conversion from spherical coordinates, adding the position of the current cell
                    //(Reference for coordinate transformation: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction

                    /// This block creates the vector pointing to the cell whose index will be stored during the current loop

                    VecCreate(PETSC_COMM_WORLD, &intersect); //Instantiates the vector
                    VecSetBlockSize(intersect, dim) >> checkError;
                    VecSetSizes(intersect, PETSC_DECIDE, dim); //Set size
                    VecSetFromOptions(intersect);
                    VecSetValues(intersect, dim, i, direction, INSERT_VALUES); //Actually input the values of the vector (There are 'dim' values to input)

                    /// Loop through points to try to get a list (vector) of cells that are sitting on that line
                    //Use std vector to store points (build vector outside of while loop?) (or use a set prevents index duplication in the list?)
                    PetscSF cellSF = nullptr;  // PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
                    DMLocatePoints(cellDM, intersect, DM_POINTLOCATION_NONE, &cellSF) >> checkError;  // Locate the points in v in the mesh and return a PetscSF of the containing cells
                    /// An array that maps each point to its containing cell can be obtained with the below
                    //We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
                    PetscInt nFound;
                    const PetscInt* point = nullptr;
                    const PetscSFNode* cell = NULL;
                    PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError; //Using this to get the petsc int cell number from the struct (SF)

                    ///IF THE CELL NUMBER IS RETURNED NEGATIVE, THEN WE HAVE REACHED THE BOUNDARY OF THE DOMAIN >> This exits the loop
                    if (nFound > -1) {
                        ///This function returns multiple values if multiple points are input to it
                        if (cell[0].index >= 0) {
                            /// Assemble a vector of vectors etc associated with each cell index, angular coordinate, and space step?
                            rayPhis.push_back(cell[0].index);
                            //PetscPrintf(PETSC_COMM_WORLD, "Intersect x: %f, y: %f, z: %f Value: %i\n", direction[0], direction[1], direction[2], cell[0].index);
                            // PetscPrintf(PETSC_COMM_WORLD, "Cell: %i, nTheta: %i, Theta: %g, nPhi: %i, Phi: %g, Value: %i\n", iCell, ntheta, theta, nphi, phi, cell[p].index);
                        }else {boundary = true;}
                    }else {boundary = true;}

                    ///Increase the step size of the ray toward the boundary by one more minimum cell radius
                    magnitude += h;  // Increase the magnitude of the ray tracing vector by one space step
                    nsteps++; //Increase the number of steps taken, informs how many indices the vector has

                    // Cleanup
                    PetscSFDestroy(&cellSF);
                    VecDestroy(&intersect);
                }
                rayThetas.push_back(rayPhis);
                //rayThetas[nphi] = rayPhis;
            }
            rayCells.push_back(rayThetas); //This lin is for dynamically allocated assignment
            //rayCells[ntheta] = rayThetas; //This line is for preallocated assignment
        }
        rays.push_back(rayCells); //These lines append the new values into the main vector (They need to be layered on every time)
        //rays[ncells] = rayCells;
        ncells++;
    }
    PetscPrintf(PETSC_COMM_WORLD, "Finished!\n");
    return rays; //Outputs the collection of cell indices in order that their values and locations can be later read from.
}

void ablate::radiationSolver::RadiationSolver::raysGetLoc(std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rays) { ///Write the locations of the ray cells that the initializer has stored (for a single parent cell)

    /// An array that maps each point to its containing cell can be obtained with the below
    const PetscScalar* cellGeomArray; //Declare the variables that will contain the geometry of the cells
    Vec cellGeomVec;

    // March over each cell in this region to create the stencil
    IS cellIS;
    PetscInt cStart, cEnd; //Keep these as local variables, the cell indices in the radiation initializer should be updated with every radiation initialization. The variables here are owned by the DM
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);

    DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, nullptr); //Obtain the geometric information about the cells in the DM?
    VecGetArrayRead(cellGeomVec, &cellGeomArray);

    ///Get the file path for the output
    std::filesystem::path radOutput = environment::RunEnvironment::Get().GetOutputDirectory() / "radOutput.txt";
    std::ofstream stream(radOutput);

    for(int iCell = 10000; iCell <= 10000; iCell++) {     // loop through subdomain cell indices
        for (int ntheta = 0; ntheta < nTheta; ntheta++) {  // for every angle theta
            for (int nphi = 0; nphi < nPhi; nphi++) {
                int numPoints = static_cast<int>(rays[0][ntheta][nphi].size());
                for (int n = 0; n < numPoints; n++) {  /// Every cell point that is stored within the ray
                    for (int i = 0; i < dim; i++) {  // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
                        /// cellGeom represents vector location?
                        PetscFVCellGeom* cellGeom;
                        DMPlexPointLocalRead(cellDM, rays[0][ntheta][nphi][n], cellGeomArray, &cellGeom);  // Reads the cell location from the point in question

                        stream << cellGeom->centroid[i];  // Print the value of the centroid to a 3 (4) column txt file of x,y,z, (ray number ?). (Ultimately these values will need to be stored in program or used)
                        stream << " ";
                    }
                    stream << rays[0][ntheta][nphi][n];  // Prints the index value stored at this point in the ray
                    stream << " ";
                    stream << 0;
                    stream << " ";
                    stream << ntheta;
                    stream << " ";
                    stream << nphi;
                    stream << "\n";
                    PetscPrintf(PETSC_COMM_WORLD, "Saving...\n");
                }
            }
        }
    }
    stream.close();
}

//TODO: Make the rays vector a global variable
//TODO: Make all method titles capitalized
std::vector<PetscReal> ablate::radiationSolver::RadiationSolver::rayTrace(std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rays) { ///Gets the total intensity/radiative gain at a single cell

    //std::cout << "Called ray tracing function. nTheta = " << nTheta << ", nPhi = " << nPhi << ", h step = " << h << "\n"; //DEBUGGING COMMENT

    /// An array that maps each point to its containing cell can be obtained with the below
    const PetscScalar* cellGeomArray; //Declare the variables that will contain the geometry of the cells
    Vec cellGeomVec;
    // March over each cell in this region to create the stencil
    IS cellIS;
    PetscInt cStart, cEnd; //Keep these as local variables, the cell indices in the radiation initializer should be updated with every radiation initialization. The variables here are owned by the DM
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);
    std::set<PetscInt> stencilSet;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c //TODO: Instead of storing the set, put this logic into the cell iteration
        const PetscInt cell = cells ? cells[c] : c;

        // keep a list of cells in the stencil
        stencilSet.insert(cell);
    }

    //Obtain the cell information
    DMPlexGetGeometryFVM(cellDM, nullptr, &cellGeomVec, nullptr); //Obtain the geometric information about the cells in the DM?
    VecGetArrayRead(cellGeomVec, &cellGeomArray);

    ///Declare the basic information
    std::vector<PetscReal> radGain(stencilSet.size(), 0); //Represents the total final radiation intensity (for every cell, index as index)
    PetscReal temperature; //The temperature at any given location
    PetscReal dTheta = pi/nTheta;
    PetscReal dPhi = (2*pi)/nPhi;
    double kappa = 1; //Absorptivity coefficient, property of each cell
    double theta;

    ///Get the file path for the output
    std::filesystem::path radOutput = environment::RunEnvironment::Get().GetOutputDirectory() / "tempPlot.txt";
    std::ofstream stream(radOutput);

    std::vector<std::vector<PetscReal>> locations; //2 Dimensional vector which stores the locations of the cell centers
    PetscInt ncells = 0;
    for(int iCell: stencilSet) {     // loop through subdomain cell indices
        ///Print location of the current cell
        for (int i = 0; i < dim; i++) {                   // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
            PetscFVCellGeom* cellGeom;
            DMPlexPointLocalRead(cellDM, iCell, cellGeomArray, &cellGeom);  // Reads the cell location from the point in question
            stream << cellGeom->centroid[i];  // Print the value of the centroid to a 3 (4) column txt file of x,y,z, (ray number ?). (Ultimately these values will need to be stored in program or used)
            stream << " ";
        }
        PetscReal intensity = 0;
        for (int ntheta = 0; ntheta < nTheta; ntheta++) {  // for every angle theta
            theta = ((double)ntheta / (double)nTheta) * pi;    // converts the present angle number into a real angle
            for (int nphi = 0; nphi < nPhi; nphi++) {
                ///Each ray is born here. They begin at the far field temperature.
                ///Initial ray intensity should be set based on which boundary it is coming from.

                ///If the ray originates from the walls, then set the initial ray intensity to the wall temperature
                PetscReal rayIntensity; // = flameIntensity(1, refTemp); //Initialize the ray intensity as the far fie0ld flame intensity
                if (theta > pi/2) { //If sitting on the bottom boundary (everything on the lower half of the angles)
                    rayIntensity = flameIntensity(1, 1300); //Set the initial ray intensity to the bottom wall intensity
                }
                else if (theta < pi/2) { //If sitting on the top boundary
                    rayIntensity = flameIntensity(1, 700); //Set the initial ray intensity to the top wall intensity
                }

                int numPoints = static_cast<int>(rays[ncells][ntheta][nphi].size());
                for (int n = (numPoints-1); n >= 0; n--) {  /// Go through every cell point that is stored within the ray >> FROM THE BOUNDARY TO THE SOURCE
                    std::vector<PetscReal> loc(3,0);
                    for (int i = 0; i < dim; i++) {  // We will need an extra column to store the ray vector. This will probably need to be stored in a different vector
                        PetscFVCellGeom* cellGeom;
                        DMPlexPointLocalRead(cellDM, rays[ncells][ntheta][nphi][n], cellGeomArray, &cellGeom);  // Reads the cell location from the point in question

                        loc[i] = cellGeom->centroid[i]; //Get the location of the current cell and export it to the rest of the function
                        //stream << cellGeom->centroid[i];  // Print the value of the centroid to a 3 (4) column txt file of x,y,z, (ray number ?). (Ultimately these values will need to be stored in program or used)
                        //stream << " ";
                    }

                    ///Redefine the absorptivity and temperature in this section
                    ///Get the local temperature at every point along the ray. For the analytical solution, the temperature at each point will be based on the spatial location of the cell.
                    if(loc[2] <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                        temperature = -6.349E6*loc[2]*loc[2] + 2000.0;
                    }else{
                        temperature = -1.179E7*loc[2]*loc[2] + 2000.0;
                    }

                    //TODO: For ABLATE implementation, get temperature based on this function
                    //const auto &temperatureField = subDomain->GetField("temperature");
                    //PetscScalar* pt = nullptr;
                    //DMPlexPointLocalFieldRef(cellDM, rays[ncells][ntheta][nphi][n], temperatureField.id, cellGeomArray, &pt);

                    //TODO: Input absorptivity values from model here.

                    ///The ray intensity changes as a function of the environment at this point
                    rayIntensity = flameIntensity(1 - exp(-kappa*h), temperature) + rayIntensity*exp(-kappa*h);
                    //stream << rayIntensity;  // Prints the intensity at this point
                    //stream << "\n";
                    //PetscPrintf(PETSC_COMM_WORLD, "Intensity: %f\n", rayIntensity);
                }
                ///The rays end here, their intensity is added to the total intensity of the cell
                intensity += rayIntensity*sin(theta)*dTheta*dPhi; //Gives the partial impact of the ray on the total sphere. The sin(theta) is a result of the polar coordinate discretization
            }
            //PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        }
        radGain[ncells] = kappa*intensity; //Total energy gain of the current cell depends on absorptivity at the current cell
        PetscPrintf(PETSC_COMM_WORLD, "Radiative Gain: %g\n", intensity);
        ///Print the radiative gain at each cell
        stream << intensity;
        stream << "\n";

        ncells++;
    }
    stream.close();
    return radGain; //TODO: Ultimately add net radiation to the flow field directly
}

std::vector<PetscReal> ablate::radiationSolver::RadiationSolver::solveParallelPlates(){ ///Get the analytical solution for two parallel plates of different temperatures
    ///Define variables and basic information
    std::vector<PetscReal> G;//(zCellNumber, 0); //Initialize a ray representing the irradiation at each cell point
    PetscReal IT = flameIntensity(1, 700); //Intensity of rays originating from the top plate
    PetscReal IB = flameIntensity(1, 1300); //Set the initial ray intensity to the bottom wall intensity //Intensity of rays originating from the bottom plate
    PetscReal kappa = 1; //Kappa is not spatially dependant in this special case
    PetscReal zBottom = -0.0105; //Prescribe the top and bottom heights for the domain
    PetscReal zTop = 0.0105;

    PetscReal temperature;
    PetscReal Ibz;

    ///Get the file path for the output
    std::filesystem::path radOutput = environment::RunEnvironment::Get().GetOutputDirectory() / "analytical.txt"; //Will contain irradiation at each point
    std::ofstream stream(radOutput);

    PetscReal nMu = 1000; //Number of mu for which the angle is computed //1/cos(theta); //Theta never comes up in the actual process of solving so we can just iterate over mu to keep things evenly spaced
    //PetscReal mu = (nmu / nMu); //From 0 to 1
    PetscReal nZ = 1000;
    PetscReal nZp = 1000;

    ///For every z
    for(double nz = 1; nz < (nZ-1); nz++) {
        ///Get the temperature
        PetscReal z = zBottom + (nz/nZ)*(zTop-zBottom); //This calculates the z height based on the number of zs
        std::vector<PetscReal> Iplus;
        std::vector<PetscReal> Iminus;
        /// For every mu
        for(double nmu = 1; nmu < (nMu-1); nmu++) {
            PetscReal mu = (nmu / nMu); //From 0 to 1
            ///First get vectors of the value that will be integrated along z prime
            std::vector<PetscReal> plus;
            std::vector<PetscReal> minus;
            ///For every zprime
            for(double nzp = 1; nzp < (nZp-1); nzp++) { ///Plus integral goes from bottom to Z
                PetscReal zp = zBottom + (nzp/nZp)*(z-zBottom); //Calculate the z height
                ///Get the temperature
                //PetscReal z = zBottom + (nz/nZ)*(zTop-zBottom); //This calculates the z height based on the number of zs
                if(zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                    temperature = -6.349E6*zp*zp + 2000.0;
                }else{
                    temperature = -1.179E7*zp*zp + 2000.0;
                }
                ///Get the black body intensity here
                Ibz = flameIntensity(1, temperature);
                plus.push_back(Ibz * exp(-(kappa*(z-zp))/mu));
            }
            for(double nzp = 1; nzp < (nZp-1); nzp++) { ///Minus integral goes from z to top
                PetscReal zp = z + (nzp / nZp) * (zTop - z);  // Calculate the zp height
                /// Get the temperature
                // PetscReal z = zBottom + (nz/nZ)*(zTop-zBottom); //This calculates the z height based on the number of zs
                if (zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                    temperature = -6.349E6 * zp * zp + 2000.0;
                } else {
                    temperature = -1.179E7 * zp * zp + 2000.0;
                }
                /// Get the black body intensity here
                Ibz = flameIntensity(1, temperature);
                minus.push_back(Ibz * exp((kappa * (zp - z)) / mu));
            }

            /// Calculate the value (with the zprime integrals)
            Iplus.push_back(IB * exp(-(kappa * z) / mu) + (kappa / mu) * cSimp(zBottom, z, plus));  // Add the intensity from above at this height (to the vector, for integration)
            Iminus.push_back(IT * exp((kappa*((zTop-zBottom)-z))/(mu-1)) - (kappa/(mu-1)) * cSimp(z, zTop, minus)); //Add the intensity from below at this height (to the vector, for integration)
        }
        /// Integrate the G values along mu (also for every z, in the same loop)
        PetscReal intensity = 2 * pi * (cSimp(-1, 0, Iminus) + cSimp(0, 1, Iplus));
        stream << z;
        stream << ' ';
        stream << intensity;
        stream << '\n';
        G.push_back(2 * pi * (cSimp(-1, 0, Iminus) + cSimp(0, 1, Iplus)));
    }
    stream.close();
    return G; //Return the vector of values representing the irradiation at each height z
}

std::vector<PetscReal> ablate::radiationSolver::RadiationSolver::reallySolveParallelPlates(){
    ///Define variables and basic information
    std::vector<PetscReal> G; //Irradiation at each point
    PetscReal IT = flameIntensity(1, 700); //Intensity of rays originating from the top plate
    PetscReal IB = flameIntensity(1, 1300); //Set the initial ray intensity to the bottom wall intensity //Intensity of rays originating from the bottom plate
    PetscReal kappa = 1; //Kappa is not spatially dependant in this special case
    PetscReal zBottom = -0.0105; //Prescribe the top and bottom heights for the domain
    PetscReal zTop = 0.0105;

    PetscReal intensity;
    PetscReal temperature;
    PetscReal Ibz;

    ///Get the file path for the output
    std::filesystem::path radOutput = environment::RunEnvironment::Get().GetOutputDirectory() / "analytical.txt"; //Will contain irradiation at each point
    std::ofstream stream(radOutput);

    //PetscReal nMu = 1000; //Number of mu for which the angle is computed //1/cos(theta); //Theta never comes up in the actual process of solving so we can just iterate over mu to keep things evenly spaced
    //PetscReal mu = (nmu / nMu); //From 0 to 1
    PetscReal nZ = 1000;
    PetscReal nZp = 1000;

    ///For every z
    for(double nz = 1; nz < (nZ-1); nz++) {
        PetscReal z = zBottom + (nz/nZ)*(zTop-zBottom); //This calculates the z height based on the number of zs
        Ibz = 0;

        std::vector<PetscReal> Iplus;
        std::vector<PetscReal> Iminus;

        for(double nzp = 1; nzp < (nZp-1); nzp++) { ///Plus integral goes from bottom to Z
            PetscReal zp = zBottom + (nzp/nZp)*(z-zBottom); //Calculate the z height
            ///Get the temperature
            //PetscReal z = zBottom + (nz/nZ)*(zTop-zBottom); //This calculates the z height based on the number of zs
            if(zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                temperature = -6.349E6*zp*zp + 2000.0;
            }else{
                temperature = -1.179E7*zp*zp + 2000.0;
            }
            ///Get the black body intensity here
            Ibz = flameIntensity(1, temperature);
            Iplus.push_back(Ibz * eInteg(1,kappa*(z-zp)));
        }
        for(double nzp = 1; nzp < (nZp-1); nzp++) { ///Minus integral goes from z to top
            PetscReal zp = z + (nzp/nZp) * (zTop-z);  // Calculate the zp height
            /// Get the temperature
            // PetscReal z = zBottom + (nz/nZ)*(zTop-zBottom); //This calculates the z height based on the number of zs
            if (zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other?
                temperature = -6.349E6*zp*zp + 2000.0;
            } else {
                temperature = -1.179E7*zp*zp + 2000.0;
            }
            /// Get the black body intensity here
            Ibz = flameIntensity(1, temperature);
            Iminus.push_back(Ibz * eInteg(1,kappa * (zp-z)));
        }

        PetscReal term1 = IB*eInteg(2,kappa*(z-zBottom));
        PetscReal term2 = IT*eInteg(2,kappa*(zTop-z));
        PetscReal term3 = cSimp(zBottom,z,Iplus);
        PetscReal term4 = cSimp(z,zTop,Iminus);
        G.push_back(2 * pi * (term1 + term2 + term3 + term4));
        ///Write to the file
        intensity = 2 * pi * (term1 + term2 + term3 + term4);
        stream << z;
        stream << ' ';
        stream << intensity;
        stream << '\n';
    }
    return G;
}

///This function will get the set of ray cells, grab their properties, and march the intensity along them
PetscReal ablate::radiationSolver::RadiationSolver::castRay(int theta, int phi,  std::vector<PetscReal> intersect) { ///Spatially integrates intensity over current ray based on temp & absorption at each distance
    std::vector<PetscReal> ray = {0,0,0}; //vector representing the ray as it is traced back from the boundary to the origin.

    double rayIntensity = flameIntensity(1, refTemp); //Initialize the ray intensity as the far field flame intensity TODO: should this be changed to the casing (boundary node) temperature in the context of a rocket?
    double magnitude = mag(intersect); //get the magnitude of the vector between the origin cell and the boundary TODO: Set magnitude equal to intersection point minus origin
    while(magnitude - h > 0) { //Keep stepping intensity through space until the origin cell has been reached
        double kappa = 0; //Absorptivity at the current point TODO: Need the absorptivity at any point in the domain
        double temp = 0; //Temperature at the current point (used to calculate current point flame intensity)//TODO: Need temperature of any point in the domain
        //From Java, implemented in line below //Math.exp(-kappa[ijk[2]][ijk[1]][ijk[0]] * deltaRay) TODO: ask, what part is this exactly?
        rayIntensity += flameIntensity(1 - exp(-kappa*h), temp + rayIntensity*exp(-kappa*h)); //represents the flame intensity at the far field boundary
        ray = {0,0,0};//std::Subtract(intersect,origin);//TODO: vector subtract from intersection point to get next coordinate, how to identify point at which properties are measured?
        magnitude -= h; //One step towards the origin has been completed
    }
    std::cout << "rayIntensity: " << rayIntensity << "\n";
    return rayIntensity; //Final intensity of the ray as it approaches the current cell
}

PetscReal ablate::radiationSolver::RadiationSolver::eInteg(int order, double x) {
    if(x == 0 && order !=1) return 1/(order-1); //Simple solution in this case, exit
    std::vector<PetscReal> En;
    double N = 100;
    for(double n = 1; n < N; n++) {
        double mu = n/N;
        if(order == 1) {
            En.push_back(exp(-x/mu)/mu);
        }
        if(order == 2) {
            En.push_back(exp(-x/mu));
        }
        if(order == 3) {
            En.push_back(exp(-x/mu)*mu);
        }
    }
    PetscReal final = cSimp(0, 1, En);
    return final;
}

PetscReal ablate::radiationSolver::RadiationSolver::flameIntensity(double epsilon, double temperature) { ///Gets the flame intensity based on temperature and emissivity
    return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
}

PetscReal ablate::radiationSolver::RadiationSolver::mag( std::vector<PetscReal> vector) { ///Simple function to find magnitude of a vector
    PetscReal magnitude = 0;
    for (const int i : vector) { //Sum of all points
        magnitude += i*i; //Squared
    }
    magnitude = sqrt(magnitude); //Square root of the resulting sum
    return magnitude; //Return the magnitude of the vector as a double
}

PetscReal ablate::radiationSolver::RadiationSolver::cSimp(PetscReal a, PetscReal b, std::vector<double> f) {
    ///b-a represents the size of the total domain that is being integrated over
    //PetscReal b = H; //End
    //PetscReal a = 0; //Beginning
    PetscReal I;
    PetscReal n = static_cast<double>(f.size()); //The number of elements in the vector that is being integrated over
    int margin = 0;

    //PetscReal h = (b-a)/n; //Step size
    PetscReal f_sum = 0; //Initialize the sum of all middle elements

    if (a!=b) {
        ///Loop through every point except the first and last
        for (int i = margin; i < (n-margin); i++) {
            if (i%2 == 0 ) {
                f[i] = 2 * f[i]; //Weight lightly on the border
            }else {
                f[i] = 4 * f[i]; //Weight heavily in the center
            }
            f_sum += f[i]; //Add this value to the total every time
        }
        //I = ((b - a) / (3 * n)) * (f_sum);  // The ends make it upset, just don't include them :3
        I = ((b - a) / (3 * n)) * (f[0] + f_sum + f[n]);  // Compute the total final integral
    }else{
        I = 0;
    }
    return I;
}

///End of the added radiation stuff

void ablate::radiationSolver::RadiationSolver::RegisterFunction(ablate::radiationSolver::RadiationSolver::BoundarySourceFunction function, void* context, const std::vector<std::string>& sourceFields,
                                                              const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields, BoundarySourceType type) {
    // Create the FVMRHS Function
    BoundaryFunctionDescription functionDescription{.function = function, .context = context, .type = type};

    for (auto& sourceField : sourceFields) {
        auto& fieldId = subDomain->GetField(sourceField); //Source field may not be useful or necessary for ray tracing
        functionDescription.sourceFields.push_back(fieldId.subId);
    }

    for (auto& inputField : inputFields) {
        auto& inputFieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(inputFieldId.subId);
    }

    for (const auto& auxField : auxFields) { //Aux fields are an input to the RegisterFunction method //TODO: Need aux filed for temperature and absorptivity
        auto& auxFieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(auxFieldId.subId);
    }

    boundaryFunctions.push_back(functionDescription);
}
PetscErrorCode ablate::radiationSolver::RadiationSolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) { //main interface for integrating in time Inputs: local vector, x vector (current solution), local f vector
    PetscFunctionBeginUser;                                                                                             //gets fields out of the main vector
    //TODO: This is likely where the radiative gain will be calculated and updated. The aux fields need to be updated and rays cast. Tracing vectors are precalculated in init.
    PetscErrorCode ierr;

    // Extract the cell geometry, and the dm that holds the information
    auto dm = subDomain->GetDM();
    auto auxDM = subDomain->GetAuxDM();
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;
    ierr = DMPlexGetGeometryFVM(dm, nullptr, &cellGeomVec, nullptr);
    CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCell);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);
    CHKERRQ(ierr);

    // prepare to compute the source, u, and a offsets
    PetscInt nf;
    ierr = PetscDSGetNumFields(subDomain->GetDiscreteSystem(), &nf);
    CHKERRQ(ierr);

    // Create the required offset arrays. These are sized for the max possible value
    PetscInt* offsetsTotal;
    ierr = PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &offsetsTotal);
    CHKERRQ(ierr);
    PetscInt* auxOffTotal = nullptr;
    if (auto auxDS = subDomain->GetAuxDiscreteSystem()) {
        PetscDSGetComponentOffsets(auxDS, &auxOffTotal) >> checkError;
    }

    // Get the size of the field
    PetscInt scratchSize;
    PetscDSGetTotalDimension(subDomain->GetDiscreteSystem(), &scratchSize) >> checkError;
    std::vector<PetscScalar> distributedSourceScratch(scratchSize);

    // presize the offsets
    std::vector<PetscInt> sourceOffsets(subDomain->GetFields().size(), -1);
    std::vector<PetscInt> inputOffsets(subDomain->GetFields().size(), -1);
    std::vector<PetscInt> auxOffsets(subDomain->GetFields(domain::FieldLocation::AUX).size(), -1);

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;
    // Get the region to march over
    IS cellIS; //(includes what cells are needed in the solver, iterate through, get needed values)
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);
    if (cEnd > cStart) {
        dim = subDomain->GetDimensions();

        // Get pointers to sol, aux, and f vectors
        const PetscScalar *locXArray, *locAuxArray = nullptr;
        PetscScalar* locFArray;
        ierr = VecGetArrayRead(locXVec, &locXArray);
        CHKERRQ(ierr);
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            ierr = VecGetArrayRead(locAuxVec, &locAuxArray);
            CHKERRQ(ierr);
        }
        VecGetArray(locFVec, &locFArray) >> checkError;
        CHKERRQ(ierr);

        // Store pointers to the stencil variables
        std::vector<const PetscScalar*> inputStencilValues(maximumStencilSize);
        std::vector<const PetscScalar*> auxStencilValues(maximumStencilSize);

        // March over each boundary function
        for (const auto& function : boundaryFunctions) {
            for (std::size_t i = 0; i < function.sourceFields.size(); i++) {
                sourceOffsets[i] = offsetsTotal[function.sourceFields[i]];
            }
            for (std::size_t i = 0; i < function.inputFields.size(); i++) {
                inputOffsets[i] = offsetsTotal[function.inputFields[i]];
            }
            for (std::size_t i = 0; i < function.auxFields.size(); i++) {
                auxOffsets[i] = auxOffTotal[function.inputFields[i]];
            }

            auto sourceOffsetsPointer = sourceOffsets.data();
            auto inputOffsetsPointer = inputOffsets.data();
            auto auxOffsetsPointer = auxOffsets.data();

            // March over each cell in this region
            PetscInt cOffset = 0;  // Keep track of the cell offset
            for (PetscInt c = cStart; c < cEnd; ++c, cOffset++) {
                // if there is a cell array, use it, otherwise it is just c
                const PetscInt cell = cells ? cells[c] : c;

                // make sure we are not working on a ghost cell
                PetscInt ghost = -1;
                if (ghostLabel) {
                    DMLabelGetValue(ghostLabel, cell, &ghost);
                }
                if (ghost >= 0) {
                    continue;
                }

                // Get the cell geom
                const PetscFVCellGeom* cg;
                DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cg) >> checkError;

                // Get pointers to the area of interest
                const PetscScalar *solPt, *auxPt = nullptr;
                DMPlexPointLocalRead(dm, cell, locXArray, &solPt) >> checkError;
                if (auxDM) {
                    DMPlexPointLocalRead(auxDM, cell, locAuxArray, &auxPt) >> checkError;
                }

                // Get each of the stencil pts
                const auto& stencilInfo = gradientStencils[cOffset];
                for (PetscInt p = 0; p < stencilInfo.stencilSize; p++) {
                    DMPlexPointLocalRead(dm, stencilInfo.stencil[p], locXArray, &inputStencilValues[p]) >> checkError;
                    if (auxDM) {
                        DMPlexPointLocalRead(auxDM, stencilInfo.stencil[p], locAuxArray, &auxStencilValues[p]) >> checkError;
                    }
                }

                // Get the pointer to the rhs
                switch (function.type) {
                    case BoundarySourceType::Point:
                        PetscScalar* rhs;
                        DMPlexPointLocalRef(dm, cell, locFArray, &rhs) >> checkError;

                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                           const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                           const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                           PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void* ctx)*/
                        ierr = function.function(dim,
                                                 &stencilInfo.geometry,
                                                 cg,
                                                 inputOffsetsPointer,
                                                 solPt,
                                                 inputStencilValues.data(),
                                                 auxOffsetsPointer,
                                                 auxPt,
                                                 auxStencilValues.data(),
                                                 stencilInfo.stencilSize,
                                                 stencilInfo.stencil.data(),
                                                 stencilInfo.gradientWeights.data(),
                                                 sourceOffsetsPointer,
                                                 rhs,
                                                 function.context);
                        break;
                    case BoundarySourceType::Distributed:
                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                                                   const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                                                   const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                                                   PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[],
                           void* ctx)*/

                        // zero out the distributedSourceScratch
                        PetscArrayzero(distributedSourceScratch.data(), (PetscInt)distributedSourceScratch.size()) >> checkError;

                        ierr = function.function(dim,
                                                 &stencilInfo.geometry,
                                                 cg,
                                                 inputOffsetsPointer,
                                                 solPt,
                                                 inputStencilValues.data(),
                                                 auxOffsetsPointer,
                                                 auxPt,
                                                 auxStencilValues.data(),
                                                 stencilInfo.stencilSize,
                                                 stencilInfo.stencil.data(),
                                                 stencilInfo.gradientWeights.data(),
                                                 sourceOffsetsPointer,
                                                 distributedSourceScratch.data(),
                                                 function.context);

                        // Now distribute to each stencil point
                        for (PetscInt s = 0; s < stencilInfo.stencilSize; ++s) {
                            // Get the point in the rhs for this point.  It might be ghost but that is ok, the values are added together later
                            DMPlexPointLocalRef(dm, stencilInfo.stencil[s], locFArray, &rhs) >> checkError;

                            // Now over the entire rhs, the function should have added the values correctly using the sourceOffsetsPointer
                            for (PetscInt sc = 0; sc < scratchSize; sc++) {
                                rhs[sc] += (distributedSourceScratch[sc] * stencilInfo.distributionWeights[s]) / stencilInfo.volumes[s];
                            }
                        }

                        break;
                }

                CHKERRQ(ierr);
            }
        }

        // clean up access
        ierr = VecRestoreArrayRead(locXVec, &locXArray);
        CHKERRQ(ierr);
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            ierr = VecRestoreArrayRead(locAuxVec, &locXArray);
            CHKERRQ(ierr);
        }
        VecRestoreArray(locFVec, &locFArray) >> checkError;
        CHKERRQ(ierr);

        // clean up the geom
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    }

    RestoreRange(cellIS, cStart, cEnd, cells);

    PetscFunctionReturn(0);
}
void ablate::radiationSolver::RadiationSolver::InsertFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, PetscReal time) {
    for (const auto& fieldFunction : fieldFunctions) {
        // Get the field
        const auto& field = subDomain->GetField(fieldFunction->GetName());

        // Get the vec that goes with the field
        auto vec = subDomain->GetVec(field);
        auto dm = subDomain->GetFieldDM(field);

        // Get the raw array
        PetscScalar* array;
        VecGetArray(vec, &array) >> checkError;

        // March over each cell
        IS cellIS;
        PetscInt cStart, cEnd;
        const PetscInt* cells;
        GetCellRange(cellIS, cStart, cEnd, cells);
        dim = subDomain->GetDimensions();

        // Get the petscFunction/context
        auto petscFunction = fieldFunction->GetFieldFunction()->GetPetscFunction();
        auto context = fieldFunction->GetFieldFunction()->GetContext();

        // March over each cell in this region
        PetscInt cOffset = 0;  // Keep track of the cell offset
        for (PetscInt c = cStart; c < cEnd; ++c, cOffset++) {
            // if there is a cell array, use it, otherwise it is just c
            const PetscInt cell = cells ? cells[c] : c;

            // Get pointers to sol, aux, and f vectors
            PetscScalar* pt = nullptr;
            switch (field.location) {
                case domain::FieldLocation::SOL:
                    DMPlexPointGlobalFieldRef(dm, cell, field.id, array, &pt) >> checkError;
                    break;
                case domain::FieldLocation::AUX:
                    DMPlexPointLocalFieldRef(dm, cell, field.id, array, &pt) >> checkError;
                    break;
            }
            if (pt) {
                petscFunction(dim, time, gradientStencils[cOffset].geometry.centroid, field.numberComponents, pt, context);
            }
        }

        RestoreRange(cellIS, cStart, cEnd, cells);
        VecRestoreArray(vec, &array);
    }
}

void ablate::radiationSolver::RadiationSolver::ComputeGradient(PetscInt dim, PetscScalar boundaryValue, PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights,
                                                             PetscScalar* grad) {
    PetscArrayzero(grad, dim);

    for (PetscInt c = 0; c < stencilSize; ++c) {
        PetscScalar delta = stencilValues[c] - boundaryValue;

        for (PetscInt d = 0; d < dim; ++d) {
            grad[d] += stencilWeights[c * dim + d] * delta;
        }
    }
}

void ablate::radiationSolver::RadiationSolver::ComputeGradientAlongNormal(PetscInt dim, const ablate::radiationSolver::RadiationSolver::BoundaryFVFaceGeom* fg, PetscScalar boundaryValue,
                                                                        PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights, PetscScalar& dPhiDNorm) {
    dPhiDNorm = 0.0;
    for (PetscInt c = 0; c < stencilSize; ++c) {
        PetscScalar delta = stencilValues[c] - boundaryValue;

        for (PetscInt d = 0; d < dim; ++d) {
            dPhiDNorm += stencilWeights[c * dim + d] * delta * fg->normal[d];
        }
    }
}

const ablate::radiationSolver::RadiationSolver::BoundaryFVFaceGeom& ablate::radiationSolver::RadiationSolver::GetBoundaryGeometry(PetscInt cell) const {
    IS cellIS;
    PetscInt cstart, cend;
    const PetscInt* cells;
    GetCellRange(cellIS, cstart, cend, cells);

    // Locate the index
    PetscInt location;
    ISLocate(cellIS, cell, &location) >> checkError;
    RestoreRange(cellIS, cstart, cend, cells);

    return gradientStencils[location].geometry;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiationSolver::RadiationSolver, "A solver for radiative heat transfer in participating media", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "the region describing the faces between the boundary and field"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"));