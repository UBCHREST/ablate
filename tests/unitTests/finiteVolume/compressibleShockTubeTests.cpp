static char help[] = "Compressible ShockTube 1D Tests";

#include <petsc.h>
#include <cmath>
#include <domain/modifiers/ghostBoundaryCells.hpp>
#include <finiteVolume/compressibleFlowFields.hpp>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/dmTransfer.hpp"
#include "environment/runEnvironment.hpp"
#include "eos/perfectGas.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

typedef struct {
    PetscReal gamma;
    PetscReal length;
    PetscReal rhoL;
    PetscReal uL;
    PetscReal pL;
    PetscReal rhoR;
    PetscReal uR;
    PetscReal pR;
} InitialConditions;

struct CompressibleShockTubeParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    InitialConditions initialConditions;
    std::shared_ptr<finiteVolume::fluxCalculator::FluxCalculator> fluxCalculator;
    PetscInt nx;
    PetscReal maxTime;
    PetscReal cfl;
    std::map<std::string, std::vector<PetscReal>> expectedValues;
};

class CompressibleShockTubeTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleShockTubeParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if (x[0] < initialConditions->length / 2.0) {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoL;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoL * initialConditions->uL;

        PetscReal e = initialConditions->pL / ((initialConditions->gamma - 1.0) * initialConditions->rhoL);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uL);
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoL;

    } else {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoR;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoR * initialConditions->uR;

        PetscReal e = initialConditions->pR / ((initialConditions->gamma - 1.0) * initialConditions->rhoR);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uR);
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoR;
    }

    return 0;
}

static PetscErrorCode Extract1DPrimitives(DM dm, Vec v, std::map<std::string, std::vector<double>> &results) {
    Vec cellgeom;
    PetscCall(DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL));
    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
    DM dmCell;
    PetscCall(VecGetDM(cellgeom, &dmCell));
    const PetscScalar *cgeom;
    PetscCall(VecGetArrayRead(cellgeom, &cgeom));
    const PetscScalar *x;
    PetscCall(VecGetArrayRead(v, &x));

    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom *cg;
        const PetscReal *xc;

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc));
        if (xc) {  // must be real cell and not ghost
            results["x"].push_back(cg->centroid[0]);
            PetscReal rho = xc[ablate::finiteVolume::CompressibleFlowFields::RHO];
            results["rho"].push_back(rho);
            PetscReal u = xc[ablate::finiteVolume::CompressibleFlowFields::RHOU] / rho;
            results["u"].push_back(u);
            PetscReal e = (xc[ablate::finiteVolume::CompressibleFlowFields::RHOE] / rho) - 0.5 * u * u;
            results["e"].push_back(e);
        }
    }

    PetscCall(VecRestoreArrayRead(cellgeom, &cgeom));
    PetscCall(VecRestoreArrayRead(v, &x));
    return 0;
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if (c[0] < initialConditions->length / 2.0) {
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoL;

        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoL * initialConditions->uL;

        PetscReal e = initialConditions->pL / ((initialConditions->gamma - 1.0) * initialConditions->rhoL);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uL);
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoL;
    } else {
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoR;

        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoR * initialConditions->uR;

        PetscReal e = initialConditions->pR / ((initialConditions->gamma - 1.0) * initialConditions->rhoR);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uR);
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoR;
    }
    return 0;
    PetscFunctionReturn(0);
}

TEST_P(CompressibleShockTubeTestFixture, ShouldReproduceExpectedResult) {
    StartWithMPI
        {
            DM dmCreate; /* problem definition */

            // Get the testing param
            auto &testingParam = GetParam();

            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize(help);

            // Create a mesh
            // hard code the problem setup to act like a oneD problem
            PetscReal start[] = {0.0};
            PetscReal end[] = {testingParam.initialConditions.length};
            PetscInt nx[] = {testingParam.nx};
            DMBoundaryType bcType[] = {DM_BOUNDARY_NONE};
            DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 1, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dmCreate) >> testErrorChecker;

            auto eos = std::make_shared<ablate::eos::PerfectGas>(
                std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", std::to_string(testingParam.initialConditions.gamma)}}));

            // define the fields based upon a compressible flow
            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};
            auto mesh = std::make_shared<ablate::domain::DMTransfer>(
                dmCreate, fieldDescriptors, std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::GhostBoundaryCells>()});

            auto initialCondition = std::make_shared<mathFunctions::FieldFunction>("euler", mathFunctions::Create(SetInitialCondition, (void *)&testingParam.initialConditions));

            // create a time stepper
            auto timeStepper = ablate::solver::TimeStepper(
                mesh,
                ablate::parameters::MapParameters::Create(
                    {{"ts_max_time", std::to_string(testingParam.maxTime)}, {"ts_adapt_type", "physics"}, {"ts_adapt_safety", "1.0"}, {"ts_exact_final_time", "matchstep"}}),
                {},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{initialCondition});

            // Setup the flow data
            auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", std::to_string(testingParam.cfl)}});

            auto boundaryConditions = std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>>{
                std::make_shared<finiteVolume::boundaryConditions::Ghost>("euler", "wall left", 1, PhysicsBoundary_Euler, (void *)&testingParam.initialConditions),
                std::make_shared<finiteVolume::boundaryConditions::Ghost>("euler", "right left", 2, PhysicsBoundary_Euler, (void *)&testingParam.initialConditions)};

            auto flowObject = std::make_shared<ablate::finiteVolume::CompressibleFlowSolver>("testFlow",
                                                                                             ablate::domain::Region::ENTIREDOMAIN,
                                                                                             nullptr /*options*/,
                                                                                             eos,
                                                                                             parameters,
                                                                                             nullptr /*transportModel*/,
                                                                                             testingParam.fluxCalculator,
                                                                                             boundaryConditions /*boundary conditions*/);

            // run
            timeStepper.Register(flowObject);
            timeStepper.Solve();

            // extract the results
            std::map<std::string, std::vector<PetscReal>> results;
            Extract1DPrimitives(mesh->GetDM(), mesh->GetSolutionVector(), results) >> testErrorChecker;

            // Compare the expected values
            for (const auto &expectedResults : testingParam.expectedValues) {
                // get the computed value
                const auto &computedResults = results[expectedResults.first];

                ASSERT_EQ(expectedResults.second.size(), computedResults.size())
                    << "expected/computed result vectors for " << expectedResults.first << "  are of different lengths " << expectedResults.second.size() << "/" << computedResults.size();
                for (std::size_t i = 0; i < computedResults.size(); i++) {
                    ASSERT_NEAR(expectedResults.second[i], computedResults[i], 1E-6) << " in " << expectedResults.first << " at [" << i << "]";
                }
            }
        }
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, CompressibleShockTubeTestFixture,
    testing::Values(
        (CompressibleShockTubeParameters){
            .mpiTestParameter = testingResources::MpiTestParameter("ausm case 1 sod problem"),
            .initialConditions = {.gamma = 1.4, .length = 1.0, .rhoL = 1.0, .uL = 0.0, .pL = 1.0, .rhoR = 0.125, .uR = 0.0, .pR = .1},
            .fluxCalculator = std::make_shared<finiteVolume::fluxCalculator::Ausm>(),
            .nx = 100,
            .maxTime = 0.25,
            .cfl = 0.5,
            .expectedValues =
                {{"x", {0.005000, 0.015000, 0.025000, 0.035000, 0.045000, 0.055000, 0.065000, 0.075000, 0.085000, 0.095000, 0.105000, 0.115000, 0.125000, 0.135000, 0.145000, 0.155000, 0.165000,
                        0.175000, 0.185000, 0.195000, 0.205000, 0.215000, 0.225000, 0.235000, 0.245000, 0.255000, 0.265000, 0.275000, 0.285000, 0.295000, 0.305000, 0.315000, 0.325000, 0.335000,
                        0.345000, 0.355000, 0.365000, 0.375000, 0.385000, 0.395000, 0.405000, 0.415000, 0.425000, 0.435000, 0.445000, 0.455000, 0.465000, 0.475000, 0.485000, 0.495000, 0.505000,
                        0.515000, 0.525000, 0.535000, 0.545000, 0.555000, 0.565000, 0.575000, 0.585000, 0.595000, 0.605000, 0.615000, 0.625000, 0.635000, 0.645000, 0.655000, 0.665000, 0.675000,
                        0.685000, 0.695000, 0.705000, 0.715000, 0.725000, 0.735000, 0.745000, 0.755000, 0.765000, 0.775000, 0.785000, 0.795000, 0.805000, 0.815000, 0.825000, 0.835000, 0.845000,
                        0.855000, 0.865000, 0.875000, 0.885000, 0.895000, 0.905000, 0.915000, 0.925000, 0.935000, 0.945000, 0.955000, 0.965000, 0.975000, 0.985000, 0.995000}},
                 {"rho", {1.000000, 1.000000, 1.000000, 0.999999, 0.999997, 0.999992, 0.999980, 0.999954, 0.999896, 0.999775, 0.999536, 0.999083, 0.998268, 0.996875, 0.994620, 0.991162, 0.986145,
                          0.979257, 0.970287, 0.959177, 0.946029, 0.931079, 0.914641, 0.897052, 0.878628, 0.859637, 0.840293, 0.820763, 0.801168, 0.781601, 0.762128, 0.742796, 0.723642, 0.704690,
                          0.685956, 0.667453, 0.649187, 0.631162, 0.613377, 0.595829, 0.578511, 0.561415, 0.544528, 0.527829, 0.511293, 0.494878, 0.478519, 0.462099, 0.445373, 0.427631, 0.394493,
                          0.415279, 0.421802, 0.422726, 0.422831, 0.422784, 0.422669, 0.422502, 0.422277, 0.421972, 0.421541, 0.420903, 0.419927, 0.418413, 0.416093, 0.412641, 0.407711, 0.401003,
                          0.392340, 0.381738, 0.369446, 0.355940, 0.341863, 0.327929, 0.314816, 0.303071, 0.293050, 0.284902, 0.278585, 0.273918, 0.270632, 0.268432, 0.267032, 0.266187, 0.265699,
                          0.265417, 0.265220, 0.264988, 0.264547, 0.263559, 0.261284, 0.256083, 0.244433, 0.219878, 0.179709, 0.143757, 0.128993, 0.125675, 0.125107, 0.125017}},
                 {"u", {0.000000, 0.000000, 0.000001, 0.000002, 0.000005, 0.000012, 0.000029, 0.000068, 0.000151, 0.000322, 0.000658, 0.001287, 0.002405, 0.004292, 0.007311, 0.011884, 0.018447,
                        0.027375, 0.038918, 0.053153, 0.069977, 0.089147, 0.110338, 0.133201, 0.157412, 0.182693, 0.208823, 0.235630, 0.262986, 0.290794, 0.318985, 0.347507, 0.376324, 0.405412,
                        0.434752, 0.464337, 0.494163, 0.524232, 0.554551, 0.585135, 0.616001, 0.647177, 0.678699, 0.710617, 0.743003, 0.775962, 0.809662, 0.844401, 0.880794, 0.920599, 0.998209,
                        0.947086, 0.931702, 0.929284, 0.928708, 0.928439, 0.928270, 0.928148, 0.928054, 0.927977, 0.927917, 0.927874, 0.927854, 0.927861, 0.927904, 0.927992, 0.928133, 0.928335,
                        0.928600, 0.928922, 0.929277, 0.929627, 0.929919, 0.930098, 0.930121, 0.929971, 0.929664, 0.929246, 0.928777, 0.928314, 0.927898, 0.927548, 0.927263, 0.927025, 0.926799,
                        0.926526, 0.926088, 0.925228, 0.923364, 0.919147, 0.909484, 0.887345, 0.836997, 0.724718, 0.499883, 0.209345, 0.048515, 0.008337, 0.001329, 0.000208}},
                 {"e", {2.500000, 2.500000, 2.500000, 2.499999, 2.499997, 2.499992, 2.499980, 2.499954, 2.499896, 2.499775, 2.499536, 2.499082, 2.498265, 2.496866, 2.494594, 2.491096, 2.485994,
                        2.478938, 2.469672, 2.458088, 2.444242, 2.428335, 2.410661, 2.391552, 2.371321, 2.350242, 2.328532, 2.306363, 2.283862, 2.261124, 2.238220, 2.215203, 2.192112, 2.168974,
                        2.145811, 2.122636, 2.099460, 2.076288, 2.053120, 2.029955, 2.006785, 1.983599, 1.960378, 1.937095, 1.913710, 1.890158, 1.866336, 1.842054, 1.816907, 1.789725, 1.738208,
                        1.774796, 1.785637, 1.787790, 1.788779, 1.789639, 1.790530, 1.791524, 1.792694, 1.794158, 1.796112, 1.798903, 1.803094, 1.809542, 1.819448, 1.834353, 1.856052, 1.886427,
                        1.927208, 1.979682, 2.044387, 2.120816, 2.207195, 2.300413, 2.396201, 2.489628, 2.575862, 2.651007, 2.712735, 2.760506, 2.795328, 2.819226, 2.834640, 2.843934, 2.849089,
                        2.851582, 2.852354, 2.851798, 2.849633, 2.844476, 2.832727, 2.805981, 2.745427, 2.613658, 2.378205, 2.132874, 2.026528, 2.004348, 2.000687, 2.000110}}}},
        (CompressibleShockTubeParameters){
            .mpiTestParameter = testingResources::MpiTestParameter("case 2 expansion left and expansion right"),
            .initialConditions = {.gamma = 1.4, .length = 1.0, .rhoL = 1.0, .uL = -2.0, .pL = 0.4, .rhoR = 1.0, .uR = 2.0, .pR = 0.4},
            .fluxCalculator = std::make_shared<finiteVolume::fluxCalculator::Ausm>(),
            .nx = 100,
            .maxTime = 0.15,
            .cfl = 0.5,
            .expectedValues =
                {{"x", {0.005000, 0.015000, 0.025000, 0.035000, 0.045000, 0.055000, 0.065000, 0.075000, 0.085000, 0.095000, 0.105000, 0.115000, 0.125000, 0.135000, 0.145000, 0.155000, 0.165000,
                        0.175000, 0.185000, 0.195000, 0.205000, 0.215000, 0.225000, 0.235000, 0.245000, 0.255000, 0.265000, 0.275000, 0.285000, 0.295000, 0.305000, 0.315000, 0.325000, 0.335000,
                        0.345000, 0.355000, 0.365000, 0.375000, 0.385000, 0.395000, 0.405000, 0.415000, 0.425000, 0.435000, 0.445000, 0.455000, 0.465000, 0.475000, 0.485000, 0.495000, 0.505000,
                        0.515000, 0.525000, 0.535000, 0.545000, 0.555000, 0.565000, 0.575000, 0.585000, 0.595000, 0.605000, 0.615000, 0.625000, 0.635000, 0.645000, 0.655000, 0.665000, 0.675000,
                        0.685000, 0.695000, 0.705000, 0.715000, 0.725000, 0.735000, 0.745000, 0.755000, 0.765000, 0.775000, 0.785000, 0.795000, 0.805000, 0.815000, 0.825000, 0.835000, 0.845000,
                        0.855000, 0.865000, 0.875000, 0.885000, 0.895000, 0.905000, 0.915000, 0.925000, 0.935000, 0.945000, 0.955000, 0.965000, 0.975000, 0.985000, 0.995000}},
                 {"rho", {0.990693, 0.985358, 0.977885, 0.967857, 0.954925, 0.938846, 0.919504, 0.896920, 0.871239, 0.842708, 0.811650, 0.778438, 0.743467, 0.707136, 0.669835, 0.631935, 0.593781,
                          0.555690, 0.517951, 0.480826, 0.444547, 0.409320, 0.375325, 0.342714, 0.311615, 0.282128, 0.254329, 0.228271, 0.203983, 0.181470, 0.160719, 0.141698, 0.124359, 0.108638,
                          0.094457, 0.081721, 0.070351, 0.060477, 0.051927, 0.044460, 0.037920, 0.032184, 0.027155, 0.022753, 0.018918, 0.015573, 0.013455, 0.012614, 0.012130, 0.010813, 0.010813,
                          0.012130, 0.012614, 0.013455, 0.015573, 0.018918, 0.022753, 0.027155, 0.032184, 0.037920, 0.044460, 0.051927, 0.060477, 0.070351, 0.081721, 0.094457, 0.108638, 0.124359,
                          0.141698, 0.160719, 0.181470, 0.203983, 0.228271, 0.254329, 0.282128, 0.311615, 0.342714, 0.375325, 0.409320, 0.444547, 0.480826, 0.517951, 0.555690, 0.593781, 0.631935,
                          0.669835, 0.707136, 0.743467, 0.778438, 0.811650, 0.842708, 0.871239, 0.896920, 0.919504, 0.938846, 0.954925, 0.967857, 0.977885, 0.985358, 0.990693}},
                 {"u", {-1.993037, -1.989039, -1.983428, -1.975870, -1.966071, -1.953793, -1.938875, -1.921233, -1.900854, -1.877786, -1.852121, -1.823984, -1.793515, -1.760860, -1.726162,
                        -1.689556, -1.651165, -1.611100, -1.569454, -1.526307, -1.481727, -1.435768, -1.388472, -1.339871, -1.289987, -1.238836, -1.186424, -1.132753, -1.077817, -1.021608,
                        -0.964111, -0.905310, -0.845186, -0.783719, -0.720894, -0.656702, -0.590610, -0.520851, -0.450550, -0.380484, -0.310038, -0.238681, -0.165907, -0.091212, -0.013950,
                        0.058689,  0.093648,  0.075235,  0.026865,  -0.000107, 0.000107,  -0.026865, -0.075235, -0.093648, -0.058689, 0.013950,  0.091212,  0.165907,  0.238681,  0.310038,
                        0.380484,  0.450550,  0.520851,  0.590610,  0.656702,  0.720894,  0.783719,  0.845186,  0.905310,  0.964111,  1.021608,  1.077817,  1.132753,  1.186424,  1.238836,
                        1.289987,  1.339871,  1.388472,  1.435768,  1.481727,  1.526307,  1.569454,  1.611100,  1.651165,  1.689556,  1.726162,  1.760860,  1.793515,  1.823984,  1.852121,
                        1.877786,  1.900854,  1.921233,  1.938875,  1.953793,  1.966071,  1.975870,  1.983428,  1.989039,  1.993037}},
                 {"e", {0.996311, 0.994214, 0.991291, 0.987389, 0.982383, 0.976188, 0.968769, 0.960138, 0.950352, 0.939503, 0.927710, 0.915110, 0.901849, 0.888074, 0.873930, 0.859554, 0.845074,
                        0.830604, 0.816248, 0.802094, 0.788218, 0.774683, 0.761538, 0.748822, 0.736562, 0.724774, 0.713467, 0.702640, 0.692288, 0.682397, 0.672950, 0.663928, 0.655311, 0.647086,
                        0.639272, 0.631994, 0.626029, 0.622515, 0.618529, 0.613393, 0.607427, 0.600835, 0.593754, 0.586279, 0.578463, 0.570150, 0.581665, 0.625485, 0.700210, 0.847984, 0.847984,
                        0.700210, 0.625485, 0.581665, 0.570150, 0.578463, 0.586279, 0.593754, 0.600835, 0.607427, 0.613393, 0.618529, 0.622515, 0.626029, 0.631994, 0.639272, 0.647086, 0.655311,
                        0.663928, 0.672950, 0.682397, 0.692288, 0.702640, 0.713467, 0.724774, 0.736562, 0.748822, 0.761538, 0.774683, 0.788218, 0.802094, 0.816248, 0.830604, 0.845074, 0.859554,
                        0.873930, 0.888074, 0.901849, 0.915110, 0.927710, 0.939503, 0.950352, 0.960138, 0.968769, 0.976188, 0.982383, 0.987389, 0.991291, 0.994214, 0.996311}}}},
        (CompressibleShockTubeParameters){
            .mpiTestParameter = testingResources::MpiTestParameter("case 5 shock collision shock left and shock right"),
            .initialConditions = {.gamma = 1.4, .length = 1.0, .rhoL = 5.99924, .uL = 19.5975, .pL = 460.894, .rhoR = 5.99242, .uR = -6.19633, .pR = 46.0950},
            .fluxCalculator = std::make_shared<finiteVolume::fluxCalculator::Ausm>(),
            .nx = 100,
            .maxTime = 0.035,
            .cfl = 0.5,
            .expectedValues =
                {{"x", {0.005000, 0.015000, 0.025000, 0.035000, 0.045000, 0.055000, 0.065000, 0.075000, 0.085000, 0.095000, 0.105000, 0.115000, 0.125000, 0.135000, 0.145000, 0.155000, 0.165000,
                        0.175000, 0.185000, 0.195000, 0.205000, 0.215000, 0.225000, 0.235000, 0.245000, 0.255000, 0.265000, 0.275000, 0.285000, 0.295000, 0.305000, 0.315000, 0.325000, 0.335000,
                        0.345000, 0.355000, 0.365000, 0.375000, 0.385000, 0.395000, 0.405000, 0.415000, 0.425000, 0.435000, 0.445000, 0.455000, 0.465000, 0.475000, 0.485000, 0.495000, 0.505000,
                        0.515000, 0.525000, 0.535000, 0.545000, 0.555000, 0.565000, 0.575000, 0.585000, 0.595000, 0.605000, 0.615000, 0.625000, 0.635000, 0.645000, 0.655000, 0.665000, 0.675000,
                        0.685000, 0.695000, 0.705000, 0.715000, 0.725000, 0.735000, 0.745000, 0.755000, 0.765000, 0.775000, 0.785000, 0.795000, 0.805000, 0.815000, 0.825000, 0.835000, 0.845000,
                        0.855000, 0.865000, 0.875000, 0.885000, 0.895000, 0.905000, 0.915000, 0.925000, 0.935000, 0.945000, 0.955000, 0.965000, 0.975000, 0.985000, 0.995000}},
                 {"rho", {5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,
                          5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,
                          5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,
                          5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  6.001963,  8.261163,  14.209665, 14.273952, 14.266171, 14.263294, 14.264514, 14.268053, 14.272093,
                          14.275407, 14.277637, 14.279339, 14.281838, 14.287083, 14.297693, 14.317384, 14.351741, 14.409155, 14.501634, 14.645233, 14.859850, 15.168092, 15.592990, 16.154615,
                          16.865975, 17.729946, 18.739155, 19.874075, 21.103112, 22.382463, 23.658517, 24.897900, 26.080438, 27.156451, 28.098670, 28.886758, 29.511560, 29.965892, 30.223051,
                          30.127375, 29.487040, 27.181684, 13.159955, 6.203000,  5.992420,  5.992420,  5.992420,  5.992420,  5.992420}},
                 {"u", {19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500,
                        19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500,
                        19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500,
                        19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.592788, 15.181666, 8.778382,  8.720769,  8.717497,  8.713758,  8.710248,  8.707428,  8.705557,
                        8.704562,  8.704028,  8.703348,  8.701977,  8.699659,  8.696517,  8.692960,  8.689481,  8.686462,  8.684045,  8.682041,  8.679877,  8.676686,  8.671698,  8.664902,
                        8.657514,  8.651395,  8.647628,  8.646946,  8.649888,  8.657565,  8.671195,  8.684040,  8.688499,  8.690005,  8.688847,  8.685869,  8.679851,  8.666416,  8.633869,
                        8.564686,  8.337653,  7.725351,  4.165598,  -5.531340, -6.196330, -6.196330, -6.196330, -6.196330, -6.196330}},
                 {"e",
                  {192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495,
                   192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495,
                   192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495,
                   192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.098504, 241.477162, 295.055504, 295.673547, 295.903016, 296.028278, 296.048128, 295.990655, 295.891594,
                   295.784363, 295.693864, 295.630796, 295.586799, 295.530824, 295.405677, 295.122723, 294.553374, 293.518808, 291.783491, 289.061478, 285.043904, 279.449907, 272.092656, 262.942160,
                   252.166156, 240.133833, 227.363407, 214.429498, 201.870996, 190.123399, 179.500211, 170.214153, 162.351066, 155.845952, 150.604664, 146.488895, 143.334577, 140.939385, 139.026918,
                   137.126946, 134.697600, 128.956932, 88.524424,  23.351223,  19.230545,  19.230545,  19.230545,  19.230545,  19.230545}}}}),
    [](const testing::TestParamInfo<CompressibleShockTubeParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
