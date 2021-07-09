static char help[] = "Compressible ShockTube 1D Tests";

#include <petsc.h>
#include <cmath>
#include <flow/fluxCalculator/ausm.hpp>
#include <flow/processes/eulerAdvection.hpp>
#include <memory>
#include <mesh/dmWrapper.hpp>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "eos/perfectGas.hpp"
#include "flow/boundaryConditions/ghost.hpp"
#include "flow/compressibleFlow.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"

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
    std::shared_ptr<flow::fluxCalculator::FluxCalculator> fluxCalculator;
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
        u[ablate::flow::processes::EulerAdvection::RHO] = initialConditions->rhoL;
        u[ablate::flow::processes::EulerAdvection::RHOU + 0] = initialConditions->rhoL * initialConditions->uL;
        u[ablate::flow::processes::EulerAdvection::RHOU + 1] = 0.0;

        PetscReal e = initialConditions->pL / ((initialConditions->gamma - 1.0) * initialConditions->rhoL);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uL);
        u[ablate::flow::processes::EulerAdvection::RHOE] = et * initialConditions->rhoL;

    } else {
        u[ablate::flow::processes::EulerAdvection::RHO] = initialConditions->rhoR;
        u[ablate::flow::processes::EulerAdvection::RHOU + 0] = initialConditions->rhoR * initialConditions->uR;
        u[ablate::flow::processes::EulerAdvection::RHOU + 1] = 0.0;

        PetscReal e = initialConditions->pR / ((initialConditions->gamma - 1.0) * initialConditions->rhoR);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uR);
        u[ablate::flow::processes::EulerAdvection::RHOE] = et * initialConditions->rhoR;
    }

    return 0;
}

static PetscErrorCode Extract1DPrimitives(DM dm, Vec v, std::map<std::string, std::vector<double>> &results) {
    Vec cellgeom;
    PetscErrorCode ierr = DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL);
    CHKERRQ(ierr);
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);
    CHKERRQ(ierr);
    DM dmCell;
    ierr = VecGetDM(cellgeom, &dmCell);
    CHKERRQ(ierr);
    const PetscScalar *cgeom;
    ierr = VecGetArrayRead(cellgeom, &cgeom);
    CHKERRQ(ierr);
    const PetscScalar *x;
    ierr = VecGetArrayRead(v, &x);
    CHKERRQ(ierr);

    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom *cg;
        const PetscReal *xc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);
        CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);
        CHKERRQ(ierr);
        if (xc) {  // must be real cell and not ghost
            results["x"].push_back(cg->centroid[0]);
            PetscReal rho = xc[ablate::flow::processes::EulerAdvection::RHO];
            results["rho"].push_back(rho);
            PetscReal u = xc[ablate::flow::processes::EulerAdvection::RHOU] / rho;
            results["u"].push_back(u);
            PetscReal e = (xc[ablate::flow::processes::EulerAdvection::RHOE] / rho) - 0.5 * u * u;
            results["e"].push_back(e);
        }
    }

    ierr = VecRestoreArrayRead(cellgeom, &cgeom);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v, &x);
    CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if (c[0] < initialConditions->length / 2.0) {
        a_xG[ablate::flow::processes::EulerAdvection::RHO] = initialConditions->rhoL;

        a_xG[ablate::flow::processes::EulerAdvection::RHOU + 0] = initialConditions->rhoL * initialConditions->uL;
        a_xG[ablate::flow::processes::EulerAdvection::RHOU + 1] = 0.0;

        PetscReal e = initialConditions->pL / ((initialConditions->gamma - 1.0) * initialConditions->rhoL);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uL);
        a_xG[ablate::flow::processes::EulerAdvection::RHOE] = et * initialConditions->rhoL;
    } else {
        a_xG[ablate::flow::processes::EulerAdvection::RHO] = initialConditions->rhoR;

        a_xG[ablate::flow::processes::EulerAdvection::RHOU + 0] = initialConditions->rhoR * initialConditions->uR;
        a_xG[ablate::flow::processes::EulerAdvection::RHOU + 1] = 0.0;

        PetscReal e = initialConditions->pR / ((initialConditions->gamma - 1.0) * initialConditions->rhoR);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uR);
        a_xG[ablate::flow::processes::EulerAdvection::RHOE] = et * initialConditions->rhoR;
    }
    return 0;
    PetscFunctionReturn(0);
}

TEST_P(CompressibleShockTubeTestFixture, ShouldReproduceExpectedResult) {
    StartWithMPI
        {
            DM dmCreate; /* problem definition */
            TS ts;       /* timestepper */

            // Get the testing param
            auto &testingParam = GetParam();

            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, help) >> testErrorChecker;

            // Create a ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> testErrorChecker;
            TSSetProblemType(ts, TS_NONLINEAR) >> testErrorChecker;
            TSSetType(ts, TSEULER) >> testErrorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> testErrorChecker;

            // Create a mesh
            // hard code the problem setup to act like a oneD problem
            PetscReal start[] = {0.0, 0.0};
            PetscReal end[] = {testingParam.initialConditions.length, 1};
            PetscInt nx[] = {testingParam.nx, 1};
            DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
            DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dmCreate) >> testErrorChecker;

            // Setup the flow data
            auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", std::to_string(testingParam.cfl)}, {"mu", "0.0"}, {"k", "0.0"}});

            auto eos = std::make_shared<ablate::eos::PerfectGas>(
                std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", std::to_string(testingParam.initialConditions.gamma)}}));

            auto initialCondition = std::make_shared<mathFunctions::FieldSolution>("euler", mathFunctions::Create(SetInitialCondition, (void *)&testingParam.initialConditions));

            auto boundaryConditions = std::vector<std::shared_ptr<flow::boundaryConditions::BoundaryCondition>>{
                std::make_shared<flow::boundaryConditions::Ghost>("euler", "wall left", "Face Sets", 4, PhysicsBoundary_Euler, (void *)&testingParam.initialConditions),
                std::make_shared<flow::boundaryConditions::Ghost>("euler", "right left", "Face Sets", 2, PhysicsBoundary_Euler, (void *)&testingParam.initialConditions),
                std::make_shared<flow::boundaryConditions::Ghost>("euler", "mirrorWall", "Face Sets", std::vector<int>{1, 3}, PhysicsBoundary_Euler, (void *)&testingParam.initialConditions)};

            auto flowObject = std::make_shared<ablate::flow::CompressibleFlow>("testFlow",
                                                                               std::make_shared<ablate::mesh::DMWrapper>(dmCreate),
                                                                               eos,
                                                                               parameters,
                                                                               testingParam.fluxCalculator,
                                                                               nullptr /*options*/,
                                                                               std::vector<std::shared_ptr<mathFunctions::FieldSolution>>{initialCondition} /*initialization*/,
                                                                               boundaryConditions /*boundary conditions*/,
                                                                               std::vector<std::shared_ptr<mathFunctions::FieldSolution>>{} /*exactSolution*/);

            // Complete the problem setup
            flowObject->CompleteProblemSetup(ts);

            // Setup the TS
            TSSetFromOptions(ts) >> testErrorChecker;
            TSSetMaxTime(ts, testingParam.maxTime) >> testErrorChecker;

            TSSolve(ts, flowObject->GetSolutionVector()) >> testErrorChecker;

            // extract the results
            std::map<std::string, std::vector<PetscReal>> results;
            Extract1DPrimitives(flowObject->GetDM(), flowObject->GetSolutionVector(), results) >> testErrorChecker;

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

            // Cleanup
            TSDestroy(&ts) >> testErrorChecker;
        }
        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, CompressibleShockTubeTestFixture,
    testing::Values(
        (CompressibleShockTubeParameters){
            .mpiTestParameter = {.testName = "ausm case 1 sod problem", .nproc = 1, .arguments = ""},
            .initialConditions = {.gamma = 1.4, .length = 1.0, .rhoL = 1.0, .uL = 0.0, .pL = 1.0, .rhoR = 0.125, .uR = 0.0, .pR = .1},
            .fluxCalculator = std::make_shared<flow::fluxCalculator::Ausm>(),
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
                 {"rho", {1.000000, 1.000000, 0.999999, 0.999998, 0.999996, 0.999989, 0.999975, 0.999942, 0.999871, 0.999725, 0.999441, 0.998912, 0.997975, 0.996399, 0.993890, 0.990103, 0.984695,
                          0.977377, 0.967979, 0.956482, 0.943021, 0.927851, 0.911289, 0.893663, 0.875271, 0.856366, 0.837147, 0.817765, 0.798335, 0.778940, 0.759642, 0.740485, 0.721504, 0.702719,
                          0.684148, 0.665802, 0.647686, 0.629804, 0.612155, 0.594736, 0.577542, 0.560563, 0.543786, 0.527192, 0.510755, 0.494434, 0.478165, 0.461831, 0.445190, 0.427532, 0.393912,
                          0.415146, 0.422115, 0.422982, 0.423021, 0.422932, 0.422794, 0.422614, 0.422383, 0.422079, 0.421658, 0.421042, 0.420103, 0.418652, 0.416430, 0.413118, 0.408376, 0.401900,
                          0.393501, 0.383170, 0.371125, 0.357811, 0.343844, 0.329924, 0.316728, 0.304818, 0.294573, 0.286169, 0.279594, 0.274686, 0.271193, 0.268825, 0.267298, 0.266361, 0.265814,
                          0.265499, 0.265295, 0.265091, 0.264742, 0.263983, 0.262243, 0.258253, 0.249244, 0.229571, 0.192632, 0.151401, 0.131051, 0.126044, 0.125167, 0.125027}},
                 {"u", {0.000000, 0.000000, 0.000001, 0.000002, 0.000006, 0.000016, 0.000037, 0.000085, 0.000186, 0.000392, 0.000790, 0.001522, 0.002802, 0.004928, 0.008275, 0.013267, 0.020322,
                        0.029784, 0.041858, 0.056575, 0.073797, 0.093261, 0.114638, 0.137590, 0.161809, 0.187036, 0.213065, 0.239738, 0.266936, 0.294572, 0.322581, 0.350915, 0.379542, 0.408437,
                        0.437587, 0.466983, 0.496621, 0.526507, 0.556646, 0.587052, 0.617745, 0.648751, 0.680106, 0.711860, 0.744084, 0.776883, 0.810424, 0.845003, 0.881233, 0.920863, 0.999721,
                        0.947418, 0.931020, 0.928750, 0.928334, 0.928169, 0.928067, 0.927990, 0.927927, 0.927873, 0.927831, 0.927801, 0.927790, 0.927804, 0.927850, 0.927938, 0.928077, 0.928274,
                        0.928533, 0.928848, 0.929200, 0.929552, 0.929854, 0.930051, 0.930100, 0.929978, 0.929698, 0.929300, 0.928843, 0.928382, 0.927963, 0.927607, 0.927317, 0.927078, 0.926861,
                        0.926618, 0.926255, 0.925575, 0.924131, 0.920896, 0.913500, 0.896521, 0.857687, 0.769584, 0.579653, 0.281600, 0.072662, 0.012872, 0.002063, 0.000323}},
                 {"e", {2.500000, 2.500000, 2.499999, 2.499998, 2.499996, 2.499989, 2.499975, 2.499942, 2.499871, 2.499725, 2.499441, 2.498910, 2.497971, 2.496387, 2.493857, 2.490022, 2.484512,
                        2.477002, 2.467273, 2.455259, 2.441053, 2.424878, 2.407037, 2.387851, 2.367621, 2.346600, 2.324993, 2.302955, 2.280606, 2.258031, 2.235298, 2.212454, 2.189536, 2.166570,
                        2.143576, 2.120566, 2.097551, 2.074534, 2.051516, 2.028496, 2.005465, 1.982413, 1.959321, 1.936162, 1.912897, 1.889461, 1.865751, 1.841578, 1.816540, 1.789471, 1.737037,
                        1.774517, 1.786013, 1.788048, 1.788912, 1.789683, 1.790508, 1.791445, 1.792559, 1.793957, 1.795827, 1.798495, 1.802498, 1.808655, 1.818120, 1.832381, 1.853187, 1.882391,
                        1.921721, 1.972507, 2.035374, 2.109961, 2.194686, 2.286656, 2.381806, 2.475340, 2.562443, 2.639100, 2.702755, 2.752599, 2.789395, 2.815003, 2.831784, 2.842100, 2.847983,
                        2.850984, 2.852143, 2.851980, 2.850423, 2.846502, 2.837507, 2.816975, 2.770113, 2.664507, 2.452125, 2.183781, 2.040254, 2.006730, 2.001068, 2.000172}}}},
        (CompressibleShockTubeParameters){
            .mpiTestParameter = {.testName = "case 2 expansion left and expansion right", .nproc = 1, .arguments = ""},
            .initialConditions = {.gamma = 1.4, .length = 1.0, .rhoL = 1.0, .uL = -2.0, .pL = 0.4, .rhoR = 1.0, .uR = 2.0, .pR = 0.4},
            .fluxCalculator = std::make_shared<flow::fluxCalculator::Ausm>(),
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
                 {"rho", {0.989370, 0.983478, 0.975333, 0.964536, 0.950769, 0.933827, 0.913639, 0.890262, 0.863874, 0.834745, 0.803213, 0.769654, 0.734463, 0.698033, 0.660744, 0.622954, 0.584997,
                          0.547177, 0.509771, 0.473030, 0.437175, 0.402402, 0.368881, 0.336756, 0.306145, 0.277145, 0.249823, 0.224228, 0.200381, 0.178288, 0.157930, 0.139274, 0.122270, 0.106854,
                          0.092948, 0.080454, 0.069309, 0.059648, 0.051263, 0.043930, 0.037501, 0.031857, 0.026904, 0.022565, 0.018781, 0.015493, 0.013453, 0.012662, 0.012181, 0.010831, 0.010831,
                          0.012181, 0.012662, 0.013453, 0.015493, 0.018781, 0.022565, 0.026904, 0.031857, 0.037501, 0.043930, 0.051263, 0.059648, 0.069309, 0.080454, 0.092948, 0.106854, 0.122270,
                          0.139274, 0.157930, 0.178288, 0.200381, 0.224228, 0.249823, 0.277145, 0.306145, 0.336756, 0.368881, 0.402402, 0.437175, 0.473030, 0.509771, 0.547177, 0.584997, 0.622954,
                          0.660744, 0.698033, 0.734463, 0.769654, 0.803213, 0.834745, 0.863874, 0.890262, 0.913639, 0.933827, 0.950769, 0.964536, 0.975333, 0.983478, 0.989370}},
                 {"u", {-1.992044, -1.987626, -1.981501, -1.973348, -1.962890, -1.949913, -1.934282, -1.915934, -1.894879, -1.871181, -1.844941, -1.816292, -1.785374, -1.752333, -1.717309,
                        -1.680435, -1.641828, -1.601593, -1.559821, -1.516587, -1.471954, -1.425974, -1.378685, -1.330118, -1.280293, -1.229223, -1.176915, -1.123368, -1.068577, -1.012532,
                        -0.955220, -0.896623, -0.836724, -0.775504, -0.712947, -0.649048, -0.583112, -0.513602, -0.443886, -0.374336, -0.304362, -0.233452, -0.161112, -0.086851, -0.010042,
                        0.061113,  0.093517,  0.073487,  0.025623,  -0.000326, 0.000326,  -0.025623, -0.073487, -0.093517, -0.061113, 0.010042,  0.086851,  0.161112,  0.233452,  0.304362,
                        0.374336,  0.443886,  0.513602,  0.583112,  0.649048,  0.712947,  0.775504,  0.836724,  0.896623,  0.955220,  1.012532,  1.068577,  1.123368,  1.176915,  1.229223,
                        1.280293,  1.330118,  1.378685,  1.425974,  1.471954,  1.516587,  1.559821,  1.601593,  1.641828,  1.680435,  1.717309,  1.752333,  1.785374,  1.816292,  1.844941,
                        1.871181,  1.894879,  1.915934,  1.934282,  1.949913,  1.962890,  1.973348,  1.981501,  1.987626,  1.992044}},
                 {"e", {0.995787, 0.993472, 0.990287, 0.986086, 0.980756, 0.974227, 0.966477, 0.957534, 0.947465, 0.936370, 0.924374, 0.911616, 0.898241, 0.884393, 0.870216, 0.855841, 0.841391,
                        0.826977, 0.812698, 0.798636, 0.784865, 0.771443, 0.758417, 0.745822, 0.733682, 0.722012, 0.710820, 0.700103, 0.689855, 0.680061, 0.670706, 0.661768, 0.653228, 0.645078,
                        0.637340, 0.630157, 0.624440, 0.621045, 0.616929, 0.611724, 0.605730, 0.599135, 0.592070, 0.584622, 0.576800, 0.568842, 0.581491, 0.626016, 0.700842, 0.848200, 0.848200,
                        0.700842, 0.626016, 0.581491, 0.568842, 0.576800, 0.584622, 0.592070, 0.599135, 0.605730, 0.611724, 0.616929, 0.621045, 0.624440, 0.630157, 0.637340, 0.645078, 0.653228,
                        0.661768, 0.670706, 0.680061, 0.689855, 0.700103, 0.710820, 0.722012, 0.733682, 0.745822, 0.758417, 0.771443, 0.784865, 0.798636, 0.812698, 0.826977, 0.841391, 0.855841,
                        0.870216, 0.884393, 0.898241, 0.911616, 0.924374, 0.936370, 0.947465, 0.957534, 0.966477, 0.974227, 0.980756, 0.986086, 0.990287, 0.993472, 0.995787}}}},
        (CompressibleShockTubeParameters){
            .mpiTestParameter = {.testName = "case 5 shock collision shock left and shock right", .nproc = 1, .arguments = ""},
            .initialConditions = {.gamma = 1.4, .length = 1.0, .rhoL = 5.99924, .uL = 19.5975, .pL = 460.894, .rhoR = 5.99242, .uR = -6.19633, .pR = 46.0950},
            .fluxCalculator = std::make_shared<flow::fluxCalculator::Ausm>(),
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
                          5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  5.999240,  6.001868,  8.242960,  14.204950, 14.274383, 14.266546, 14.263534, 14.264623, 14.268066, 14.272052,
                          14.275351, 14.277597, 14.279322, 14.281827, 14.287025, 14.297497, 14.316914, 14.350802, 14.407475, 14.498850, 14.640885, 14.853394, 15.158935, 15.580542, 16.138348,
                          16.845523, 17.705208, 18.710339, 19.841725, 21.068084, 22.345882, 23.621770, 24.862369, 26.047229, 27.126573, 28.072856, 28.865510, 29.495294, 29.955397, 30.220687,
                          30.140594, 29.536421, 27.432257, 13.785880, 6.233767,  5.992420,  5.992420,  5.992420,  5.992420,  5.992420}},
                 {"u", {19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500,
                        19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500,
                        19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500,
                        19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.597500, 19.592953, 15.209942, 8.782093,  8.720759,  8.717512,  8.713798,  8.710309,  8.707504,  8.705639,
                        8.704638,  8.704087,  8.703383,  8.701985,  8.699647,  8.696496,  8.692943,  8.689483,  8.686495,  8.684117,  8.682151,  8.680010,  8.676821,  8.671815,  8.664988,
                        8.657563,  8.651394,  8.647574,  8.646847,  8.649765,  8.657442,  8.671060,  8.683888,  8.688372,  8.689909,  8.688798,  8.685904,  8.680085,  8.667174,  8.636023,
                        8.570417,  8.352693,  7.766784,  4.496187,  -5.422600, -6.196330, -6.196330, -6.196330, -6.196330, -6.196330}},
                 {"e",
                  {192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495,
                   192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495,
                   192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495,
                   192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.063495, 192.097277, 241.196515, 295.026642, 295.666546, 295.897938, 296.026411, 296.049135, 295.993510, 295.895216,
                   295.787895, 295.696809, 295.633094, 295.588862, 295.533576, 295.410649, 295.132258, 294.570952, 293.549398, 291.833765, 289.139614, 285.158719, 279.609220, 272.301174, 263.199638,
                   252.466617, 240.466140, 227.713044, 214.781004, 202.210206, 190.439033, 179.784673, 170.463187, 162.563018, 156.021637, 150.746516, 146.600564, 143.420897, 141.007097, 139.086731,
                   137.198973, 134.830308, 129.505969, 91.027856,  23.976675,  19.230545,  19.230545,  19.230545,  19.230545,  19.230545}}}}),
    [](const testing::TestParamInfo<CompressibleShockTubeParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
