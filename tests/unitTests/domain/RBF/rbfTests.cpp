#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

struct RBFParameters_RBFValues {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::shared_ptr<domain::rbf::RBF> rbf;
    std::vector<PetscReal> x0;  // Center location
    std::vector<PetscReal> x;   // Another location
    PetscReal expectedValue;    // Expected radial function value between x0 and x
    std::vector<PetscInt> dx;   // Derivatives to compute at location x
    std::vector<PetscInt> dy;
    std::vector<PetscInt> dz;
    std::vector<PetscReal> expectedDerivatives;  // Expected derivative values
};

class RBFTestFixture_RBFValues : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_RBFValues> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

// Tests the radial functions RBFVal and RBFDer, which returns the function value between two locations and the derivative at a location, respectively.
TEST_P(RBFTestFixture_RBFValues, CheckRBFFunctions) {
    StartWithMPI
        {
            // initialize petsc and mpi
            environment::RunEnvironment::Initialize(argc, argv);
            utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();
            std::shared_ptr<domain::rbf::RBF> rbf = testingParam.rbf;

            auto mesh = std::make_shared<domain::BoxMesh>("mesh",
                                                          std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>>{},
                                                          std::vector<std::shared_ptr<domain::modifiers::Modifier>>{},
                                                          testingParam.meshFaces,
                                                          testingParam.meshStart,
                                                          testingParam.meshEnd);

            mesh->InitializeSubDomains();

            auto subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

            rbf->Setup(subDomain);

            PetscReal *x0 = &(testingParam.x0[0]), *x = &(testingParam.x[0]);
            std::vector<PetscInt> dx = testingParam.dx, dy = testingParam.dy, dz = testingParam.dz;
            PetscInt dim = subDomain->GetDimensions();

            // Check the radial function value
            EXPECT_DOUBLE_EQ(testingParam.expectedValue, rbf->RBFVal(dim, x0, x));

            // Now check derivatives
            for (std::size_t i = 0; i < dx.size(); ++i) {
                EXPECT_DOUBLE_EQ(testingParam.expectedDerivatives[i], rbf->RBFDer(dim, x, dx[i], dy[i], dz[i]));
            }
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_RBFValues,
    testing::Values(
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("GA1D"),
                                  .meshFaces = {1},
                                  .meshStart = {0.0},
                                  .meshEnd = {1.0},
                                  .rbf = std::make_shared<domain::rbf::GA>(8, 2, true, false),
                                  .x0 = {1.0},
                                  .x = {2.0},
                                  .expectedValue = 1.0 / exp(0.25),
                                  .dx = {0, 1, 2},
                                  .dy = {0, 0, 0},
                                  .dz = {0, 0, 0},
                                  .expectedDerivatives = {1.0 / M_E, 1.0 / M_E, 1.0 / (2.0 * M_E)}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("GA2D"),
                                  .meshFaces = {1, 1},
                                  .meshStart = {0.0, 0.0},
                                  .meshEnd = {1.0, 1.0},
                                  .rbf = std::make_shared<domain::rbf::GA>(8, 2, true, false),
                                  .x0 = {1.0, 3.0},
                                  .x = {2.0, 4.0},
                                  .expectedValue = 1.0 / sqrt(M_E),
                                  .dx = {0, 1, 2, 0, 0, 1},
                                  .dy = {0, 0, 0, 1, 2, 1},
                                  .dz = {0, 0, 0, 0, 0, 0},
                                  .expectedDerivatives = {1.0 / exp(5.0), 1.0 / exp(5.0), 0.5 / exp(5.0), 2.0 / exp(5.0), 3.5 / exp(5.0), 2.0 / exp(5.0)}},
        (RBFParameters_RBFValues){
            .mpiTestParameter = testingResources::MpiTestParameter("GA3D"),
            .meshFaces = {1, 1, 1},
            .meshStart = {0.0, 0.0, 0.0},
            .meshEnd = {1.0, 1.0, 1.0},
            .rbf = std::make_shared<domain::rbf::GA>(8, 2, true, false),
            .x0 = {1.0, 3.0, 5.0},
            .x = {2.0, 4.0, -2.0},
            .expectedValue = 1.0 / exp(51.0 / 4.0),
            .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
            .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
            .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
            .expectedDerivatives =
                {1.0 / exp(6.0), 1.0 / exp(6.0), 0.5 / exp(6.0), 2.0 / exp(6.0), 3.5 / exp(6.0), 2.0 / exp(6.0), -1.0 / exp(6.0), 0.5 / exp(6.0), -1.0 / exp(6.0), -2.0 / exp(6.0), -2.0 / exp(6.0)}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("IMQ1D"),
                                  .meshFaces = {1},
                                  .meshStart = {0.0},
                                  .meshEnd = {1.0},
                                  .rbf = std::make_shared<domain::rbf::IMQ>(8, 2, true, false),
                                  .x0 = {1.0},
                                  .x = {2.0},
                                  .expectedValue = 2.0 / sqrt(5.0),
                                  .dx = {0, 1, 2},
                                  .dy = {0, 0, 0},
                                  .dz = {0, 0, 0},
                                  .expectedDerivatives = {1.0 / sqrt(2.0), 0.25 / sqrt(2.0), 0.0625 / sqrt(2.0)}},
        (RBFParameters_RBFValues){
            .mpiTestParameter = testingResources::MpiTestParameter("IMQ2D"),
            .meshFaces = {1, 1},
            .meshStart = {0.0, 0.0},
            .meshEnd = {1.0, 1.0},
            .rbf = std::make_shared<domain::rbf::IMQ>(8, 2, true, false),
            .x0 = {1.0, 3.0},
            .x = {2.0, 4.0},
            .expectedValue = sqrt(2.0 / 3.0),
            .dx = {0, 1, 2, 0, 0, 1},
            .dy = {0, 0, 0, 1, 2, 1},
            .dz = {0, 0, 0, 0, 0, 0},
            .expectedDerivatives = {1.0 / sqrt(6.0), 1.0 / (12.0 * sqrt(6.0)), -1.0 / (48.0 * sqrt(6.0)), 1.0 / (6.0 * sqrt(6.0)), 1.0 / (24.0 * sqrt(6.0)), 1.0 / (24.0 * sqrt(6.0))}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("IMQ3D"),
                                  .meshFaces = {1, 1, 1},
                                  .meshStart = {0.0, 0.0, 0.0},
                                  .meshEnd = {1.0, 1.0, 1.0},
                                  .rbf = std::make_shared<domain::rbf::IMQ>(8, 2, true, false),
                                  .x0 = {1.0, 3.0, 5.0},
                                  .x = {2.0, 4.0, 6.0},
                                  .expectedValue = 2.0 / sqrt(7.0),
                                  .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                  .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                  .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                  .expectedDerivatives = {1.0 / sqrt(15.0),
                                                          1.0 / (30.0 * sqrt(15.0)),
                                                          -1.0 / (75.0 * sqrt(15.0)),
                                                          1.0 / (15.0 * sqrt(15.0)),
                                                          -1.0 / (300.0 * sqrt(15.0)),
                                                          1.0 / (150.0 * sqrt(15.0)),
                                                          1.0 / (10.0 * sqrt(15.0)),
                                                          1.0 / (75.0 * sqrt(15.0)),
                                                          1.0 / (100.0 * sqrt(15.0)),
                                                          1.0 / (50.0 * sqrt(15.0)),
                                                          1.0 / (300.0 * sqrt(15.0))}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("MQ1D"),
                                  .meshFaces = {1},
                                  .meshStart = {0.0},
                                  .meshEnd = {1.0},
                                  .rbf = std::make_shared<domain::rbf::MQ>(8, 2, true, false),
                                  .x0 = {1.0},
                                  .x = {2.0},
                                  .expectedValue = 0.5 * sqrt(5.0),
                                  .dx = {0, 1, 2},
                                  .dy = {0, 0, 0},
                                  .dz = {0, 0, 0},
                                  .expectedDerivatives = {sqrt(2.0), -0.5 / sqrt(2.0), 0.125 / sqrt(2.0)}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("MQ2D"),
                                  .meshFaces = {1, 1},
                                  .meshStart = {0.0, 0.0},
                                  .meshEnd = {1.0, 1.0},
                                  .rbf = std::make_shared<domain::rbf::MQ>(8, 2, true, false),
                                  .x0 = {1.0, 3.0},
                                  .x = {2.0, 4.0},
                                  .expectedValue = sqrt(1.5),
                                  .dx = {0, 1, 2, 0, 0, 1},
                                  .dy = {0, 0, 0, 1, 2, 1},
                                  .dz = {0, 0, 0, 0, 0, 0},
                                  .expectedDerivatives = {sqrt(6.0), -0.5 / sqrt(6.0), 5.0 / (24.0 * sqrt(6.0)), -1.0 / sqrt(6.0), 1.0 / (12.0 * sqrt(6.0)), -1.0 / (12.0 * sqrt(6.0))}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("MQ3D"),
                                  .meshFaces = {1, 1, 1},
                                  .meshStart = {0.0, 0.0, 0.0},
                                  .meshEnd = {1.0, 1.0, 1.0},
                                  .rbf = std::make_shared<domain::rbf::MQ>(8, 2, true, false),
                                  .x0 = {1.0, 3.0, 5.0},
                                  .x = {2.0, 4.0, 6.0},
                                  .expectedValue = 0.5 * sqrt(7.0),
                                  .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                  .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                  .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                  .expectedDerivatives = {sqrt(15.0),
                                                          -0.5 / sqrt(15.0),
                                                          7.0 / (30.0 * sqrt(15.0)),
                                                          -1.0 / sqrt(15.0),
                                                          11.0 / (60.0 * sqrt(15.0)),
                                                          -1.0 / (30.0 * sqrt(15.0)),
                                                          -0.5 * sqrt(3.0 / 5.0),
                                                          1.0 / (10.0 * sqrt(15.0)),
                                                          -1.0 / (20.0 * sqrt(15.0)),
                                                          -1.0 / (10.0 * sqrt(15.0)),
                                                          -1.0 / (100.0 * sqrt(15.0))}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("PHS1D"),
                                  .meshFaces = {1},
                                  .meshStart = {0.0},
                                  .meshEnd = {1.0},
                                  .rbf = std::make_shared<domain::rbf::PHS>(8, 2, true, false),
                                  .x0 = {1.0},
                                  .x = {2.0},
                                  .expectedValue = 1.0,
                                  .dx = {0, 1, 2},
                                  .dy = {0, 0, 0},
                                  .dz = {0, 0, 0},
                                  .expectedDerivatives = {32.0, -80.0, 160.0}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("PHS2D"),
                                  .meshFaces = {1, 1},
                                  .meshStart = {0.0, 0.0},
                                  .meshEnd = {1.0, 1.0},
                                  .rbf = std::make_shared<domain::rbf::PHS>(8, 2, true, false),
                                  .x0 = {1.0, 3.0},
                                  .x = {2.0, 4.0},
                                  .expectedValue = 4.0 * sqrt(2.0),
                                  .dx = {0, 1, 2, 0, 0, 1},
                                  .dy = {0, 0, 0, 1, 2, 1},
                                  .dz = {0, 0, 0, 0, 0, 0},
                                  .expectedDerivatives = {800.0 * sqrt(5.0), -400.0 * sqrt(5.0), 320.0 * sqrt(5.0), -800.0 * sqrt(5.0), 680.0 * sqrt(5.0), 240.0 * sqrt(5.0)}},
        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("PHS3D"),
                                  .meshFaces = {1, 1, 1},
                                  .meshStart = {0.0, 0.0, 0.0},
                                  .meshEnd = {1.0, 1.0, 1.0},
                                  .rbf = std::make_shared<domain::rbf::PHS>(8, 2, true, false),
                                  .x0 = {1.0, 3.0, 5.0},
                                  .x = {2.0, 4.0, 6.0},
                                  .expectedValue = 9.0 * sqrt(3.0),
                                  .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                  .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                  .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                  .expectedDerivatives = {6272.0 * sqrt(14.0),
                                                          -1120.0 * sqrt(14.0),
                                                          680.0 * sqrt(14.0),
                                                          -2240.0 * sqrt(14.0),
                                                          1040.0 * sqrt(14.0),
                                                          240.0 * sqrt(14.0),
                                                          -3360.0 * sqrt(14.0),
                                                          1640.0 * sqrt(14.0),
                                                          360.0 * sqrt(14.0),
                                                          720.0 * sqrt(14.0),
                                                          -180.0 * sqrt(2.0 / 7.0)}}),
    [](const testing::TestParamInfo<RBFParameters_RBFValues> &info) { return info.param.mpiTestParameter.getTestName(); });

struct RBFParameters_DerivativeInterpolation {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    bool meshSimplex;
    std::vector<std::shared_ptr<domain::rbf::RBF>> rbfList;  // = {std::make_shared<domain::rbf::GA>(4, 0.001, true, false), std::make_shared<domain::rbf::MQ>(4, 0.001, true, false),
                                                             // std::make_shared<domain::rbf::IMQ>(4, 0.001, true, false), std::make_shared<domain::rbf::PHS>(4, 2, true, false)};
    std::vector<PetscInt> dx;
    std::vector<PetscInt> dy;
    std::vector<PetscInt> dz;
    PetscInt cell;
    std::vector<std::vector<PetscReal>> x;
    std::vector<PetscReal> maxError;
};

class RBFTestFixture_Derivative : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_DerivativeInterpolation> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscReal RBFTestFixture_Function(PetscReal x[3], PetscInt dx, PetscInt dy, PetscInt dz) {
    switch (dx + 10 * dy + 100 * dz) {
        case 0:  // Function
            return (1 + sin(4.0 * x[0]) + sin(4.0 * x[1]) + sin(4.0 * x[2]) + cos(x[0] + x[1] + x[2]));
        case 1:  // x
            return (4.0 * cos(4.0 * x[0]) - sin(x[0] + x[1] + x[2]));
        case 2:  // xx
            return (-cos(x[0] + x[1] + x[2]) - 16.0 * sin(4.0 * x[0]));
        case 10:  // y
            return (4.0 * cos(4.0 * x[1]) - sin(x[0] + x[1] + x[2]));
        case 11:  // xy
            return (-cos(x[0] + x[1] + x[2]));
        case 20:  // yy
            return (-cos(x[0] + x[1] + x[2]) - 16.0 * sin(4.0 * x[1]));
        case 100:  // z
            return (4.0 * cos(4.0 * x[2]) - sin(x[0] + x[1] + x[2]));
        case 101:  // xz
            return (-cos(x[0] + x[1] + x[2]));
        case 110:  // yz
            return (-cos(x[0] + x[1] + x[2]));
        case 200:  // zz
            return (-cos(x[0] + x[1] + x[2]) - 16.0 * sin(4.0 * x[2]));
        default:
            throw std::invalid_argument("Unknown derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ").");
    }
}

void RBFTestFixture_SetData(ablate::domain::Range cellRange, const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
    PetscReal *array, *val, x[3] = {0.0, 0.0, 0.0};
    Vec vec = subDomain->GetVec(*field);
    DM dm = subDomain->GetFieldDM(*field);

    VecGetArray(vec, &array) >> utilities::PetscUtilities::checkError;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;
        DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalFieldRef(dm, cell, field->id, array, &val) >> utilities::PetscUtilities::checkError;
        *val = RBFTestFixture_Function(x, 0, 0, 0);
    }
    VecRestoreArray(vec, &array) >> utilities::PetscUtilities::checkError;
}

// This tests single-cell derivative functions.
TEST_P(RBFTestFixture_Derivative, CheckDerivativeFunctions) {
    StartWithMPI
        // initialize petsc and mpi
        environment::RunEnvironment::Initialize(argc, argv);
        utilities::PetscUtilities::Initialize();
        auto testingParam = GetParam();
        std::vector<std::shared_ptr<domain::rbf::RBF>> rbfList = testingParam.rbfList;

        //             Make the field
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {
            std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

        //             Create the mesh
        //      Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
        auto mesh = std::make_shared<domain::BoxMesh>("mesh",
                                                      fieldDescriptor,
                                                      std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                                      testingParam.meshFaces,
                                                      testingParam.meshStart,
                                                      testingParam.meshEnd,
                                                      std::vector<std::string>{},
                                                      testingParam.meshSimplex);

        mesh->InitializeSubDomains();

        std::shared_ptr<ablate::domain::SubDomain> subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

        // The field containing the data
        const ablate::domain::Field *field = &(subDomain->GetField("fieldA"));

        ablate::domain::Range cellRange;
        subDomain->GetCellRange(nullptr, cellRange);
        for (std::size_t j = 0; j < rbfList.size(); ++j) {
            rbfList[j]->Setup(subDomain);       // This causes issues (I think)
            rbfList[j]->Initialize(cellRange);  //         Initialize
        }

        RBFTestFixture_SetData(cellRange, field, subDomain);

        // Now check derivatives
        std::vector<PetscInt> dx = testingParam.dx, dy = testingParam.dy, dz = testingParam.dz;
        PetscInt cell;
        PetscReal maxError;
        PetscReal x[3];
        PetscReal err = -1.0, val;
        DM dm = subDomain->GetDM();

        for (std::size_t i = 0; i < dx.size(); ++i) {  // Iterate over each of the requested derivatives
            maxError = testingParam.maxError[i];

            if (testingParam.cell > -1) {
                // 3D results take too long to run, so just check a corner
                for (std::size_t j = 0; j < rbfList.size(); ++j) {  // Check each RBF
                    PetscInt c = testingParam.cell;

                    cell = cellRange.points ? cellRange.points[c] : c;
                    val = rbfList[j]->EvalDer(field, c, dx[i], dy[i], dz[i]);

                    DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
                    err = PetscAbsReal(val - RBFTestFixture_Function(x, dx[i], dy[i], dz[i]));

                    EXPECT_LT(err, maxError) << "RBF: " << rbfList[j]->type() << ", dx: " << dx[i] << ", dy:" << dy[i] << ", dz: " << dz[i] << " Error: " << err;
                }
            } else {
                for (std::size_t j = 0; j < rbfList.size(); ++j) {  // Check each RBF
                    err = -1.0;
                    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {  // Iterate over the entire subDomain

                        cell = cellRange.points ? cellRange.points[c] : c;
                        val = rbfList[j]->EvalDer(field, c, dx[i], dy[i], dz[i]);

                        DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
                        err = PetscMax(err, PetscAbsReal(val - RBFTestFixture_Function(x, dx[i], dy[i], dz[i])));
                    }

                    EXPECT_LT(err, maxError) << "RBF: " << rbfList[j]->type() << ", dx: " << dx[i] << ", dy:" << dy[i] << ", dz: " << dz[i] << " Error: " << err;
                }
            }
        }

        subDomain->RestoreRange(cellRange);

        //    ablate::environment::RunEnvironment::Finalize();

    EndWithMPI
}

// This tests both the absolute error and the convergence for two data points
INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_Derivative,
    testing::Values(
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("1DN81_1Proc"),
                                                .meshFaces = {81},
                                                .meshStart = {-1.0},
                                                .meshEnd = {1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 2.469135802469125e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 2.469135802469125e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 2.469135802469125e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 2.469135802469125e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2},
                                                .dy = {0, 0, 0},
                                                .dz = {0, 0, 0},
                                                .cell = -1,
                                                .maxError = {2.0e-15, 2e-03, 2e-01}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("1DN161_1Proc"),
                                                .meshFaces = {161},
                                                .meshStart = {-1.0},
                                                .meshEnd = {1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 1.242236024844701e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 1.242236024844701e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 1.242236024844701e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 1.242236024844701e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2},
                                                .dy = {0, 0, 0},
                                                .dz = {0, 0, 0},
                                                .cell = -1,
                                                .maxError = {2e-15, 7e-05, 2e-02}},
        (RBFParameters_DerivativeInterpolation){
            .mpiTestParameter = testingResources::MpiTestParameter("2DQuadN21_1Proc"),
            .meshFaces = {21, 21},
            .meshStart = {-1.0, -1.0},
            .meshEnd = {1.0, 1.0},
            .meshSimplex = false,
            .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 9.523810e-02, false, false),
                        std::make_shared<ablate::domain::rbf::MQ>(4, 9.523810e-02, false, false),
                        std::make_shared<ablate::domain::rbf::IMQ>(4, 9.523810e-02, false, false),
                        std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                        std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                      std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{std::make_shared<ablate::domain::rbf::GA>(4, 9.523810e-02, false, false),
                                                                                                                             std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                      false, false)},
            .dx = {0, 1, 2, 0, 1, 0},
            .dy = {0, 0, 0, 1, 1, 2},
            .dz = {0, 0, 0, 0, 0, 0},
            .cell = -1,
            .maxError = {6.0e-15, 3.4e-02, 1.5e+00, 3.3e-02, 3.6e-01, 1.5e+00}},
        (RBFParameters_DerivativeInterpolation){
            .mpiTestParameter = testingResources::MpiTestParameter("2DQuadN41_1Proc"),
            .meshFaces = {41, 41},
            .meshStart = {-1.0, -1.0},
            .meshEnd = {1.0, 1.0},
            .meshSimplex = false,
            .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 4.878049e-02, false, false),
                        std::make_shared<ablate::domain::rbf::MQ>(4, 4.878049e-02, false, false),
                        std::make_shared<ablate::domain::rbf::IMQ>(4, 4.878049e-02, false, false),
                        std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                        std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                      std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{std::make_shared<ablate::domain::rbf::GA>(4, 4.878049e-02, false, false),
                                                                                                                             std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                      false, false)},
            .dx = {0, 1, 2, 0, 1, 0},
            .dy = {0, 0, 0, 1, 1, 2},
            .dz = {0, 0, 0, 0, 0, 0},
            .cell = -1,
            .maxError = {5.0e-15, 2.3e-03, 1.9e-01, 2.3e-03, 5.0e-02, 1.8e-01}},
        (RBFParameters_DerivativeInterpolation){
            .mpiTestParameter = testingResources::MpiTestParameter("2DTriN21_1Proc"),
            .meshFaces = {21, 21},
            .meshStart = {-1.0, -1.0},
            .meshEnd = {1.0, 1.0},
            .meshSimplex = true,
            .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 4.489567e-02, false, false),
                        std::make_shared<ablate::domain::rbf::MQ>(4, 4.489567e-02, false, false),
                        std::make_shared<ablate::domain::rbf::IMQ>(4, 4.489567e-02, false, false),
                        std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                        std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                      std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{std::make_shared<ablate::domain::rbf::GA>(4, 4.489567e-02, false, false),
                                                                                                                             std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                      false, false)},
            .dx = {0, 1, 2, 0, 1, 0},
            .dy = {0, 0, 0, 1, 1, 2},
            .dz = {0, 0, 0, 0, 0, 0},
            .cell = -1,
            .maxError = {5.0e-15, 1.5e-02, 6.2e-01, 1.6e-02, 2.6e-01, 7.6e-01}},
        (RBFParameters_DerivativeInterpolation){
            .mpiTestParameter = testingResources::MpiTestParameter("2DTriN41_1Proc"),
            .meshFaces = {41, 41},
            .meshStart = {-1.0, -1.0},
            .meshEnd = {1.0, 1.0},
            .meshSimplex = true,
            .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 2.299534e-02, false, false),
                        std::make_shared<ablate::domain::rbf::MQ>(4, 2.299534e-02, false, false),
                        std::make_shared<ablate::domain::rbf::IMQ>(4, 2.299534e-02, false, false),
                        std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                        std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                      std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{std::make_shared<ablate::domain::rbf::GA>(4, 2.299534e-02, false, false),
                                                                                                                             std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                      false, false)},
            .dx = {0, 1, 2, 0, 1, 0},
            .dy = {0, 0, 0, 1, 1, 2},
            .dz = {0, 0, 0, 0, 0, 0},
            .cell = -1,
            .maxError = {4.6e-15, 1.8e-03, 1.5e-01, 9.0e-04, 4.9e-02, 1.2e-01}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadN21_2Proc", 2),
                                                .meshFaces = {21, 21},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809490e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0},
                                                .dy = {0, 0, 0, 1, 1, 2},
                                                .dz = {0, 0, 0, 0, 0, 0},
                                                .cell = -1,
                                                .maxError = {3.7e-15, 4.4e-02, 1.8e+00, 3.3e-02, 4.2e-01, 1.4e+00}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadN41_2Proc", 2),
                                                .meshFaces = {41, 41},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 4.878048780487765e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 4.878048780487765e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 4.878048780487765e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 4.878048780487765e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0},
                                                .dy = {0, 0, 0, 1, 1, 2},
                                                .dz = {0, 0, 0, 0, 0, 0},
                                                .cell = -1,
                                                .maxError = {5.0e-15, 2.4e-03, 1.9e-01, 2.3e-03, 5.0e-02, 1.8e-01}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("2DTriN21_2Proc", 2),
                                                .meshFaces = {21, 21},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 4.489566864676461e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 4.489566864676461e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 4.489566864676461e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 4.489566864676461e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0},
                                                .dy = {0, 0, 0, 1, 1, 2},
                                                .dz = {0, 0, 0, 0, 0, 0},
                                                .cell = -1,
                                                .maxError = {5.0e-15, 2.3e-02, 9.0e-01, 1.8e-02, 4.5e-01, 7.5e-01}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("2DTriN41_2Proc", 2),
                                                .meshFaces = {41, 41},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 2.299534247761080e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 2.299534247761080e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 2.299534247761080e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 2.299534247761080e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0},
                                                .dy = {0, 0, 0, 1, 1, 2},
                                                .dz = {0, 0, 0, 0, 0, 0},
                                                .cell = -1,
                                                .maxError = {4.1e-15, 2.0e-03, 1.5e-01, 9.4e-04, 4.8e-02, 1.1e-01}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("3DQuadN21_1Proc"),
                                                .meshFaces = {21, 21, 21},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809490e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0, 0, 1, 0, 0},
                                                .dy = {0, 0, 0, 1, 1, 2, 0, 0, 1, 0},
                                                .dz = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2},
                                                .cell = 0,
                                                .maxError = {2E-15, 1.7e-02, 7.2e-01, 1.7e-02, 2.1e-03, 7.2e-01, 1.7e-02, 2.1e-03, 2.1e-03, 7.2e-01}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("3DQuadN41_1Proc"),
                                                .meshFaces = {41, 41, 41},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 4.878048780487743e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 4.878048780487743e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 4.878048780487743e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 4.878048780487743e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0, 0, 1, 0, 0},
                                                .dy = {0, 0, 0, 1, 1, 2, 0, 0, 1, 0},
                                                .dz = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2},
                                                .cell = 0,
                                                .maxError = {3.0e-15, 1.1e-03, 8.9e-02, 1.1e-03, 1.7e-04, 8.9e-02, 1.1e-03, 1.7e-04, 1.9e-04, 8.9e-02}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("3DTriN21_1Proc"),
                                                .meshFaces = {21, 21, 21},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 3.888078956798655e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0, 0, 1, 0, 0},
                                                .dy = {0, 0, 0, 1, 1, 2, 0, 0, 1, 0},
                                                .dz = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2},
                                                .cell = 0,
                                                .maxError = {1e-15, 3.3e-04, 3.5e-02, 4.0e-04, 1.8e-02, 7.0e-03, 5.0e-04, 1.5e-02, 9.0e-03, 9e-03}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("3DTriN41_1Proc"),
                                                .meshFaces = {41, 41, 41},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 1.991455075433425e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 1.991455075433425e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 1.991455075433425e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 1.991455075433425e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)},
                                                .dx = {0, 1, 2, 0, 1, 0, 0, 1, 0, 0},
                                                .dy = {0, 0, 0, 1, 1, 2, 0, 0, 1, 0},
                                                .dz = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2},
                                                .cell = 0,
                                                .maxError = {1.2e-16, 5.6e-06, 2.0e-03, 2.1e-05, 8.1e-05, 2.0e-03, 2.2e-05, 2.0e-04, 7.1e-05, 1.6e-03}}),
    [](const testing::TestParamInfo<RBFParameters_DerivativeInterpolation> &info) { return info.param.mpiTestParameter.getTestName(); });

class RBFTestFixture_Interpolation : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_DerivativeInterpolation> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

// This tests single-cell derivative functions.
TEST_P(RBFTestFixture_Interpolation, CheckInterpolationFunctions) {
    StartWithMPI

        // initialize petsc and mpi
        environment::RunEnvironment::Initialize(argc, argv);
        utilities::PetscUtilities::Initialize();
        auto testingParam = GetParam();
        std::vector<std::shared_ptr<domain::rbf::RBF>> rbfList = testingParam.rbfList;
        std::vector<std::vector<PetscReal>> x = testingParam.x;

        //             Make the field
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {
            std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

        //             Create the mesh
        //      Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
        auto mesh = std::make_shared<domain::BoxMesh>("mesh",
                                                      fieldDescriptor,
                                                      std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                                      testingParam.meshFaces,
                                                      testingParam.meshStart,
                                                      testingParam.meshEnd,
                                                      std::vector<std::string>{},
                                                      testingParam.meshSimplex);

        mesh->InitializeSubDomains();

        std::shared_ptr<ablate::domain::SubDomain> subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

        // The field containing the data
        const ablate::domain::Field *field = &(subDomain->GetField("fieldA"));

        ablate::domain::Range cellRange;
        subDomain->GetCellRange(nullptr, cellRange);
        for (std::size_t j = 0; j < rbfList.size(); ++j) {
            rbfList[j]->Setup(subDomain);       // This causes issues (I think)
            rbfList[j]->Initialize(cellRange);  //         Initialize
        }

        // Now set the data using the first RBF. All will use the same data

        RBFTestFixture_SetData(cellRange, field, subDomain);

        std::vector<PetscInt> dx = testingParam.dx, dy = testingParam.dy, dz = testingParam.dz;
        PetscReal maxError;
        PetscReal err, val, truth;

        for (std::size_t i = 0; i < x.size(); ++i) {  // Iterate over each of the requested locations
            maxError = testingParam.maxError[i];

            for (std::size_t j = 0; j < rbfList.size(); ++j) {  // Check each RBF
                truth = RBFTestFixture_Function(x[i].data(), 0, 0, 0);
                val = rbfList[j]->Interpolate(field, x[i].data());
                err = PetscAbsReal(val - truth);
                EXPECT_LT(err, maxError) << "RBF: " << rbfList[j]->type() << " Error: " << err;
            }
        }

        subDomain->RestoreRange(cellRange);

    EndWithMPI
}

// This tests both the absolute error and the convergene for two data points
INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_Interpolation,
    testing::Values(
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_1DN11"),
                                                .meshFaces = {11},
                                                .meshStart = {-1.0},
                                                .meshEnd = {1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 1.818181818181817e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 1.818181818181817e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 1.818181818181817e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 1.818181818181817e-01, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.0, 0.0}},
                                                .maxError = {2.9e-3}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_1DN21"),
                                                .meshFaces = {21},
                                                .meshStart = {-1.0},
                                                .meshEnd = {1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809512e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 9.523809523809512e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 9.523809523809512e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 9.523809523809512e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.0, 0.0}},
                                                .maxError = {5.4e-5}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_2DQuadN11"),
                                                .meshFaces = {11, 11},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 1.818181818181814e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 1.818181818181814e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 1.818181818181814e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 1.818181818181814e-01, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.0}},
                                                .maxError = {8.0e-4}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_2DQuadN21"),
                                                .meshFaces = {21, 21},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 9.523809523809490e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.0}},
                                                .maxError = {9.85e-5}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_2DTriN11"),
                                                .meshFaces = {11, 11},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 8.570991287109628e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 8.570991287109628e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 8.570991287109628e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 8.570991287109628e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.0}},
                                                .maxError = {2.5e-5}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_2DTriN21"),
                                                .meshFaces = {41, 41},
                                                .meshStart = {-1.0, -1.0},
                                                .meshEnd = {1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 4.489566864676461e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 4.489566864676461e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 4.489566864676461e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 4.489566864676461e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.0}},
                                                .maxError = {4.1e-5}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_3DQuadN11"),
                                                .meshFaces = {11, 11, 11},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 1.818181818181814e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 1.818181818181814e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 1.818181818181814e-01, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 1.818181818181814e-01, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.52}},
                                                .maxError = {1.5e-3}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_3DQuadN21"),
                                                .meshFaces = {21, 21, 21},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = false,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 9.523809523809490e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 9.523809523809490e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.52}},
                                                .maxError = {9.7e-5}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_3DTriN11"),
                                                .meshFaces = {11, 11, 11},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 7.422696190252021e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 7.422696190252021e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 7.422696190252021e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 7.422696190252021e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.52}},
                                                .maxError = {2.65e-4}},
        (RBFParameters_DerivativeInterpolation){.mpiTestParameter = testingResources::MpiTestParameter("Interp_3DTriN21"),
                                                .meshFaces = {21, 21, 21},
                                                .meshStart = {-1.0, -1.0, -1.0},
                                                .meshEnd = {1.0, 1.0, 1.0},
                                                .meshSimplex = true,
                                                .rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(-1, 3.888078956798655e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(-1, 2, false, false)},
                                                                                                          false, false)},
                                                .x = {{0.52, 0.52, 0.52}},
                                                .maxError = {1.35e-5}}),
    [](const testing::TestParamInfo<RBFParameters_DerivativeInterpolation> &info) { return info.param.mpiTestParameter.getTestName(); });
