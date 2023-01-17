#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "PetscTestErrorChecker.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/RBF/ga.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"

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
    std::vector<PetscReal> expectedDerivatives; // Expected derivative values
};

class RBFTestFixture_RBFValues : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_RBFValues> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

// Tests the actual radial functions
TEST_P(RBFTestFixture_RBFValues, CheckRBFFunctions) {
    StartWithMPI
        {
            // initialize petsc and mpi
            environment::RunEnvironment::Initialize(argc, argv);
            utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();
            std::shared_ptr<domain::rbf::RBF> rbf = testingParam.rbf;

            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", std::vector<std::shared_ptr<domain::FieldDescriptor>>{},
                std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd);

            mesh->InitializeSubDomains();

            auto subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

            rbf->Setup(subDomain);

            PetscReal *x0 = &(testingParam.x0[0]), *x = &(testingParam.x[0]);
            std::vector<PetscInt> dx = testingParam.dx, dy = testingParam.dy, dz = testingParam.dz;

            // Check the radial function value
            EXPECT_DOUBLE_EQ(testingParam.expectedValue, rbf->RBFVal(x0, x));

            // Now check derivatives
            for (int i = 0; i < dx.size(); ++i) {
              EXPECT_DOUBLE_EQ(testingParam.expectedDerivatives[i], rbf->RBFDer(x, dx[i], dy[i], dz[i]));
            }




//GA::GA(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//IMQ::IMQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//MQ::MQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//PHS::PHS(PetscInt p, PetscInt phsOrder, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), phsOrder(phsOrder) {};





        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}



INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_RBFValues,
    testing::Values((RBFParameters_RBFValues){.mpiTestParameter = {.testName = "GA1D"},
                                              .meshFaces = {1},
                                              .meshStart = {0.0},
                                              .meshEnd = {1.0},
                                              .rbf = std::make_shared<domain::rbf::GA>(8, 2, true, false),
                                              .x0 = {1.0},
                                              .x = {2.0},
                                              .expectedValue = 1.0/exp(0.25),
                                              .dx = {0, 1, 2},
                                              .dy = {0, 0, 0},
                                              .dz = {0, 0, 0},
                                              .expectedDerivatives = {1.0/M_E, -1.0/M_E, 1.0/(2.0*M_E)}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "GA2D"},
                                              .meshFaces = {1, 1},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::GA>(8, 2, true, false),
                                              .x0 = {1.0, 3.0},
                                              .x = {2.0, 4.0},
                                              .expectedValue = 1.0/sqrt(M_E),
                                              .dx = {0, 1, 2, 0, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {1.0/exp(5.0), -1.0/exp(5.0), 0.5/exp(5.0), -2.0/exp(5.0), 3.5/exp(5.0), 2.0/exp(5.0)}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "GA3D"},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {1.0, 1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::GA>(8, 2, true, false),
                                              .x0 = {1.0, 3.0, 5.0},
                                              .x = {2.0, 4.0, -2.0},
                                              .expectedValue = 1.0/exp(51.0/4.0),
                                              .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                              .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                              .expectedDerivatives = {1.0/exp(6.0), -1.0/exp(6.0), 0.5/exp(6.0), -2.0/exp(6.0), 3.5/exp(6.0), 2.0/exp(6.0), 1.0/exp(6.0), 0.5/exp(6.0), -1.0/exp(6.0), -2.0/exp(6.0), 2.0/exp(6.0)}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "IMQ1D"},
                                              .meshFaces = {1},
                                              .meshStart = {0.0},
                                              .meshEnd = {1.0},
                                              .rbf = std::make_shared<domain::rbf::IMQ>(8, 2, true, false),
                                              .x0 = {1.0},
                                              .x = {2.0},
                                              .expectedValue = 2.0/sqrt(5.0),
                                              .dx = {0, 1, 2},
                                              .dy = {0, 0, 0},
                                              .dz = {0, 0, 0},
                                              .expectedDerivatives = {1.0/sqrt(2.0), -0.25/sqrt(2.0), 0.0625/sqrt(2.0)}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "IMQ2D"},
                                              .meshFaces = {1, 1},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::IMQ>(8, 2, true, false),
                                              .x0 = {1.0, 3.0},
                                              .x = {2.0, 4.0},
                                              .expectedValue = sqrt(2.0/3.0),
                                              .dx = {0, 1, 2, 0, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {1.0/sqrt(6.0), -1.0/(12.0*sqrt(6.0)), -1.0/(48.0*sqrt(6.0)), -1.0/(6.0*sqrt(6.0)), 1.0/(24.0*sqrt(6.0)), 1.0/(24.0*sqrt(6.0))}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "IMQ3D"},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {1.0, 1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::IMQ>(8, 2, true, false),
                                              .x0 = {1.0, 3.0, 5.0},
                                              .x = {2.0, 4.0, 6.0},
                                              .expectedValue = 2.0/sqrt(7.0),
                                              .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                              .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                              .expectedDerivatives = {1.0/sqrt(15.0), -1.0/(30.0*sqrt(15.0)), -1.0/(75.0*sqrt(15.0)), -1.0/(15.0*sqrt(15.0)), -1.0/(300.0*sqrt(15.0)), 1.0/(150.0*sqrt(15.0)), -1.0/(10.0*sqrt(15.0)), 1.0/(75.0*sqrt(15.0)), 1.0/(100.0*sqrt(15.0)), 1.0/(50.0*sqrt(15.0)), -1.0/(300.0*sqrt(15.0))}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "MQ1D"},
                                              .meshFaces = {1},
                                              .meshStart = {0.0},
                                              .meshEnd = {1.0},
                                              .rbf = std::make_shared<domain::rbf::MQ>(8, 2, true, false),
                                              .x0 = {1.0},
                                              .x = {2.0},
                                              .expectedValue = 0.5*sqrt(5.0),
                                              .dx = {0, 1, 2},
                                              .dy = {0, 0, 0},
                                              .dz = {0, 0, 0},
                                              .expectedDerivatives = {sqrt(2.0), 0.5/sqrt(2.0), 0.125/sqrt(2.0)}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "MQ2D"},
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
                                              .expectedDerivatives = {sqrt(6.0), 0.5/sqrt(6.0), 5.0/(24.0*sqrt(6.0)), 1.0/sqrt(6.0), 1.0/(12.0*sqrt(6.0)), -1.0/(12.0*sqrt(6.0))}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "MQ3D"},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {1.0, 1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::MQ>(8, 2, true, false),
                                              .x0 = {1.0, 3.0, 5.0},
                                              .x = {2.0, 4.0, 6.0},
                                              .expectedValue = 0.5*sqrt(7.0),
                                              .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                              .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                              .expectedDerivatives = {sqrt(15.0), 0.5/sqrt(15.0), 7.0/(30.0*sqrt(15.0)), 1.0/sqrt(15.0), 11.0/(60.0*sqrt(15.0)), -1.0/(30.0*sqrt(15.0)), 0.5*sqrt(3.0/5.0), 1.0/(10.0*sqrt(15.0)), -1.0/(20.0*sqrt(15.0)), -1.0/(10.0*sqrt(15.0)), 1.0/(100.0*sqrt(15.0))}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "PHS1D"},
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
                                              .expectedDerivatives = {32.0, 80.0, 160.0}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "PHS2D"},
                                              .meshFaces = {1, 1},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::PHS>(8, 2, true, false),
                                              .x0 = {1.0, 3.0},
                                              .x = {2.0, 4.0},
                                              .expectedValue = 4.0*sqrt(2.0),
                                              .dx = {0, 1, 2, 0, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {800.0*sqrt(5.0), 400.0*sqrt(5.0), 320.0*sqrt(5.0), 800.0*sqrt(5.0), 680.0*sqrt(5.0), 240.0*sqrt(5.0)}},
                    (RBFParameters_RBFValues){.mpiTestParameter = {.testName = "PHS3D"},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {1.0, 1.0, 1.0},
                                              .rbf = std::make_shared<domain::rbf::PHS>(8, 2, true, false),
                                              .x0 = {1.0, 3.0, 5.0},
                                              .x = {2.0, 4.0, 6.0},
                                              .expectedValue = 9.0*sqrt(3.0),
                                              .dx = {0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 1},
                                              .dy = {0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1},
                                              .dz = {0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1},
                                              .expectedDerivatives = {6272.0*sqrt(14.0), 1120.0*sqrt(14.0), 680.0*sqrt(14.0), 2240.0*sqrt(14.0), 1040.0*sqrt(14.0), 240.0*sqrt(14.0), 3360.0*sqrt(14.0), 1640.0*sqrt(14.0), 360.0*sqrt(14.0), 720.0*sqrt(14.0), 180.0*sqrt(2.0/7.0)}}
                  ),
    [](const testing::TestParamInfo<RBFParameters_RBFValues>& info) { return info.param.mpiTestParameter.getTestName(); });




struct RBFParameters_Derivative {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    bool meshSimplex;
    std::shared_ptr<domain::rbf::RBF> rbf;
    PetscInt c;
    std::vector<PetscInt> dx;
    std::vector<PetscInt> dy;
    std::vector<PetscInt> dz;
    std::vector<PetscReal> expectedDerivatives;
};

class RBFTestFixture_Derivative : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_Derivative> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};


void RBFTestFixture_SetData(ablate::solver::Range cellRange, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  DM            dm = subDomain->GetDM();
  PetscReal    *array, *val;
  Vec           solVec = subDomain->GetSolutionVector();
  const ablate::domain::Field field = subDomain->GetField("fieldA");

  VecGetArray(solVec, &array) >> ablate::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    DMPlexPointLocalFieldRef(dm, cell, field.id, array, &val)  >> ablate::checkError;
    *val = 1.0;
  }
  VecRestoreArray(solVec, &array) >> ablate::checkError;

}


// This tests single-cell functions EvalDer and Interpolate.
TEST_P(RBFTestFixture_Derivative, CheckPointFunctions) {
    StartWithMPI
        {
            // initialize petsc and mpi
            environment::RunEnvironment::Initialize(argc, argv);
            utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();

            // Make the field
            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM)};

            // Create the mesh
            // Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", fieldDescriptor, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)}, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd, std::vector<std::string>{}, testingParam.meshSimplex);

            mesh->InitializeSubDomains();

            auto subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

            testingParam.rbf->Setup(subDomain);


            // Initialize the RBF
            PetscInt depth;
            DMPlexGetDepth(mesh->GetDM(), &depth) >> checkError;
            solver::Range cellRange;
            testingParam.rbf->GetRange(subDomain, nullptr, depth, cellRange);
            testingParam.rbf->Initialize(cellRange);





            testingParam.rbf->RestoreRange(cellRange);







//GA::GA(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//IMQ::IMQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//MQ::MQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//PHS::PHS(PetscInt p, PetscInt phsOrder, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), phsOrder(phsOrder) {};





        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_Derivative,
    testing::Values((RBFParameters_Derivative){.mpiTestParameter = {.testName = "MQ_Der1", .nproc = 1},
                                              .meshFaces = {10, 10},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshSimplex = false,
                                              .rbf = std::make_shared<domain::rbf::GA>(8, 0.1, true, false)}
                  ),
    [](const testing::TestParamInfo<RBFParameters_Derivative>& info) { return info.param.mpiTestParameter.getTestName(); });
