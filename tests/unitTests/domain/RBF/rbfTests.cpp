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


void RBFTestFixture_SetData(ablate::solver::Range cellRange, const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  PetscReal    *array, *val, x[3] = {0.0, 0.0, 0.0};
  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);

  VecGetArray(vec, &array) >> utilities::PetscUtilities::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> utilities::PetscUtilities::checkError;
    DMPlexPointLocalFieldRef(dm, cell, field->id, array, &val) >> utilities::PetscUtilities::checkError;
    *val = 1.0 + sin(4.0*x[0] + 3.0*x[1] - x[2]) + sin(x[0] - x[1]) - cos(2.0*x[0]+x[2]) - sin(x[1]-3.0*x[2])+cos(3.0*x[0]*x[1]*x[2]);
  }
  VecRestoreArray(vec, &array) >> utilities::PetscUtilities::checkError;

}


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

// Tests the radial functions RBFVal and RBFDer, which returns the function value between two locations and the derivative at a location, respectively.
TEST_P(RBFTestFixture_RBFValues, CheckRBFFunctions) {
    StartWithMPI
        {
            // initialize petsc and mpi
            environment::RunEnvironment::Initialize(argc, argv);
            utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();
            std::shared_ptr<domain::rbf::RBF> rbf = testingParam.rbf;

            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>>{},
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
                                              .expectedDerivatives = {1.0/M_E, 1.0/M_E, 1.0/(2.0*M_E)}},
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
                                              .expectedDerivatives = {1.0/exp(5.0), 1.0/exp(5.0), 0.5/exp(5.0), 2.0/exp(5.0), 3.5/exp(5.0), 2.0/exp(5.0)}},
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
                                              .expectedDerivatives = {1.0/exp(6.0), 1.0/exp(6.0), 0.5/exp(6.0), 2.0/exp(6.0), 3.5/exp(6.0), 2.0/exp(6.0), -1.0/exp(6.0), 0.5/exp(6.0), -1.0/exp(6.0), -2.0/exp(6.0), -2.0/exp(6.0)}},
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
                                              .expectedDerivatives = {1.0/sqrt(2.0), 0.25/sqrt(2.0), 0.0625/sqrt(2.0)}},
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
                                              .expectedDerivatives = {1.0/sqrt(6.0), 1.0/(12.0*sqrt(6.0)), -1.0/(48.0*sqrt(6.0)), 1.0/(6.0*sqrt(6.0)), 1.0/(24.0*sqrt(6.0)), 1.0/(24.0*sqrt(6.0))}},
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
                                              .expectedDerivatives = {1.0/sqrt(15.0), 1.0/(30.0*sqrt(15.0)), -1.0/(75.0*sqrt(15.0)), 1.0/(15.0*sqrt(15.0)), -1.0/(300.0*sqrt(15.0)), 1.0/(150.0*sqrt(15.0)), 1.0/(10.0*sqrt(15.0)), 1.0/(75.0*sqrt(15.0)), 1.0/(100.0*sqrt(15.0)), 1.0/(50.0*sqrt(15.0)), 1.0/(300.0*sqrt(15.0))}},
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
                                              .expectedDerivatives = {sqrt(2.0), -0.5/sqrt(2.0), 0.125/sqrt(2.0)}},
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
                                              .expectedDerivatives = {sqrt(6.0), -0.5/sqrt(6.0), 5.0/(24.0*sqrt(6.0)), -1.0/sqrt(6.0), 1.0/(12.0*sqrt(6.0)), -1.0/(12.0*sqrt(6.0))}},
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
                                              .expectedDerivatives = {sqrt(15.0), -0.5/sqrt(15.0), 7.0/(30.0*sqrt(15.0)), -1.0/sqrt(15.0), 11.0/(60.0*sqrt(15.0)), -1.0/(30.0*sqrt(15.0)), -0.5*sqrt(3.0/5.0), 1.0/(10.0*sqrt(15.0)), -1.0/(20.0*sqrt(15.0)), -1.0/(10.0*sqrt(15.0)), -1.0/(100.0*sqrt(15.0))}},
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
                                              .expectedDerivatives = {32.0, -80.0, 160.0}},
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
                                              .expectedDerivatives = {800.0*sqrt(5.0), -400.0*sqrt(5.0), 320.0*sqrt(5.0), -800.0*sqrt(5.0), 680.0*sqrt(5.0), 240.0*sqrt(5.0)}},
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
                                              .expectedDerivatives = {6272.0*sqrt(14.0), -1120.0*sqrt(14.0), 680.0*sqrt(14.0), -2240.0*sqrt(14.0), 1040.0*sqrt(14.0), 240.0*sqrt(14.0), -3360.0*sqrt(14.0), 1640.0*sqrt(14.0), 360.0*sqrt(14.0), 720.0*sqrt(14.0), -180.0*sqrt(2.0/7.0)}}
                  ),
    [](const testing::TestParamInfo<RBFParameters_RBFValues>& info) { return info.param.mpiTestParameter.getTestName(); });




struct RBFParameters_Derivative {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    bool meshSimplex;
    std::vector<std::shared_ptr<domain::rbf::RBF>> rbfList;
    PetscInt c;
    std::vector<PetscInt> dx;
    std::vector<PetscInt> dy;
    std::vector<PetscInt> dz;
    std::vector<PetscReal> expectedDerivatives;
    PetscReal tolerance;
};

class RBFTestFixture_Derivative : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_Derivative> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};


// This tests single-cell derivative functions.
TEST_P(RBFTestFixture_Derivative, CheckPointFunctions) {
    StartWithMPI

    {
      // initialize petsc and mpi
      environment::RunEnvironment::Initialize(argc, argv);
      utilities::PetscUtilities::Initialize();
      auto testingParam = GetParam();
      std::vector<std::shared_ptr<domain::rbf::RBF>> rbfList = testingParam.rbfList;


      //             Make the field
      std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

      //             Create the mesh
//      Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
      auto mesh = std::make_shared<domain::BoxMesh>(
      "mesh", fieldDescriptor, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)}, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd, std::vector<std::string>{}, testingParam.meshSimplex);

      mesh->InitializeSubDomains();

      std::shared_ptr<ablate::domain::SubDomain> subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);


      // The field containing the data
      const ablate::domain::Field *field = &(subDomain->GetField("fieldA"));


      ablate::solver::Range cellRange;
      for (long int j = 0; j < rbfList.size(); ++j) {
        rbfList[j]->Setup(subDomain);

        // Initialize
        rbfList[j]->GetCellRange(subDomain, nullptr, cellRange);
        rbfList[j]->Initialize(cellRange);
        rbfList[j]->RestoreRange(cellRange);
      }

      // Now set the data using the first RBF. All will use the same data
      rbfList[0]->GetCellRange(subDomain, nullptr, cellRange);
      RBFTestFixture_SetData(cellRange, field, subDomain);
      rbfList[0]->RestoreRange(cellRange);

      // Now check derivatives
      std::vector<PetscInt> dx = testingParam.dx, dy = testingParam.dy, dz = testingParam.dz;
      PetscInt c = testingParam.c;
      PetscReal expectedVal, howClose;

      for (int i = 0; i < dx.size(); ++i) {
        expectedVal = testingParam.expectedDerivatives[i];
        howClose = (testingParam.tolerance)*PetscAbsReal(expectedVal);
//      Make sure that the value is within 1e-7 of the true derivative calculated to double-precision using Mathematica.
        for (long int j = 0; j < rbfList.size(); ++j) {
          ASSERT_NEAR(rbfList[j]->EvalDer(field, c, dx[i], dy[i], dz[i]), expectedVal, howClose) << "RBF: " << rbfList[j]->type() << ", dx: " << dx[i] << ", dy:" << dy[i] << ", dz: " << dz[i];
        }
      }
      ablate::environment::RunEnvironment::Finalize();
      exit(0);

    }

    EndWithMPI
}



INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_Derivative,
    testing::Values((RBFParameters_Derivative){.mpiTestParameter = {.testName = "2D_Center"},
                                              .meshFaces = {50, 50},
                                              .meshStart = {0.4875, 0.4875},
                                              .meshEnd = {0.5125, 0.5125},
                                              .meshSimplex = false,
                                              .rbfList = {std::make_shared<domain::rbf::GA>(4, 0.001, true, false), std::make_shared<domain::rbf::MQ>(4, 0.001, true, false), std::make_shared<domain::rbf::IMQ>(4, 0.001, true, false), std::make_shared<domain::rbf::PHS>(4, 3, true, false)},
                                              .c = 1275,
                                              .dx = {0, 1, 2, 0, 1, 0},
                                              .dy = {0, 0, 0, 1, 1, 2},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {0.62805208896495, -1.0598834704955244, 7.798269834129567, -4.684986827102714, 4.229057867030736, 3.651438319533394},
                                              .tolerance = 1.0e-7},
                    (RBFParameters_Derivative){.mpiTestParameter = {.testName = "2D_XEdge"},
                                              .meshFaces = {50, 50},
                                              .meshStart = {0.4875, 0.4875},
                                              .meshEnd = {0.5125, 0.5125},
                                              .meshSimplex = false,
                                              .rbfList = {std::make_shared<domain::rbf::GA>(4, 0.001, true, false), std::make_shared<domain::rbf::MQ>(4, 0.001, true, false), std::make_shared<domain::rbf::IMQ>(4, 0.001, true, false), std::make_shared<domain::rbf::PHS>(4, 3, true, false)},
                                              .c = 1250,
                                              .dx = {0, 1, 2, 0, 1, 0},
                                              .dy = {0, 0, 0, 1, 1, 2},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {0.6418927948463902, -1.1532556194062509, 7.1388518673486825, -4.734241228746797, 3.6500020727390305, 3.239020904157607},
                                              .tolerance = 1.0e-7},
                    (RBFParameters_Derivative){.mpiTestParameter = {.testName = "2D_YEdge"},
                                              .meshFaces = {50, 50},
                                              .meshStart = {0.4875, 0.4875},
                                              .meshEnd = {0.5125, 0.5125},
                                              .meshSimplex = false,
                                              .rbfList = {std::make_shared<domain::rbf::GA>(4, 0.001, true, false), std::make_shared<domain::rbf::MQ>(4, 0.002, true, false), std::make_shared<domain::rbf::IMQ>(4, 0.001, true, false), std::make_shared<domain::rbf::PHS>(4, 3, true, false)},
                                              .c = 2475,
                                              .dx = {0, 1, 2, 0, 1, 0},
                                              .dy = {0, 0, 0, 1, 1, 2},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {0.5721029428579234, -1.006792297334477, 8.345544245254365, -4.639223671294841, 4.618514179370705, 3.975024815201783},
                                              .tolerance = 1.0e-7},
                    (RBFParameters_Derivative){.mpiTestParameter = {.testName = "2D_Corner"},
                                              .meshFaces = {50, 50},
                                              .meshStart = {0.4875, 0.4875},
                                              .meshEnd = {0.5125, 0.5125},
                                              .meshSimplex = false,
                                              .rbfList = {std::make_shared<domain::rbf::GA>(4, 0.001, true, false), std::make_shared<domain::rbf::MQ>(4, 0.002, true, false), std::make_shared<domain::rbf::IMQ>(4, 0.001, true, false), std::make_shared<domain::rbf::PHS>(4, 3, true, false)},
                                              .c = 0,
                                              .dx = {0, 1, 2, 0, 1, 0},
                                              .dy = {0, 0, 0, 1, 1, 2},
                                              .dz = {0, 0, 0, 0, 0, 0},
                                              .expectedDerivatives = {0.7013148528341804, -1.1962700487490516, 6.551681570646172, -4.772565622202147, 3.2314987805551407, 2.8922635350594175},
                                              .tolerance = 2.0e-7}
                  ),
    [](const testing::TestParamInfo<RBFParameters_Derivative>& info) { return info.param.mpiTestParameter.getTestName(); });

