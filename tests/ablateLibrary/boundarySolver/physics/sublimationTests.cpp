#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/physics/sublimation.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"

struct SublimationTestParameters {
    std::string description;
    // Setup
    PetscReal latentHeatOfFusion;
    PetscReal effectiveConductivity;
    std::shared_ptr<ablate::mathFunctions::MathFunction> additionalHeatTransfer = {};
    PetscInt numberSpecies = 0;
    std::shared_ptr<ablate::mathFunctions::FieldFunction> speciesMassFractions = {};

    // Geometry
    PetscInt dim;
    ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom fvFaceGeom;

    // Store the required aux variable (T)
    std::vector<double> boundaryValues;
    PetscReal boundaryTemperature;
    PetscReal stencilTemperature;
    std::vector<PetscScalar> expectedResults;
};

class SublimationTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<SublimationTestParameters> {};

TEST_P(SublimationTestFixture, ShouldComputeCorrectSourceTerm) {
    // arrange
    // get the required variables
    const auto& params = GetParam();

    // create the boundary
    auto boundary = std::make_shared<ablate::boundarySolver::physics::Sublimation>(params.latentHeatOfFusion, params.effectiveConductivity, params.speciesMassFractions, params.additionalHeatTransfer);

    // initialization is not needed for testing if species are not set
    boundary->Initialize(params.numberSpecies);

    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscInt sOff[2] = {0, (PetscInt)GetParam().expectedResults.size() - params.numberSpecies};
    const PetscScalar* allAuxStencilValues[1] = {&params.stencilTemperature};
    const PetscInt stencil[1] = {-1};
    const PetscScalar stencilWeights[3] = {1.0, 1.0, 1.0};
    // NOTE: Because this is a made of stencil value, dPhi is computed as  stencil-boundary
    // therefore: //dPhiDn = stencil-boundary ... stencil = boundary+dPhiDn

    // size up the sourceResults
    std::vector<PetscScalar> sourceResults(GetParam().expectedResults.size());

    // act
    ablate::boundarySolver::physics::Sublimation::SublimationFunction(params.dim,
                                                                      &params.fvFaceGeom,
                                                                      nullptr /*boundaryCell*/,
                                                                      uOff,
                                                                      params.boundaryValues.data(),
                                                                      nullptr,
                                                                      aOff,
                                                                      &params.boundaryTemperature /*auxValues*/,
                                                                      allAuxStencilValues /*stencilAuxValues*/,
                                                                      1,
                                                                      stencil,
                                                                      stencilWeights,
                                                                      sOff,
                                                                      sourceResults.data(),
                                                                      boundary.get());

    // assert
    for (std::size_t i = 0; i < GetParam().expectedResults.size(); i++) {
        ASSERT_TRUE(PetscAbs(GetParam().expectedResults[i] - sourceResults[i]) / (GetParam().expectedResults[i] + 1E-30) < 1E-6)
            << "The actual source term (" << sourceResults[i] << ") for index " << i << " should match expected " << GetParam().expectedResults[i];
    }
}

INSTANTIATE_TEST_SUITE_P(
    PhysicsBoundaryTests, SublimationTestFixture,
    testing::Values(
        (SublimationTestParameters){.description = "1D left boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {-.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, -62.5, 0.000000162760417}},
        (SublimationTestParameters){.description = "1D left boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {-.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 150,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "1D left boundary with heating and additional heat transfer",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .additionalHeatTransfer = ablate::mathFunctions::Create(25.0 * 2.5),
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {-.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 325,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, -62.5, 0.000000162760417}},
        (SublimationTestParameters){.description = "1D right boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {1.0, NAN, NAN}, .areas = {.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, -62.5, -0.000000162760417}},
        (SublimationTestParameters){.description = "1D right boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {1.0, NAN, NAN}, .areas = {.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "3D bottom boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {0.0, 0.0, -0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, -62.5, 0.0, 0.0, 0.000000162760417}},
        (SublimationTestParameters){.description = "3D bottom boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {0.0, 0.0, -0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "3D top boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, 1.0}, .areas = {0.0, 0.0, 0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, -62.5, 0.0, 0.0, -0.000000162760417}},
        (SublimationTestParameters){.description = "3D top boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, 1.0}, .areas = {0.0, 0.0, 0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D lower left corner boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {-0.70710678118655, -0.70710678118655}, .areas = {-0.3535533906, -0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 335.3553390593,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0003125, -62.5, 1.1508899433575995E-7, 1.1508899433575995E-7}},
        (SublimationTestParameters){.description = "2D lower left corner boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal =
                                                       {
                                                           -0.70710678118655,
                                                           -0.70710678118655,
                                                       },
                                                   .areas = {-0.3535533906, -0.3535533906},
                                                   .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {0.70710678118655, 0.70710678118655}, .areas = {0.3535533906, 0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 264.6446609407,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0003125, -62.5, -1.1508899433575995E-7, -1.1508899433575995E-7}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal =
                                                       {
                                                           0.70710678118655,
                                                           0.70710678118655,
                                                       },
                                                   .areas = {0.3535533906, 0.3535533906},
                                                   .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with heating and species",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .numberSpecies = 3,
                                    .speciesMassFractions = std::make_shared<ablate::mathFunctions::FieldFunction>("massFractions", ablate::mathFunctions::Create(std::vector<double>{.5, .3, .2})),
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {0.70710678118655, 0.70710678118655}, .areas = {0.3535533906, 0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 264.6446609407,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0003125, -62.5, -1.1508899433575995E-7, -1.1508899433575995E-7, 0.0003125 * .5, 0.0003125 * .3, 0.0003125 * .2}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with cooling and species",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .numberSpecies = 3,
                                    .speciesMassFractions = std::make_shared<ablate::mathFunctions::FieldFunction>("massFractions", ablate::mathFunctions::Create(std::vector<double>{.5, .3, .2})),
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal =
                                                       {
                                                           0.70710678118655,
                                                           0.70710678118655,
                                                       },
                                                   .areas = {0.3535533906, 0.3535533906},
                                                   .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D lower left no gradient",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .numberSpecies = 0,
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {0.70710678118655, -0.70710678118655}, .areas = {-0.3535533906, -0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 300,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0}}),
    [](const testing::TestParamInfo<SublimationTestParameters>& info) { return testingResources::PetscTestFixture::SanitizeTestName(info.param.description); });