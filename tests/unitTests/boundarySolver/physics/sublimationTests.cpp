#include <functional>
#include "boundarySolver/physics/subModels/completeSublimation.hpp"
#include "boundarySolver/physics/sublimation.hpp"
#include "eos/mockEOS.hpp"
#include "eos/transport/constant.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "petscTestFixture.hpp"

struct SublimationTestParameters {
    std::string description;
    // Setup
    PetscReal latentHeatOfFusion;
    PetscReal effectiveConductivity;
    PetscReal boundaryViscosity;
    PetscReal sensibleEnthalpy;
    PetscReal boundaryPressure;
    std::shared_ptr<ablate::mathFunctions::MathFunction> additionalHeatTransfer = {};
    PetscInt numberSpecies = 0;
    std::shared_ptr<ablate::mathFunctions::FieldFunction> speciesMassFractions = {};

    // Geometry
    PetscInt dim;
    ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom fvFaceGeom;

    // Store the required aux variable (T)
    std::vector<double> boundaryValues;
    std::vector<double> stencilValues;
    PetscReal boundaryTemperature;
    PetscReal stencilTemperature;
    std::vector<PetscScalar> expectedResults;
};

class SublimationTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<SublimationTestParameters> {};

TEST_P(SublimationTestFixture, ShouldComputeCorrectSourceTerm) {
    // arrange
    // get the required variables
    const auto& params = GetParam();

    // use a mock eos for testing the sensible enthalpy call
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SensibleEnthalpy, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(
            [params](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = params.sensibleEnthalpy; })));

    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Pressure, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(
            [params](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = params.boundaryPressure; })));

    // Create a sublimation model
    auto sublimationModel = std::make_shared<ablate::boundarySolver::physics::subModels::CompleteSublimation>(params.latentHeatOfFusion);

    // create the boundary
    auto transportModel = std::make_shared<ablate::eos::transport::Constant>(params.effectiveConductivity, params.boundaryViscosity);
    auto boundary = std::make_shared<ablate::boundarySolver::physics::Sublimation>(sublimationModel, transportModel, eos, params.speciesMassFractions, params.additionalHeatTransfer);

    // initialization is not needed for testing if species are not set
    boundary->Setup(params.numberSpecies);

    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscInt sOff[2] = {0, (PetscInt)GetParam().expectedResults.size() - params.numberSpecies};
    const PetscScalar* allAuxStencilValues[1] = {&params.stencilTemperature};
    const PetscScalar* allStencilValues[1] = {params.stencilValues.data()};
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
                                                                      allStencilValues,
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
        ASSERT_TRUE(PetscAbs(GetParam().expectedResults[i] - sourceResults[i]) / PetscAbs(GetParam().expectedResults[i] + 1E-30) < 1E-6)
            << "The actual source term (" << sourceResults[i] << ") for index " << i << " should match expected " << GetParam().expectedResults[i];
    }
}

INSTANTIATE_TEST_SUITE_P(
    PhysicsBoundaryTests, SublimationTestFixture,
    testing::Values(
        (SublimationTestParameters){.description = "1D left boundary with heating zero vel grad",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 2.5,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {-.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 2.0 * 0.00052083333333333333},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50.0 * .5, 0.000000162760417}},
        (SublimationTestParameters){.description = "1D left boundary with cooling zero vel grad",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 2.5,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {-.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 150,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 150 * 2.5 * .5, 0.0}},
        (SublimationTestParameters){.description = "1D left boundary with heating and additional heat transfer",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.5,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 101325.0,
                                    .additionalHeatTransfer = ablate::mathFunctions::Create(25.0 * 2.5),
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {-.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 2.0 * (0.0005208333333 + 10)},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 325,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 25.0 * .5, 0.000000162760417 - 3.3333333333333224 + 101325 * .5}},
        (SublimationTestParameters){.description = "1D right boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.5,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 101325.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {1.0, NAN, NAN}, .areas = {.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .stencilValues = {2.0, NAN, -2.0 * (0.0005208333333 + 10)},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50.0 * .5, -0.000000162760417 - 3.3333333333333224 - 101325 * .5}},
        (SublimationTestParameters){.description = "1D right boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 1,
                                    .fvFaceGeom = {.normal = {1.0, NAN, NAN}, .areas = {.5, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 2.5 * 50.0 * .5, 0.0}},
        (SublimationTestParameters){.description = "3D bottom boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {0.0, 0.0, -0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50.0 * .5, 0.0, 0.0, 0.000000162760417}},
        (SublimationTestParameters){.description = "3D bottom boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {0.0, 0.0, -0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 2.5 * 50.0 * .5, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "3D top boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, 1.0}, .areas = {0.0, 0.0, 0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 250,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50.0 * .5, 0.0, 0.0, -0.000000162760417}},
        (SublimationTestParameters){.description = "3D top boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 3,
                                    .fvFaceGeom = {.normal = {0.0, 0.0, 1.0}, .areas = {0.0, 0.0, 0.5}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 350,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 2.5 * 50.0 * .5, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D lower left corner boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {-0.70710678118655, -0.70710678118655}, .areas = {-0.3535533906, -0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 335.3553390593,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50 * .5, 1.1508899433575995E-7, 1.1508899433575995E-7}},
        (SublimationTestParameters){.description = "2D lower left corner boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal =
                                                       {
                                                           -0.70710678118655,
                                                           -0.70710678118655,
                                                       },
                                                   .areas = {-0.3535533906, -0.3535533906},
                                                   .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 264.6446609407,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 2.5 * 50 * .5, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with heating",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {0.70710678118655, 0.70710678118655}, .areas = {0.3535533906, 0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 264.6446609407,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50 * .5, -1.1508899433575995E-7, -1.1508899433575995E-7}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with cooling",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal =
                                                       {
                                                           0.70710678118655,
                                                           0.70710678118655,
                                                       },
                                                   .areas = {0.3535533906, 0.3535533906},
                                                   .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 335.3553390593,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 2.5 * 50 * .5, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with heating and species",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 3,
                                    .speciesMassFractions = std::make_shared<ablate::mathFunctions::FieldFunction>("massFractions", ablate::mathFunctions::Create(std::vector<double>{.5, .3, .2})),
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {0.70710678118655, 0.70710678118655}, .areas = {0.3535533906, 0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 264.6446609407,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0003125, 0.0003125 * 5.0E3 - 2.5 * 50 * .5, -1.1508899433575995E-7, -1.1508899433575995E-7, 0.0003125 * .5, 0.0003125 * .3, 0.0003125 * .2}},
        (SublimationTestParameters){.description = "2D upper right corner boundary with cooling and species",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 3,
                                    .speciesMassFractions = std::make_shared<ablate::mathFunctions::FieldFunction>("massFractions", ablate::mathFunctions::Create(std::vector<double>{.5, .3, .2})),
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal =
                                                       {
                                                           0.70710678118655,
                                                           0.70710678118655,
                                                       },
                                                   .areas = {0.3535533906, 0.3535533906},
                                                   .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 335.3553390593,  // delta T = stencil-boundary ... stencil = boundary+deltaT
                                    .expectedResults = {0.0, 2.5 * 50 * .5, 0.0, 0.0, 0.0, 0.0, 0.0}},
        (SublimationTestParameters){.description = "2D lower left no gradient",
                                    // setup
                                    .latentHeatOfFusion = 2.0e+5,
                                    .effectiveConductivity = 2.5,
                                    .boundaryViscosity = 0.0,
                                    .sensibleEnthalpy = 5.0E3,
                                    .boundaryPressure = 0.0,
                                    .additionalHeatTransfer = nullptr,
                                    .numberSpecies = 0,
                                    .speciesMassFractions = {},
                                    // geometry
                                    .dim = 2,
                                    .fvFaceGeom = {.normal = {0.70710678118655, -0.70710678118655}, .areas = {-0.3535533906, -0.3535533906}, .centroid = {NAN, NAN, NAN}},
                                    // values
                                    .boundaryValues = {1.2, NAN, NAN, NAN},
                                    .stencilValues = {2.0, NAN, 0.0, 0.0},
                                    .boundaryTemperature = 300,
                                    .stencilTemperature = 300,  // note for this case a dT/dn of 50 is need on the diagonal
                                    .expectedResults = {0.0, 0.0, 0.0, 0.0}}),
    [](const testing::TestParamInfo<SublimationTestParameters>& info) { return testingResources::PetscTestFixture::SanitizeTestName(info.param.description); });