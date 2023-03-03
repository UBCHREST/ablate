#include <petsc.h>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include "MpiTestFixture.hpp"
#include "builder.hpp"
#include "convergenceTester.hpp"
#include "domain/boxMesh.hpp"
#include "domain/boxMeshBoundaryCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "environment/runEnvironment.hpp"
#include "eos/perfectGas.hpp"
#include "eos/radiationProperties/constant.hpp"
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "monitors/timeStepMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "radiation/radiation.hpp"
#include "radiation/volumeRadiation.hpp"
#include "utilities/petscUtilities.hpp"

struct RadiationTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::function<std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>()> initialization;
    std::shared_ptr<ablate::mathFunctions::MathFunction> expectedResult;
    std::function<std::shared_ptr<ablate::radiation::Radiation>(std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn)> radiationFactory;
};

class RadiationTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RadiationTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscReal CSimp(PetscReal a, PetscReal b, std::vector<double>& f) {
    /** b-a represents the size of the total domain that is being integrated over
     * The number of elements in the vector that is being integrated over
     * Initialize the sum of all middle elements
     * Weight lightly on the borders
     * Weight heavily in the center
     * Add this value to the total every time
     * Compute the total final integral
     * */
    PetscReal I;
    int n = (int)f.size();  //!< The number of elements in the vector that is being integrated over
    int margin = 0;
    PetscReal f_sum = 0;  //!< Initialize the sum of all middle elements

    if (a != b) {
        /** Loop through every point except the first and last*/
        for (int i = margin; i < (n - margin); i++) {
            if (i % 2 == 0) {
                f[i] = 2 * f[i];  //!< Weight lightly on the borders
            } else {
                f[i] = 4 * f[i];  //!< Weight heavily in the center
            }
            f_sum += f[i];  //!< Add this value to the total every time
        }
        I = ((b - a) / (3 * n)) * (f[0] + f_sum + f[n - 1]);  //!< Compute the total final integral
    } else {
        I = 0;
    }
    return I;
}

static PetscReal EInteg(int order, double x) {
    if (x == 0 && order != 1) return 1.0 / (order - 1.0);  // Simple solution in this case, exit
    std::vector<PetscReal> En;
    int N = 100;
    for (int n = 1; n < N; n++) {
        double mu = (double)n / N;
        if (order == 1) {
            En.push_back(exp(-x / mu) / mu);
        }
        if (order == 2) {
            En.push_back(exp(-x / mu));
        }
    }
    PetscReal final = CSimp(0, 1, En);
    return final;
}

static PetscReal ReallySolveParallelPlates(PetscReal z) {
    /** Analytical solution of a special verification case.
     * Define variables and basic information
     * Intensity of rays originating from the top plate
     * Set the initial ray intensity to the bottom wall intensity
     * Intensity of rays originating from the bottom plate
     * Kappa is not spatially dependant in this special case
     * Prescribe the top and bottom heights for the domain
     * */
    PetscReal G;
    PetscReal IT = ablate::radiation::Radiation::GetBlackBodyTotalIntensity(700, 1);  // Intensity of rays originating from the top plate
    PetscReal IB =
        ablate::radiation::Radiation::GetBlackBodyTotalIntensity(1300, 1);  // Set the initial ray intensity to the bottom wall intensity //Intensity of rays originating from the bottom plate
    PetscReal kappa = 1;                                                    // Kappa is not spatially dependant in this special case
    PetscReal zBottom = -0.0105;                                            // Prescribe the top and bottom heights for the domain
    PetscReal zTop = 0.0105;

    PetscReal temperature;
    PetscReal Ibz;

    PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    const PetscReal sbc = 5.6696e-8;
    PetscInt nZp = 1000;

    std::vector<PetscReal> Iplus;
    std::vector<PetscReal> Iminus;

    for (PetscInt nzp = 1; nzp < (nZp - 1); nzp++) {
        /** Plus integral goes from bottom to Z
         * Calculate the z height
         * Get the temperature
         * Two parabolas, is the z coordinate in one half of the domain or the other
         * */
        PetscReal zp = zBottom + ((PetscReal)nzp / nZp) * (z - zBottom);  // Calculate the z height
        if (zp <= 0) {                                                    // Two parabolas, is the z coordinate in one half of the domain or the other
            temperature = -6.349E6 * zp * zp + 2000.0;
        } else {
            temperature = -1.179E7 * zp * zp + 2000.0;
        }
        /** Get the black body intensity here*/
        Ibz = ablate::radiation::Radiation::GetBlackBodyTotalIntensity(temperature, 1);
        Iplus.push_back(Ibz * EInteg(1, kappa * (z - zp)));
    }
    for (PetscInt nzp = 1; nzp < (nZp - 1); nzp++) {             /** Minus integral goes from z to top*/
        PetscReal zp = z + ((PetscReal)nzp / nZp) * (zTop - z);  // Calculate the zp height
        /** Get the temperature*/
        if (zp <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other
            temperature = -6.349E6 * zp * zp + 2000.0;
        } else {
            temperature = -1.179E7 * zp * zp + 2000.0;
        }
        /** Get the black body intensity here*/
        Ibz = ablate::radiation::Radiation::GetBlackBodyTotalIntensity(temperature, 1);
        Iminus.push_back(Ibz * EInteg(1, kappa * (zp - z)));
    }

    PetscReal term1 = IB * EInteg(2, kappa * (z - zBottom));
    PetscReal term2 = IT * EInteg(2, kappa * (zTop - z));
    PetscReal term3 = CSimp(zBottom, z, Iplus);
    PetscReal term4 = CSimp(z, zTop, Iminus);

    G = 2 * pi * (term1 + term2 + term3 + term4);

    /**Now compute the losses at the given input point (this is in order to match the output that is given by the ComputeRHSFunction)*/
    if (z <= 0) {  // Two parabolas, is the z coordinate in one half of the domain or the other
        temperature = -6.349E6 * z * z + 2000.0;
    } else {
        temperature = -1.179E7 * z * z + 2000.0;
    }
    PetscReal losses = 4 * sbc * temperature * temperature * temperature * temperature;
    PetscReal radTotal = -kappa * (losses - G);

    return radTotal;
}

TEST_P(RadiationTestFixture, ShouldComputeCorrectSourceTerm) {
    StartWithMPI

        // initialize petsc and mpi
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize();

        // keep track of history
        testingResources::ConvergenceTester l2History("l2");

        auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}));

        // determine required fields for radiation, this will include euler and temperature
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

        auto domain = std::make_shared<ablate::domain::BoxMeshBoundaryCells>("simpleMesh",
                                                                             fieldDescriptors,
                                                                             std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                                                             std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                                                             GetParam().meshFaces,
                                                                             GetParam().meshStart,
                                                                             GetParam().meshEnd,
                                                                             false,
                                                                             ablate::parameters::MapParameters::Create({{"dm_plex_hash_location", "true"}}));

        // Setup the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".4"}});

        // Set the initial conditions for euler (not used, so set all to zero)
        auto initialConditionEuler = std::make_shared<ablate::mathFunctions::FieldFunction>("euler", std::make_shared<ablate::mathFunctions::ConstantValue>(0.0));

            // create a time stepper
            auto timeStepper = ablate::solver::TimeStepper(
                "timeStepper", domain, ablate::parameters::MapParameters::Create({{"ts_max_steps", 0}}), {}, std::make_shared<ablate::domain::Initializer>(initialConditionEuler));

        // Create an instance of radiation
        auto radiationPropertiesModel = std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 1.0, 1.0);
        auto radiationModel = GetParam().radiationFactory(radiationPropertiesModel);
        auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
        auto radiation = std::make_shared<ablate::radiation::VolumeRadiation>("radiation", interiorLabel, nullptr, radiationModel, nullptr, nullptr);

        // register the flowSolver with the timeStepper
        timeStepper.Register(radiation, {std::make_shared<ablate::monitors::TimeStepMonitor>()});
        timeStepper.Solve();

        // force the aux variables of temperature to a known value
        auto auxVec = radiation->GetSubDomain().GetAuxVector();
        radiation->GetSubDomain().ProjectFieldFunctionsToLocalVector(GetParam().initialization(), auxVec);

        // Setup the rhs for the test
        Vec rhs;
        DMGetLocalVector(domain->GetDM(), &rhs) >> testErrorChecker;
        VecZeroEntries(rhs) >> testErrorChecker;

        // Apply the rhs function for the radiation solver
        radiation->PreRHSFunction(timeStepper.GetTS(), 0.0, true, nullptr) >> testErrorChecker;
        radiation->ComputeRHSFunction(0, rhs, rhs);  // The ray tracing function needs to be renamed in order to occupy the role of compute right hand side function

        // determine the euler field
        const auto& eulerFieldInfo = domain->GetField("euler");

        // For each cell, compare the rhs against the expected
        {
            // get the cell geometry
            Vec cellGeomVec;
            DM dmCell;
            const PetscScalar* cellGeomArray;
            DMPlexGetGeometryFVM(domain->GetDM(), nullptr, &cellGeomVec, nullptr) >> testErrorChecker;
            VecGetDM(cellGeomVec, &dmCell) >> testErrorChecker;
            VecGetArrayRead(cellGeomVec, &cellGeomArray) >> testErrorChecker;

            // extract the rhsArray
            const PetscScalar* rhsArray;
            VecGetArrayRead(rhs, &rhsArray) >> testErrorChecker;

            /// Declare L2 norm variables
            PetscReal l2sum = 0.0;
            double error;  // Number of cells in the domain

            ablate::domain::Range cellRange;
            radiation->GetCellRange(cellRange);
            // March over each cell
            for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
                const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

                if (ablate::domain::Region::InRegion(interiorLabel, dmCell, cell)) {
                    // Get the cell center
                    PetscFVCellGeom* cellGeom;
                    DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cellGeom) >> testErrorChecker;

                    // extract the result from the rhs
                    PetscScalar* rhsValues;
                    DMPlexPointLocalFieldRead(domain->GetDM(), cell, eulerFieldInfo.id, rhsArray, &rhsValues) >> testErrorChecker;
                    PetscScalar actualResult = rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE];
                    PetscScalar analyticalResult = ReallySolveParallelPlates(cellGeom->centroid[1]);  // Compute the analytical solution at this z height.

                    /// Summing of the L2 norm values
                    error = (analyticalResult - actualResult);
                    l2sum += error * error;
                }
            }
            /// Compute the L2 Norm error
            double N = (cellRange.end - cellRange.start);
            double l2 = sqrt(l2sum) / N;

            PetscPrintf(MPI_COMM_WORLD, "L2 Norm: %f\n", sqrt(l2sum) / N);
            if (l2 > 45000) {
                FAIL() << "Radiation test error exceeded.";
            }

            VecRestoreArrayRead(rhs, &rhsArray) >> testErrorChecker;
            VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> testErrorChecker;
        }

        DMRestoreLocalVector(domain->GetDM(), &rhs);
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(RadiationTests, RadiationTestFixture,
                         testing::Values((RadiationTestParameters){.mpiTestParameter = testingResources::MpiTestParameter("1D uniform temperature 1"),
                                                                   .meshFaces = {3, 20},
                                                                   .meshStart = {-0.5, -0.0105},
                                                                   .meshEnd = {0.5, 0.0105},
                                                                   .initialization =
                                                                       []() {
                                                                           return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(
                                                                                   ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                   ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                                      ablate::mathFunctions::Create("1300"),
                                                                                                                                      nullptr,
                                                                                                                                      std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                                      ablate::mathFunctions::Create("700"),
                                                                                                                                      nullptr,
                                                                                                                                      std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                                                       },
                                                                   .expectedResult = ablate::mathFunctions::Create("x + y"),
                                                                   .radiationFactory =
                                                                       [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                                           auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                                           return std::make_shared<ablate::radiation::Radiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                                       }},
                                         (RadiationTestParameters){.mpiTestParameter = testingResources::MpiTestParameter("1D uniform temperature 1.1"),
                                                                   .meshFaces = {3, 20},
                                                                   .meshStart = {-0.5, -0.0105},
                                                                   .meshEnd = {0.5, 0.0105},
                                                                   .initialization =
                                                                       []() {
                                                                           return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(
                                                                                   ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                   ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                                      ablate::mathFunctions::Create("1300"),
                                                                                                                                      nullptr,
                                                                                                                                      std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                                      ablate::mathFunctions::Create("700"),
                                                                                                                                      nullptr,
                                                                                                                                      std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                                                       },
                                                                   .expectedResult = ablate::mathFunctions::Create("x + y"),
                                                                   .radiationFactory =
                                                                       [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                                           auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                                           return std::make_shared<ablate::radiation::Radiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                                       }},
                                         (RadiationTestParameters){.mpiTestParameter = testingResources::MpiTestParameter("1D uniform temperature 2 proc.", 2),
                                                                   .meshFaces = {3, 20},
                                                                   .meshStart = {-0.5, -0.0105},
                                                                   .meshEnd = {0.5, 0.0105},
                                                                   .initialization =
                                                                       []() {
                                                                           return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(
                                                                                   ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                   ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                                      ablate::mathFunctions::Create("1300"),
                                                                                                                                      nullptr,
                                                                                                                                      std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                                                               std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                                      ablate::mathFunctions::Create("700"),
                                                                                                                                      nullptr,
                                                                                                                                      std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                                                       },
                                                                   .expectedResult = ablate::mathFunctions::Create("x + y"),
                                                                   .radiationFactory =
                                                                       [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                                           auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                                           return std::make_shared<ablate::radiation::Radiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                                       }}),
                         [](const testing::TestParamInfo<RadiationTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });
