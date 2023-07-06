#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/subModels/solidHeatTransfer.hpp"
#include "convergenceTester.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"

struct SolidHeatTransferTestParameters {
    // Creation options
    const std::shared_ptr<ablate::parameters::MapParameters> properties;
    const std::shared_ptr<ablate::parameters::MapParameters> options;

    // exact solution also used for init
    std::function<std::shared_ptr<ablate::mathFunctions::MathFunction>()> exactSolutionFactory;

    // ts options
    PetscReal timeEnd;
};

class SolidHeatTransferTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<SolidHeatTransferTestParameters> {};

TEST_P(SolidHeatTransferTestFixture, ShouldConverge) {
    // get the required variables
    const auto& params = GetParam();

    // Set the inital number of faces
    PetscInt initialNx = 20;

    testingResources::ConvergenceTester l2History("l2");

    // Get the exact solution
    auto exactSolution = params.exactSolutionFactory();

    // March over each level
    for (PetscInt l = 0; l < 2; l++) {
        // Create a mesh
        PetscInt nx1D = initialNx * PetscPowInt(2, l);
        PetscPrintf(PETSC_COMM_WORLD, "Running Calculation at Level %" PetscInt_FMT " (%" PetscInt_FMT ")\n", l, nx1D);

        // Set the nx in the solver options
        params.options->Insert("dm_plex_box_faces", nx1D);

        // Create the 1D solver
        auto solidHeatTransfer = std::make_shared<ablate::boundarySolver::subModels::SolidHeatTransfer>(params.properties, exactSolution, params.options);

        // Advance, pass in a surface heat flux and update the internal properties
        ablate::boundarySolver::subModels::SolidHeatTransfer::SurfaceState result{};
        solidHeatTransfer->Solve(0.0, params.timeEnd, result) >> ablate::utilities::PetscUtilities::checkError;

        // extract the required information from the dm
        DM dm;
        TSGetDM(solidHeatTransfer->GetTS(), &dm) >> ablate::utilities::PetscUtilities::checkError;
        PetscReal time;
        TSGetTime(solidHeatTransfer->GetTS(), &time) >> ablate::utilities::PetscUtilities::checkError;
        Vec solution;
        TSGetSolution(solidHeatTransfer->GetTS(), &solution) >> ablate::utilities::PetscUtilities::checkError;

        // Compute the error
        ablate::mathFunctions::PetscFunction petscExactFunction[1] = {exactSolution->GetPetscFunction()};
        void* petscExactContext[1] = {exactSolution->GetContext()};
        std::vector<PetscReal> fErrors = {.1};
        DMComputeL2FieldDiff(dm, time, petscExactFunction, petscExactContext, solution, fErrors.data());

        // record the error
        auto domainLength = params.options->GetExpect<double>("dm_plex_box_upper");
        l2History.Record(domainLength / nx1D, fErrors);
    }
    // ASSERt
    std::string l2Message;
    if (!l2History.CompareConvergenceRate({2.0}, l2Message)) {
        FAIL() << l2Message;
    }
}

// helper function to create a result function
static std::shared_ptr<ablate::mathFunctions::MathFunction> CreateHeatEquationDirichletExactSolution(PetscReal length, PetscReal specificHeat, PetscReal conductivity, PetscReal density,
                                                                                                     PetscReal temperatureInit, PetscReal temperatureBoundary, PetscReal timeOffset = 0.0) {
    auto function = [conductivity, density, specificHeat, temperatureInit, temperatureBoundary, length, timeOffset](int dim, double time, const double x[], int nf, double* u, void* ctx) {
        // compute the alpha in the equation
        time += timeOffset;
        PetscReal alpha = conductivity / (density * specificHeat);
        PetscReal effectiveTemperatureInit = (temperatureInit - temperatureBoundary);
        PetscReal T = 0.0;
        for (PetscInt n = 1; n < 2000; ++n) {
            PetscReal Bn = -effectiveTemperatureInit * 2.0 * (-1.0 + PetscPowReal(-1.0, n)) / (n * PETSC_PI);
            T += Bn * PetscSinReal(n * PETSC_PI * x[0] / length) * PetscExpReal(-n * n * PETSC_PI * PETSC_PI * alpha * time / (PetscSqr(length)));
        }

        u[0] = T + temperatureBoundary + time;
        return PETSC_SUCCESS;
    };

    return ablate::mathFunctions::Create(function);
}

INSTANTIATE_TEST_SUITE_P(
    SolidHeatTransfer, SolidHeatTransferTestFixture,
    testing::Values(
        // no boundary temperature
        (SolidHeatTransferTestParameters){.properties = ablate::parameters::MapParameters::Create(
                                              {{"specificHeat", 1000.0}, {"conductivity", 1.0}, {"density", 1.0}, {"maximumSurfaceTemperature", 000.0}, {"farFieldTemperature", 000.0}}),
                                          .options = ablate::parameters::MapParameters::Create({{"ts_dt", "0.01"}, {"dm_plex_box_upper", .1}}),
                                          .exactSolutionFactory = []() { return CreateHeatEquationDirichletExactSolution(.1, 1000.0, 1.0, 1.0, 1000.0, 000.0, .5); },
                                          .timeEnd = .1},
        // fixed boundary temperature
        (SolidHeatTransferTestParameters){.properties = ablate::parameters::MapParameters::Create(
                                              {{"specificHeat", 1000.0}, {"conductivity", .25}, {"density", 0.7}, {"maximumSurfaceTemperature", 400.0}, {"farFieldTemperature", 400.0}}),
                                          .options = ablate::parameters::MapParameters::Create({{"ts_dt", "0.01"}, {"dm_plex_box_upper", .25}}),
                                          .exactSolutionFactory = []() { return CreateHeatEquationDirichletExactSolution(.25, 1000.0, 0.25, 0.7, 1500.0, 400.0, .5); },
                                          .timeEnd = .1}

        ),
    [](const testing::TestParamInfo<SolidHeatTransferTestParameters>& info) { return std::to_string(info.index); });
