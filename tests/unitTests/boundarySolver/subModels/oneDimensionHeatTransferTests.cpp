#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/physics/subModels/oneDimensionHeatTransfer.hpp"
#include "convergenceTester.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "monitors/solutionErrorMonitor.hpp"
#include "parameters/mapParameters.hpp"

struct OneDimensionHeatTransferTestParameters {
    // Creation options
    const std::shared_ptr<ablate::parameters::MapParameters> properties;
    const std::shared_ptr<ablate::parameters::MapParameters> options;
    std::optional<double> maximumSurfaceTemperature;

    // exact solution also used for init
    std::function<std::shared_ptr<ablate::mathFunctions::MathFunction>()> exactSolutionFactory;

    // ts options
    PetscReal timeEnd;

    // comparisons
    PetscReal expectedConvergenceRate;
};

class OneDimensionHeatTransferTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<OneDimensionHeatTransferTestParameters> {};

/**
 * This function is only useful if ts_monitor_solution_exact is set for the options.  An example would include
 * params.options->Insert("dm_view", "hdf5:/path/to/debug/sol." + std::to_string(l) + ".h5");
 * params.options->Insert("ts_monitor_solution_exact", "hdf5:/path/to/debug/sol." + std::to_string(l) + ".h5::append");
 * @param ts
 * @param step
 * @param ptime
 * @param solution
 * @param ctx
 * @return
 */
PetscErrorCode TSMonitorSolution(TS ts, PetscInt step, PetscReal ptime, Vec solution, void* ctx) {
    PetscFunctionBegin;
    // get the function
    auto exactSolution = (ablate::mathFunctions::MathFunction*)ctx;

    // Compute the error
    ablate::mathFunctions::PetscFunction petscExactFunction[1] = {exactSolution->GetPetscFunction()};
    void* petscExactContext[1] = {exactSolution->GetContext()};

    // Get the dm from the vec
    DM dm;
    PetscCall(VecGetDM(solution, &dm));
    PetscOptions options;
    PetscObjectGetOptions((PetscObject)ts, &options);

    Vec exact;
    PetscCall(VecDuplicate(solution, &exact));
    PetscCall(PetscObjectSetName((PetscObject)exact, "exact"));
    PetscCall(PetscObjectSetOptions((PetscObject)exact, options));
    PetscCall(PetscObjectSetOptions((PetscObject)solution, options));

    PetscCall(DMProjectFunction(dm, ptime, petscExactFunction, petscExactContext, INSERT_VALUES, exact));
    PetscCall(DMSetOutputSequenceNumber(dm, step, ptime));
    PetscCall(VecViewFromOptions(solution, nullptr, "-ts_monitor_solution_exact"));
    PetscCall(VecViewFromOptions(exact, nullptr, "-ts_monitor_solution_exact"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

TEST_P(OneDimensionHeatTransferTestFixture, ShouldConverge) {
    // get the required variables
    const auto& params = GetParam();

    // Set the initial number of faces
    PetscInt initialNx = 20;

    testingResources::ConvergenceTester l2History("l2");

    // Get the exact solution
    auto exactSolution = params.exactSolutionFactory();

    // March over each level
    for (PetscInt l = 0; l < 3; l++) {
        // Create a mesh
        PetscInt nx1D = initialNx * PetscPowInt(2, l);
        PetscPrintf(PETSC_COMM_WORLD, "Running Calculation at Level %" PetscInt_FMT " (%" PetscInt_FMT ")\n", l, nx1D);

        // Set the nx in the solver options
        params.options->Insert("dm_plex_box_faces", nx1D);

        // Create the 1D solver
        auto solidHeatTransfer = std::make_shared<ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer>(
            "test", params.properties, exactSolution, params.options, params.maximumSurfaceTemperature.value_or(PETSC_DEFAULT));

        auto ts = solidHeatTransfer->GetTS();
        TSMonitorSet(ts, TSMonitorSolution, exactSolution.get(), nullptr);

        // Advance, pass in a surface heat flux and update the internal properties
        PetscReal surfaceTemperature;
        PetscReal heatFlux;
        solidHeatTransfer->Solve(0.0, params.timeEnd, surfaceTemperature, heatFlux) >> ablate::utilities::PetscUtilities::checkError;

        // extract the required information from the dm
        DM dm;
        TSGetDM(solidHeatTransfer->GetTS(), &dm) >> ablate::utilities::PetscUtilities::checkError;
        PetscReal time;
        TSGetTime(solidHeatTransfer->GetTS(), &time) >> ablate::utilities::PetscUtilities::checkError;
        Vec solution;
        TSGetSolution(solidHeatTransfer->GetTS(), &solution) >> ablate::utilities::PetscUtilities::checkError;

        // Set the exact solution
        PetscDS ds;
        DMGetDS(dm, &ds) >> ablate::utilities::PetscUtilities::checkError;
        PetscDSSetExactSolution(ds, 0, exactSolution->GetPetscFunction(), exactSolution->GetContext());

        // Get the L2 and LInf norms
        std::vector<PetscReal> l2Norm =
            ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::utilities::MathUtilities::Norm::L2_NORM).ComputeError(ts, time, solution);

        // Compute the error
        ablate::mathFunctions::PetscFunction petscExactFunction[1] = {exactSolution->GetPetscFunction()};
        void* petscExactContext[1] = {exactSolution->GetContext()};
        std::vector<PetscReal> fErrors = {0.0};
        DMComputeL2FieldDiff(dm, time, petscExactFunction, petscExactContext, solution, fErrors.data());

        // record the error
        auto domainLength = params.options->GetExpect<double>("dm_plex_box_upper");
        l2History.Record(domainLength / nx1D, fErrors);
    }
    // ASSERt
    std::string l2Message;
    if (!l2History.CompareConvergenceRate({GetParam().expectedConvergenceRate}, l2Message, false)) {
        FAIL() << l2Message;
    }

    PetscObjectsDump(nullptr, PETSC_TRUE);
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

        u[0] = PetscMax(temperatureBoundary, T + temperatureBoundary);
        return PETSC_SUCCESS;
    };

    return ablate::mathFunctions::Create(function);
}

INSTANTIATE_TEST_SUITE_P(SolidHeatTransfer, OneDimensionHeatTransferTestFixture,
                         testing::Values(
                             // no boundary temperature
                             (OneDimensionHeatTransferTestParameters){.properties = ablate::parameters::MapParameters::Create({{"specificHeat", 1000.0}, {"conductivity", 1.0}, {"density", 1.0}}),
                                                                      .options = ablate::parameters::MapParameters::Create({{"ts_dt", "1E-4"}, {"dm_plex_box_upper", .1}, {"ts_adapt_type", "basic"}}),
                                                                      .maximumSurfaceTemperature = 0.0,
                                                                      .exactSolutionFactory = []() { return CreateHeatEquationDirichletExactSolution(.1, 1000.0, 1.0, 1.0, 1000.0, 000.0, 1E-5); },
                                                                      .timeEnd = .01,
                                                                      .expectedConvergenceRate = 2.0},
                             // fixed boundary temperature
                             (OneDimensionHeatTransferTestParameters){.properties = ablate::parameters::MapParameters::Create({{"specificHeat", 1000.0}, {"conductivity", .25}, {"density", 0.7}}),
                                                                      .options = ablate::parameters::MapParameters::Create({{"ts_dt", "0.001"}, {"dm_plex_box_upper", .25}, {"ts_adapt_type", "none"}}),
                                                                      .maximumSurfaceTemperature = 400.0,
                                                                      .exactSolutionFactory = []() { return CreateHeatEquationDirichletExactSolution(.25, 1000.0, 0.25, 0.7, 1500.0, 400.0, .01); },
                                                                      .timeEnd = .5,
                                                                      .expectedConvergenceRate = 2.0}

                             ),
                         [](const testing::TestParamInfo<OneDimensionHeatTransferTestParameters>& info) { return std::to_string(info.index); });
