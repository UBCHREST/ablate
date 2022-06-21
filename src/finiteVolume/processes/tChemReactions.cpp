#include "tChemReactions.hpp"
#include <TChem_EnthalpyMass.hpp>
#include <TChem_IgnitionZeroD.hpp>
#include <utilities/petscError.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::finiteVolume::processes::TChemReactions::TChemReactions(std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<parameters::Parameters> options, std::vector<std::string> inertSpecies,
                                                                std::vector<double> massFractionBounds)
    : eos(std::dynamic_pointer_cast<eos::TChem>(eosIn)), numberSpecies(eosIn->GetSpecies().size()) {
    // make sure that the eos is set
    if (!std::dynamic_pointer_cast<eos::TChem>(eosIn)) {
        throw std::invalid_argument("ablate::finiteVolume::processes::TChemReactions only accepts EOS of type eos::TChem");
    }

    // Make sure the massFractionBounds are valid
    if (!(massFractionBounds.empty() || massFractionBounds.size() == 2)) {
        throw std::invalid_argument("If specified, the massFractionBounds must be of length 2.");
    }
}
ablate::finiteVolume::processes::TChemReactions::~TChemReactions() {}

void ablate::finiteVolume::processes::TChemReactions::Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // determine the number of nodes we need to compute based upon the local solver
    solver::Range cellRange;
    flow.GetCellRangeWithoutGhost(cellRange);

    // determine the number of required cells
    std::size_t numberCells = cellRange.end - cellRange.start;
    flow.RestoreRange(cellRange);

    // compute the required state dimension
    auto kineticModelGasConstData = TChem::createGasKineticModelConstData<typename Tines::UseThisDevice<host_exec_space>::type>(eos->GetKineticModelData());
    const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(kineticModelGasConstData.nSpec);

    // allocate the tChem memory
    stateHost = real_type_2d_view_host("stateVectorDevices", numberCells, stateVecDim);
    stateDevice = Kokkos::create_mirror(stateHost);
    endStateHost = real_type_2d_view_host("stateVectorDevicesEnd", numberCells, stateVecDim);
    endStateDevice = Kokkos::create_mirror(endStateHost);
    internalEnergyRefHost = real_type_1d_view_host("internalEnergyRefHost", numberCells);
    internalEnergyRefDevice = Kokkos::create_mirror(internalEnergyRefHost);
    sourceTermsHost = real_type_2d_view_host("sourceTermsHost", numberCells, kineticModelGasConstData.nSpec);
    perSpeciesScratchDevice = real_type_2d_view("perSpeciesScratchDevice", numberCells, kineticModelGasConstData.nSpec);

    // Create the default timeAdvanceObject
    timeAdvanceDefault._tbeg = 0.0;
    timeAdvanceDefault._tend = 1.0;
    timeAdvanceDefault._dt = 1.0E-8;
    timeAdvanceDefault._dtmin = 1.0E-12;
    timeAdvanceDefault._dtmax = 1.0E-1;
    timeAdvanceDefault._max_num_newton_iterations = 10000;
    timeAdvanceDefault._num_time_iterations_per_interval = 100000;
    timeAdvanceDefault._num_outer_time_iterations_per_interval = 1;
    timeAdvanceDefault._jacobian_interval = 1;

    // Copy the default values to device
    timeAdvanceDevice = time_advance_type_1d_view("timeAdvanceDevice", numberCells);

    // determine the number of equations
    auto numberOfEquations = TChem::Impl::IgnitionZeroD_Problem<real_type, Tines::UseThisDevice<exec_space>::type>::getNumberOfTimeODEs(kineticModelGasConstData);

    // size up the tolerance constraints
    tolTimeDevice = real_type_2d_view("tolTimeDevice", numberOfEquations, 2);
    tolNewtonDevice = real_type_1d_view("tolNewtonDevice", 2);
    facDevice = real_type_2d_view("facDevice", numberCells, numberOfEquations);

    /// tune tolerance
    {
        auto tolTimeHost = Kokkos::create_mirror_view(tolTimeDevice);
        auto tolNewtonHost = Kokkos::create_mirror_view(tolNewtonDevice);

        for (ordinal_type i = 0, iend = tolTimeDevice.extent(0); i < iend; ++i) {
            tolTimeHost(i, 0) = 1e-12;   // atol
            tolTimeHost(i, 1) = 1.0E-4;  // rtol
        }
        tolNewtonHost(0) = 1e-10;  // atol
        tolNewtonHost(1) = 1e-6;   // rtol

        Kokkos::deep_copy(tolTimeDevice, tolTimeHost);
        Kokkos::deep_copy(tolNewtonDevice, tolNewtonHost);
    }

    // get device kineticModelGasConstData
    kineticModelGasConstDataDevice = TChem::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(eos->GetKineticModelData());
    kineticModelDataClone = eos->GetKineticModelData().clone(numberCells);
    kineticModelGasConstDataDevices = TChem::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(kineticModelDataClone);

    // Before each step, compute the source term over the entire dt
    auto chemistryPreStage = std::bind(&ablate::finiteVolume::processes::TChemReactions::ChemistryFlowPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    flow.RegisterPreStage(chemistryPreStage);

    // Add the rhs point function for the source
    flow.RegisterRHSFunction(AddChemistrySourceToFlow, this);
}

PetscErrorCode ablate::finiteVolume::processes::TChemReactions::ChemistryFlowPreStage(TS flowTs, ablate::solver::Solver& solver, PetscReal stagetime) {
    PetscFunctionBegin;

    // get time step information from the ts
    PetscInt stepNumber;
    PetscCall(TSGetStepNumber(flowTs, &stepNumber));
    PetscReal time;
    PetscCall(TSGetTime(flowTs, &time));

    // only continue if the stage time is the real time (i.e. the first stage)
    if (time != stagetime) {
        PetscFunctionReturn(0);
    }

    // Get the valid cell range over this region
    auto& fvSolver = static_cast<ablate::finiteVolume::FiniteVolumeSolver&>(solver);
    solver::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    auto numberCells = cellRange.end - cellRange.start;

    // get the dim
    PetscInt dim;
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));

    // store the current dt
    PetscReal dt;
    PetscCall(TSGetTimeStep(flowTs, &dt));

    // get access to the underlying data for the flow
    const auto& flowEulerId = fvSolver.GetSubDomain().GetField("euler").id;
    const auto& flowDensityId = fvSolver.GetSubDomain().GetField("densityYi").id;

    // get the flowSolution from the ts
    DM dm = fvSolver.GetSubDomain().GetDM();
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));
    const PetscScalar* flowArray;
    PetscCall(VecGetArrayRead(globFlowVec, &flowArray));

    // Use a parallel for loop to load up the tChem state
    Kokkos::parallel_for(
        "stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), KOKKOS_LAMBDA(const auto i) {
            // get the host data from the petsc field
            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
            const std::size_t chemIndex = i - cellRange.start;

            // Get the current state variables for this cell
            const PetscScalar* eulerField = nullptr;
            DMPlexPointLocalFieldRead(dm, cell, flowEulerId, flowArray, &eulerField) >> checkError;
            const PetscScalar* flowDensityField = nullptr;
            DMPlexPointLocalFieldRead(dm, cell, flowDensityId, flowArray, &flowDensityField) >> checkError;

            // cast the state at i to a state vector
            const auto state_at_i = Kokkos::subview(stateHost, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, state_at_i);

            // get the current state at I
            auto density = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHO];
            stateVector.Density() = density;
            stateVector.Temperature() = 300.0;
            auto ys = stateVector.MassFractions();
            real_type yiSum = 0.0;
            for (ordinal_type s = 0; s < stateVector.NumSpecies(); s++) {
                ys[s] = PetscMax(0.0, flowDensityField[s] / density);
                ys[s] = PetscMin(1.0, ys[s]);
                yiSum += ys[s];
            }
            if (yiSum > 1.0) {
                for (PetscInt s = 0; s < stateVector.NumSpecies() - 1; s++) {
                    // Limit the bounds
                    ys[s] /= yiSum;
                }
                ys[stateVector.NumSpecies()] = 0.0;
            } else {
                ys[stateVector.NumSpecies()] = 1.0 - yiSum;
            }

            // Compute the internal energy from total ener
            PetscReal speedSquare = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                speedSquare += PetscSqr(eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
            }

            // compute the internal energy needed to compute temperature
            internalEnergyRefHost[chemIndex] = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
        });

    // copy from host to device
    Kokkos::deep_copy(internalEnergyRefDevice, internalEnergyRefHost);
    Kokkos::deep_copy(stateDevice, stateHost);

    // update the default with the current dt and end time
    timeAdvanceDefault._tbeg = time;
    timeAdvanceDefault._tend = time + dt;
    Kokkos::deep_copy(timeAdvanceDevice, timeAdvanceDefault);

    // setup the enthalpy, temperature, pressure, chemistry function policies
    auto temperatureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(numberCells, Kokkos::AUTO());
    temperatureFunctionPolicy.set_scratch_size(1,
                                               Kokkos::PerTeam(TChem::Scratch<real_type_1d_view>::shmem_size(ablate::eos::tChem::Temperature::getWorkSpaceSize(kineticModelGasConstDataDevice.nSpec))));
    auto pressureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(numberCells, Kokkos::AUTO());
    pressureFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam(TChem::Scratch<real_type_1d_view>::shmem_size(ablate::eos::tChem::Pressure::getWorkSpaceSize(kineticModelGasConstDataDevice.nSpec))));
    auto chemistryFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(numberCells, Kokkos::AUTO());
    chemistryFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam(TChem::Scratch<real_type_1d_view>::shmem_size(TChem::IgnitionZeroD::getWorkSpaceSize(kineticModelGasConstDataDevice))));

    // Compute temperature into the state field in the device
    ablate::eos::tChem::Temperature::runDeviceBatch(
        temperatureFunctionPolicy, stateDevice, internalEnergyRefDevice, perSpeciesScratchDevice, eos->GetEnthalpyOfFormation(), kineticModelGasConstDataDevice);

    // Compute the pressure into the state field in the device
    ablate::eos::tChem::Pressure::runDeviceBatch(pressureFunctionPolicy, stateDevice, kineticModelGasConstDataDevice);

    real_type_1d_view timeView("time", stateDevice.size());
    Kokkos::deep_copy(timeView, time);
    real_type_1d_view dtView("delta time", stateDevice.size());
    Kokkos::deep_copy(dtView, 1E-10);

    // assume a constant pressure zero D reaction for each cell
    tChemLib::IgnitionZeroD::runDeviceBatch(
        chemistryFunctionPolicy, tolNewtonDevice, tolTimeDevice, facDevice, timeAdvanceDevice, stateDevice, timeView, dtView, endStateDevice, kineticModelGasConstDataDevices);

    // copy the updated state back to host
    Kokkos::deep_copy(endStateHost, endStateDevice);

    // Use a parallel for loop to load up the tChem state
    // TODO: move to device because eos->GetEnthalpyOfFormation() is in device
    auto enthalpyOfFormation = eos->GetEnthalpyOfFormation();
    Kokkos::parallel_for(
        "sourceTermCompute", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(0, numberCells), KOKKOS_LAMBDA(const auto chemIndex) {
            // cast the state at i to a state vector
            const auto stateAtI = Kokkos::subview(stateHost, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, stateAtI);
            const auto ys = stateVector.MassFractions();

            const auto endStateAtI = Kokkos::subview(endStateHost, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> endStateVector(kineticModelGasConstDataDevice.nSpec, endStateAtI);
            const auto ye = endStateVector.MassFractions();

            const auto sourceTermAtI = Kokkos::subview(sourceTermsHost, chemIndex, Kokkos::ALL());

            // compute the source term from the change in the heat of formation
            sourceTermAtI(0) = 0.0;
            for (ordinal_type s = 0; s < stateVector.NumSpecies(); s++) {
                sourceTermAtI(0) += (ys(s) - ye(s)) * enthalpyOfFormation(s);
            }

            for (ordinal_type s = 0; s < stateVector.NumSpecies(); ++s) {
                // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                sourceTermAtI(s + 1) = ye(s) - ys(s);
            }

            // Now scale everything by density/dt
            for (std::size_t j = 0; j < sourceTermAtI.extent(0); ++j) {
                // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                sourceTermAtI(j) *= stateVector.Density() / dt;
            }
        });

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::TChemReactions::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx) {
    PetscFunctionBegin;
    auto process = (ablate::finiteVolume::processes::TChemReactions*)ctx;

    // get the cell range
    solver::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    // get access to the fArray
    PetscScalar* fArray;
    PetscCall(VecGetArray(locFVec, &fArray));

    // get access to the underlying data for the flow
    const auto& flowEulerId = solver.GetSubDomain().GetField("euler").id;
    const auto& flowDensityYiId = solver.GetSubDomain().GetField("densityYi").id;
    const auto dim = solver.GetSubDomain().GetDimensions();

    // Use a parallel for loop to load up the tChem state
    Kokkos::parallel_for(
        "stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), KOKKOS_LAMBDA(const auto i) {
            // get the host data from the petsc field
            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
            const std::size_t chemIndex = i - cellRange.start;

            // Get the current state variables for this cell
            PetscScalar* eulerSource = nullptr;
            DMPlexPointLocalFieldRef(dm, cell, flowEulerId, fArray, &eulerSource) >> checkError;
            PetscScalar* densityYiSource = nullptr;
            DMPlexPointLocalFieldRef(dm, cell, flowDensityYiId, fArray, &densityYiSource) >> checkError;

            // cast the state at i to a state vector
            const auto sourceAtI = Kokkos::subview(process->sourceTermsHost, chemIndex, Kokkos::ALL());

            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = sourceAtI[0];
            for (PetscInt d = 0; d < dim; d++) {
                eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
            }
            for (std::size_t sp = 0; sp < process->numberSpecies; sp++) {
                densityYiSource[sp] = sourceAtI(sp + 1);
            }
        });

    // cleanup
    PetscCall(VecRestoreArray(locFVec, &fArray));

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::TChemReactions, "reactions using the TChem library", ARG(ablate::eos::EOS, "eos", "the tChem v1 eos"),
         OPT(ablate::parameters::Parameters, "options", "any PETSc options for the chemistry ts"), OPT(std::vector<std::string>, "inertSpecies", "fix the Jacobian for any undetermined inertSpecies"),
         OPT(std::vector<double>, "massFractionBounds", "sets the minimum/maximum mass fraction passed to TChem Library. Must be a vector of size two [min,max] (default is no bounds)"));
