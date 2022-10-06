#include "tChemReactions.hpp"
#include <TChem_EnthalpyMass.hpp>
#include <TChem_IgnitionZeroD.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/petscError.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::processes::TChemReactions::TChemReactions(const std::shared_ptr<eos::EOS>& eosIn, const std::shared_ptr<ablate::parameters::Parameters>& options)
    : eos(std::dynamic_pointer_cast<eos::TChem>(eosIn)), numberSpecies(eosIn->GetSpecies().size()) {
    // make sure that the eos is set
    if (!std::dynamic_pointer_cast<eos::TChem>(eosIn)) {
        throw std::invalid_argument("ablate::finiteVolume::processes::TChemReactions only accepts EOS of type eos::TChem");
    }

    // Make sure the massFractionBounds are valid
    if (options) {
        dtMin = options->Get("dtMin", dtMin);
        dtMax = options->Get("dtMax", dtMax);
        dtDefault = options->Get("dtDefault", dtDefault);
        dtEstimateFactor = options->Get("dtEstimateFactor", dtEstimateFactor);
        relToleranceTime = options->Get("relToleranceTime", relToleranceTime);
        absToleranceTime = options->Get("absToleranceTime", absToleranceTime);
        relToleranceNewton = options->Get("relToleranceNewton", relToleranceNewton);
        absToleranceNewton = options->Get("absToleranceNewton", absToleranceNewton);
        maxNumNewtonIterations = options->Get("maxNumNewtonIterations", maxNumNewtonIterations);
        numTimeIterationsPerInterval = options->Get("numTimeIterationsPerInterval", numTimeIterationsPerInterval);
        jacobianInterval = options->Get("jacobianInterval", jacobianInterval);
        maxAttempts = options->Get("maxAttempts", maxAttempts);
    }
}
ablate::finiteVolume::processes::TChemReactions::~TChemReactions() = default;

void ablate::finiteVolume::processes::TChemReactions::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // Before each step, compute the source term over the entire dt
    auto chemistryPreStage = std::bind(&ablate::finiteVolume::processes::TChemReactions::ChemistryFlowPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    flow.RegisterPreStage(chemistryPreStage);

    // Add the rhs point function for the source
    flow.RegisterRHSFunction(AddChemistrySourceToFlow, this);
}

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
    endStateDevice = real_type_2d_view("stateVectorDevicesEnd", numberCells, stateVecDim);
    internalEnergyRefHost = real_type_1d_view_host("internalEnergyRefHost", numberCells);
    internalEnergyRefDevice = Kokkos::create_mirror(internalEnergyRefHost);
    sourceTermsHost = real_type_2d_view_host("sourceTermsHost", numberCells, kineticModelGasConstData.nSpec + 1);
    sourceTermsDevice = Kokkos::create_mirror(sourceTermsHost);
    perSpeciesScratchDevice = real_type_2d_view("perSpeciesScratchDevice", numberCells, kineticModelGasConstData.nSpec);
    timeViewDevice = real_type_1d_view("time", numberCells);
    dtViewDevice = real_type_1d_view("delta time", numberCells);

    // Create the default timeAdvanceObject
    timeAdvanceDefault._tbeg = 0.0;
    timeAdvanceDefault._tend = 1.0;
    timeAdvanceDefault._dt = dtDefault;
    timeAdvanceDefault._dtmin = dtMin;
    timeAdvanceDefault._dtmax = dtMax;
    timeAdvanceDefault._max_num_newton_iterations = maxNumNewtonIterations;
    timeAdvanceDefault._num_time_iterations_per_interval = numTimeIterationsPerInterval;
    timeAdvanceDefault._num_outer_time_iterations_per_interval = 10;
    timeAdvanceDefault._jacobian_interval = jacobianInterval;

    // Copy the default values to device
    timeAdvanceDevice = time_advance_type_1d_view("timeAdvanceDevice", numberCells);
    Kokkos::deep_copy(timeAdvanceDevice, timeAdvanceDefault);
    Kokkos::deep_copy(dtViewDevice, 1E-4);

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
            tolTimeHost(i, 0) = absToleranceTime;  // atol
            tolTimeHost(i, 1) = relToleranceTime;  // rtol
        }
        tolNewtonHost(0) = absToleranceNewton;  // atol
        tolNewtonHost(1) = relToleranceNewton;  // rtol

        Kokkos::deep_copy(tolTimeDevice, tolTimeHost);
        Kokkos::deep_copy(tolNewtonDevice, tolNewtonHost);
    }

    // get device kineticModelGasConstData
    kineticModelGasConstDataDevice = TChem::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(eos->GetKineticModelData());
    kineticModelDataClone = eos->GetKineticModelData().clone(numberCells);
    kineticModelGasConstDataDevices = TChem::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(kineticModelDataClone);
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
    auto& fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver&>(solver);
    solver::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    auto numberCells = cellRange.end - cellRange.start;

    // get the dim
    PetscInt dim;
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));

    // store the current dt
    PetscReal dt;
    PetscCall(TSGetTimeStep(flowTs, &dt));

    // get the rank
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(fvSolver.GetSubDomain().GetComm(), &rank));

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
            Impl::StateVector<real_type_1d_view_host> stateVector(kineticModelGasConstDataDevice.nSpec, state_at_i);

            // get the current state at I
            auto density = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHO];
            stateVector.Density() = density;
            stateVector.Temperature() = 300.0;
            auto ys = stateVector.MassFractions();
            real_type yiSum = 0.0;
            for (ordinal_type s = 0; s < stateVector.NumSpecies() - 1; s++) {
                ys[s] = PetscMax(0.0, flowDensityField[s] / density);
                ys[s] = PetscMin(1.0, ys[s]);
                yiSum += ys[s];
            }
            if (yiSum > 1.0) {
                for (PetscInt s = 0; s < stateVector.NumSpecies() - 1; s++) {
                    // Limit the bounds
                    ys[s] /= yiSum;
                }
                ys[stateVector.NumSpecies() - 1] = 0.0;
            } else {
                ys[stateVector.NumSpecies() - 1] = 1.0 - yiSum;
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

    // setup the enthalpy, temperature, pressure, chemistry function policies
    auto temperatureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(TChem::exec_space(), numberCells, Kokkos::AUTO());
    temperatureFunctionPolicy.set_scratch_size(1,
                                               Kokkos::PerTeam(TChem::Scratch<real_type_1d_view>::shmem_size(ablate::eos::tChem::Temperature::getWorkSpaceSize(kineticModelGasConstDataDevice.nSpec))));
    auto pressureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(TChem::exec_space(), numberCells, Kokkos::AUTO());
    pressureFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam(TChem::Scratch<real_type_1d_view>::shmem_size(ablate::eos::tChem::Pressure::getWorkSpaceSize(kineticModelGasConstDataDevice.nSpec))));

    // Compute temperature into the state field in the device
    ablate::eos::tChem::Temperature::runDeviceBatch(
        temperatureFunctionPolicy, stateDevice, internalEnergyRefDevice, perSpeciesScratchDevice, eos->GetEnthalpyOfFormation(), kineticModelGasConstDataDevice);

    // Compute the pressure into the state field in the device
    ablate::eos::tChem::Pressure::runDeviceBatch(pressureFunctionPolicy, stateDevice, kineticModelGasConstDataDevice);

    double minimumPressure = 0;
    for (int attempt = 0; (attempt < maxAttempts) && minimumPressure == 0; ++attempt) {
        // Use a parallel for updating timeAdvanceDevice dt
        Kokkos::parallel_for(
            "timeAdvanceUpdate", Kokkos::RangePolicy<typename tChemLib::exec_space>(0, numberCells), KOKKOS_LAMBDA(const auto i) {
                auto& tAdvAtI = timeAdvanceDevice(i);
                tAdvAtI._tbeg = time;
                tAdvAtI._tend = time + dt;
                tAdvAtI._dt = PetscMax(PetscMin(PetscMin(dtViewDevice(i) * dtEstimateFactor, dt), tAdvAtI._dtmax) / (PetscPowInt(2, attempt)), tAdvAtI._dtmin);
                // set the default time information
                timeViewDevice(i) = time;
            });

        auto chemistryFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(TChem::exec_space(), numberCells, Kokkos::AUTO());
        chemistryFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam(TChem::Scratch<real_type_1d_view>::shmem_size(TChem::IgnitionZeroD::getWorkSpaceSize(kineticModelGasConstDataDevice))));

        // assume a constant pressure zero D reaction for each cell
        tChemLib::IgnitionZeroD::runDeviceBatch(
            chemistryFunctionPolicy, tolNewtonDevice, tolTimeDevice, facDevice, timeAdvanceDevice, stateDevice, timeViewDevice, dtViewDevice, endStateDevice, kineticModelGasConstDataDevices);

        // check the output pressure, if it is zero the integration failed
        Kokkos::parallel_reduce(
            "pressureCheck",
            Kokkos::RangePolicy<typename tChemLib::exec_space>(0, numberCells),
            KOKKOS_LAMBDA(const int& chemIndex, double& pressureMin) {
                // cast the state at i to a state vector
                const auto stateAtI = Kokkos::subview(endStateDevice, chemIndex, Kokkos::ALL());
                Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, stateAtI);
                auto pressureAtI = stateVector.Pressure();
                if (pressureAtI < pressureMin) {
                    pressureMin = pressureAtI;
                }
            },
            Kokkos::Min<double>(minimumPressure));
    }

    // Use a parallel for computing the source term
    auto enthalpyOfFormation = eos->GetEnthalpyOfFormation();
    Kokkos::parallel_for(
        "sourceTermCompute", Kokkos::RangePolicy<typename tChemLib::exec_space>(cellRange.start, cellRange.end), KOKKOS_LAMBDA(const auto i) {
            // get the host data from the petsc field
            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
            const std::size_t chemIndex = i - cellRange.start;

            // cast the state at i to a state vector
            const auto stateAtI = Kokkos::subview(stateDevice, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, stateAtI);
            const auto ys = stateVector.MassFractions();

            const auto endStateAtI = Kokkos::subview(endStateDevice, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> endStateVector(kineticModelGasConstDataDevice.nSpec, endStateAtI);
            const auto ye = endStateVector.MassFractions();

            // get the source term at this chemIndex
            const auto sourceTermAtI = Kokkos::subview(sourceTermsDevice, chemIndex, Kokkos::ALL());

            // the IgnitionZeroD::runDeviceBatch sets the pressure to zero if it does not converge
            if (endStateVector.Pressure() > 0) {
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
            } else {
                // set to zero
                sourceTermAtI(0) = 0.0;
                for (ordinal_type s = 0; s < stateVector.NumSpecies(); ++s) {
                    sourceTermAtI(s + 1) = 0.0;
                }

                // compute the cell centroid
                PetscReal centroid[3];
                DMPlexComputeCellGeometryFVM(dm, cell, nullptr, centroid, nullptr) >> checkError;

                // Output error information
                std::stringstream warningMessage;
                warningMessage << "Warning: Could not integrate chemistry at cell " << cell << " on rank " << rank << " at location " << utilities::VectorUtilities::Concatenate(centroid, dim) << "\n";
                warningMessage << "dt: " << std::setprecision(16) << dt << "\n";
                warningMessage << "state: "
                               << "\n";
                for (std::size_t s = 0; s < stateAtI.size(); ++s) {
                    warningMessage << "\t[" << s << "] " << stateAtI[s] << "\n";
                }

                std::cout << warningMessage.str() << std::endl;
            }
        });

    // copy the updated state back to host
    Kokkos::deep_copy(sourceTermsHost, sourceTermsDevice);

    // clean up
    solver.RestoreRange(cellRange);
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

            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += sourceAtI[0];
            for (std::size_t sp = 0; sp < process->numberSpecies; sp++) {
                densityYiSource[sp] += sourceAtI(sp + 1);
            }
        });

    // cleanup
    solver.RestoreRange(cellRange);
    PetscCall(VecRestoreArray(locFVec, &fArray));

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::processes::TChemReactions::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, Vec locFVec) {
    AddChemistrySourceToFlow(solver, solver.GetSubDomain().GetDM(), NAN, nullptr, locFVec, this) >> checkError;
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::TChemReactions, "reactions using the TChem library", ARG(ablate::eos::EOS, "eos", "the tChem eos"),
         OPT(ablate::parameters::Parameters, "options",
             "time stepping options (dtMin, dtMax, dtDefault, dtEstimateFactor, relToleranceTime, relToleranceTime, absToleranceTime, relToleranceNewton, absToleranceNewton, maxNumNewtonIterations, "
             "numTimeIterationsPerInterval, jacobianInterval, maxAttempts)"));
