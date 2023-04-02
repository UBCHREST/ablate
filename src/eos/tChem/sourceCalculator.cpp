#include "sourceCalculator.hpp"
#include <TChem_Impl_IgnitionZeroD_Problem.hpp>
#include <algorithm>
#include "eos/tChem.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "ignitionZeroDTemperatureThreshold.hpp"
#include "utilities/mpiUtilities.hpp"

void ablate::eos::tChem::SourceCalculator::ChemistryConstraints::Set(const std::shared_ptr<ablate::parameters::Parameters>& options) {
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
        thresholdTemperature = options->Get("thresholdTemperature", thresholdTemperature);
    }
}

ablate::eos::tChem::SourceCalculator::SourceCalculator(const std::vector<domain::Field>& fields, const std::shared_ptr<TChem> eosIn,
                                                       ablate::eos::tChem::SourceCalculator::ChemistryConstraints constraints, const ablate::domain::Range& cellRange)
    : chemistryConstraints(constraints), eos(eosIn), numberSpecies(eosIn->GetSpeciesVariables().size()) {
    // determine the number of required cells
    std::size_t numberCells = cellRange.end - cellRange.start;

    // compute the required state dimension
    auto kineticModelGasConstData = createGasKineticModelConstData<typename Tines::UseThisDevice<host_exec_space>::type>(eos->GetKineticModelData());
    const ordinal_type stateVecDim = Impl::getStateVectorSize(kineticModelGasConstData.nSpec);

    // allocate the tChem memory
    stateDevice = real_type_2d_view("stateVectorDevices", numberCells, stateVecDim);
    stateHost = Kokkos::create_mirror(stateDevice);
    endStateDevice = real_type_2d_view("stateVectorDevicesEnd", numberCells, stateVecDim);
    internalEnergyRefDevice =  real_type_1d_view("internalEnergyRefHost", numberCells);
    internalEnergyRefHost =Kokkos::create_mirror(internalEnergyRefDevice);
    sourceTermsDevice = real_type_2d_view("sourceTermsHost", numberCells, kineticModelGasConstData.nSpec + 1);
    sourceTermsHost = Kokkos::create_mirror(sourceTermsDevice);
    perSpeciesScratchDevice = real_type_2d_view("perSpeciesScratchDevice", numberCells, kineticModelGasConstData.nSpec);
    timeViewDevice = real_type_1d_view("time", numberCells);
    dtViewDevice = real_type_1d_view("delta time", numberCells);

    // Create the default timeAdvanceObject
    timeAdvanceDefault._tbeg = 0.0;
    timeAdvanceDefault._tend = 1.0;
    timeAdvanceDefault._dt = constraints.dtDefault;
    timeAdvanceDefault._dtmin = constraints.dtMin;
    timeAdvanceDefault._dtmax = constraints.dtMax;
    timeAdvanceDefault._max_num_newton_iterations = constraints.maxNumNewtonIterations;
    timeAdvanceDefault._num_time_iterations_per_interval = constraints.numTimeIterationsPerInterval;
    timeAdvanceDefault._num_outer_time_iterations_per_interval = 10;
    timeAdvanceDefault._jacobian_interval = constraints.jacobianInterval;

    // Copy the default values to device
    timeAdvanceDevice = time_advance_type_1d_view("timeAdvanceDevice", numberCells);
    Kokkos::deep_copy(timeAdvanceDevice, timeAdvanceDefault);
    Kokkos::deep_copy(dtViewDevice, 1E-4);

    // determine the number of equations
    auto numberOfEquations = ::tChemLib::Impl::IgnitionZeroD_Problem<real_type, Tines::UseThisDevice<host_exec_space>::type>::getNumberOfTimeODEs(kineticModelGasConstData);

    // size up the tolerance constraints
    tolTimeDevice = real_type_2d_view("tolTimeDevice", numberOfEquations, 2);
    tolNewtonDevice = real_type_1d_view("tolNewtonDevice", 2);
    facDevice = real_type_2d_view("facDevice", numberCells, numberOfEquations);

    /// tune tolerance
    {
        auto tolTimeHost = Kokkos::create_mirror_view(tolTimeDevice);
        auto tolNewtonHost = Kokkos::create_mirror_view(tolNewtonDevice);

        for (ordinal_type i = 0, iend = tolTimeDevice.extent(0); i < iend; ++i) {
            tolTimeHost(i, 0) = constraints.absToleranceTime;  // atol
            tolTimeHost(i, 1) = constraints.relToleranceTime;  // rtol
        }
        tolNewtonHost(0) = constraints.absToleranceNewton;  // atol
        tolNewtonHost(1) = constraints.relToleranceNewton;  // rtol

        Kokkos::deep_copy(tolTimeDevice, tolTimeHost);
        Kokkos::deep_copy(tolNewtonDevice, tolNewtonHost);
    }

    // get device kineticModelGasConstData
    kineticModelGasConstDataDevice = ::tChemLib::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(eos->GetKineticModelData());
    kineticModelDataClone = eos->GetKineticModelData().clone(numberCells);
    kineticModelGasConstDataDevices = ::tChemLib::createGasKineticModelConstData<typename Tines::UseThisDevice<exec_space>::type>(kineticModelDataClone);

    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("ablate::eos::tChem::BatchSource requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }
    eulerId = eulerField->id;

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("ablate::eos::tChem::BatchSource requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
    }
    densityYiId = densityYiField->id;
}

void ablate::eos::tChem::SourceCalculator::ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globFlowVec) {
    StartEvent("tChem::SourceCalculator::ComputeSource");
    // Get the valid cell range over this region
    auto numberCells = cellRange.end - cellRange.start;

    // Get the solution dm
    DM solutionDm;
    VecGetDM(globFlowVec, &solutionDm) >> utilities::PetscUtilities::checkError;

    // get the rank
    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)solutionDm), &rank) >> utilities::MpiUtilities::checkError;

    // get the flowSolution
    const PetscScalar* flowArray;
    VecGetArrayRead(globFlowVec, &flowArray) >> utilities::PetscUtilities::checkError;

    PetscInt dim;
    DMGetDimension(solutionDm, &dim) >> utilities::PetscUtilities::checkError;

    // Use a parallel for loop to load up the tChem state
    Kokkos::parallel_for(
        "stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), [&](const auto i) {
            // get the host data from the petsc field
            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
            const std::size_t chemIndex = i - cellRange.start;

            // Get the current state variables for this cell
            const PetscScalar* eulerField = nullptr;
            DMPlexPointLocalFieldRead(solutionDm, cell, eulerId, flowArray, &eulerField) >> utilities::PetscUtilities::checkError;
            const PetscScalar* flowDensityField = nullptr;
            DMPlexPointLocalFieldRead(solutionDm, cell, densityYiId, flowArray, &flowDensityField) >> utilities::PetscUtilities::checkError;

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
    auto temperatureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(::tChemLib::exec_space(), numberCells, Kokkos::AUTO());
    temperatureFunctionPolicy.set_scratch_size(
        1, Kokkos::PerTeam(::tChemLib::Scratch<real_type_1d_view>::shmem_size(ablate::eos::tChem::Temperature::getWorkSpaceSize(kineticModelGasConstDataDevice.nSpec))));
    auto pressureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(::tChemLib::exec_space(), numberCells, Kokkos::AUTO());
    pressureFunctionPolicy.set_scratch_size(1,
                                            Kokkos::PerTeam(::tChemLib::Scratch<real_type_1d_view>::shmem_size(ablate::eos::tChem::Pressure::getWorkSpaceSize(kineticModelGasConstDataDevice.nSpec))));

    // Compute temperature into the state field in the device
    ablate::eos::tChem::Temperature::runDeviceBatch(
        temperatureFunctionPolicy, stateDevice, internalEnergyRefDevice, perSpeciesScratchDevice, eos->GetEnthalpyOfFormation(), kineticModelGasConstDataDevice);

    // Compute the pressure into the state field in the device
    ablate::eos::tChem::Pressure::runDeviceBatch(pressureFunctionPolicy, stateDevice, kineticModelGasConstDataDevice);

    double minimumPressure = 0;
    for (int attempt = 0; (attempt < chemistryConstraints.maxAttempts) && minimumPressure == 0; ++attempt) {
        // Use a parallel for updating timeAdvanceDevice dt
        Kokkos::parallel_for(
            "timeAdvanceUpdate", Kokkos::RangePolicy<typename tChemLib::exec_space>(0, numberCells), KOKKOS_LAMBDA(const auto i) {
//                auto& tAdvAtI = timeAdvanceDevice(i);
//                tAdvAtI._tbeg = time;
//                tAdvAtI._tend = time + dt;
//                tAdvAtI._dt = PetscMax(PetscMin(PetscMin(dtViewDevice(i) * chemistryConstraints.dtEstimateFactor, dt), tAdvAtI._dtmax) / (PetscPowInt(2, attempt)), tAdvAtI._dtmin);
//                // set the default time information
//                timeViewDevice(i) = time;
            });

        auto chemistryFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(::tChemLib::exec_space(), numberCells, Kokkos::AUTO());
        chemistryFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam(::tChemLib::Scratch<real_type_1d_view>::shmem_size(::tChemLib::IgnitionZeroD::getWorkSpaceSize(kineticModelGasConstDataDevice))));

        // assume a constant pressure zero D reaction for each cell
        if (chemistryConstraints.thresholdTemperature != 0.0) {
            // If there is a thresholdTemperature, use the modified version of IgnitionZeroDTemperatureThreshold
            ablate::eos::tChem::IgnitionZeroDTemperatureThreshold::runDeviceBatch(chemistryFunctionPolicy,
                                                                                  tolNewtonDevice,
                                                                                  tolTimeDevice,
                                                                                  facDevice,
                                                                                  timeAdvanceDevice,
                                                                                  stateDevice,
                                                                                  timeViewDevice,
                                                                                  dtViewDevice,
                                                                                  endStateDevice,
                                                                                  kineticModelGasConstDataDevices,
                                                                                  chemistryConstraints.thresholdTemperature);
        } else {
            // else fall back to the default tChem version
            tChemLib::IgnitionZeroD::runDeviceBatch(
                chemistryFunctionPolicy, tolNewtonDevice, tolTimeDevice, facDevice, timeAdvanceDevice, stateDevice, timeViewDevice, dtViewDevice, endStateDevice, kineticModelGasConstDataDevices);
        }
        // check the output pressure, if it is zero the integration failed
//        Kokkos::parallel_reduce(
//            "pressureCheck",
//            Kokkos::RangePolicy<typename tChemLib::exec_space>(0, numberCells),
//            KOKKOS_LAMBDA(const int& chemIndex, double& pressureMin) {
//                // cast the state at i to a state vector
//                const auto stateAtI = Kokkos::subview(endStateDevice, chemIndex, Kokkos::ALL());
//                Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, stateAtI);
//                auto pressureAtI = stateVector.Pressure();
//                if (pressureAtI < pressureMin) {
//                    pressureMin = pressureAtI;
//                }
//            },
//            Kokkos::Min<double>(minimumPressure));
    }

    // Use a parallel for computing the source term
    auto enthalpyOfFormation = eos->GetEnthalpyOfFormation();
//    Kokkos::parallel_for(
//        "sourceTermCompute", Kokkos::RangePolicy<typename tChemLib::exec_space>(cellRange.start, cellRange.end), KOKKOS_LAMBDA(const auto i) {
//            // get the host data from the petsc field
//            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
//            const std::size_t chemIndex = i - cellRange.start;
//
//            // cast the state at i to a state vector
//            const auto stateAtI = Kokkos::subview(stateDevice, chemIndex, Kokkos::ALL());
//            Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, stateAtI);
//            const auto ys = stateVector.MassFractions();
//
//            const auto endStateAtI = Kokkos::subview(endStateDevice, chemIndex, Kokkos::ALL());
//            Impl::StateVector<real_type_1d_view> endStateVector(kineticModelGasConstDataDevice.nSpec, endStateAtI);
//            const auto ye = endStateVector.MassFractions();
//
//            // get the source term at this chemIndex
//            const auto sourceTermAtI = Kokkos::subview(sourceTermsDevice, chemIndex, Kokkos::ALL());
//
//            // the IgnitionZeroD::runDeviceBatch sets the pressure to zero if it does not converge
//            if (endStateVector.Pressure() > 0) {
//                // compute the source term from the change in the heat of formation
//                sourceTermAtI(0) = 0.0;
//                for (ordinal_type s = 0; s < stateVector.NumSpecies(); s++) {
//                    sourceTermAtI(0) += (ys(s) - ye(s)) * enthalpyOfFormation(s);
//                }
//
//                for (ordinal_type s = 0; s < stateVector.NumSpecies(); ++s) {
//                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
//                    sourceTermAtI(s + 1) = ye(s) - ys(s);
//                }
//
//                // Now scale everything by density/dt
//                for (std::size_t j = 0; j < sourceTermAtI.extent(0); ++j) {
//                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
//                    sourceTermAtI(j) *= stateVector.Density() / dt;
//                }
//            } else {
//                // set to zero
//                sourceTermAtI(0) = 0.0;
//                for (ordinal_type s = 0; s < stateVector.NumSpecies(); ++s) {
//                    sourceTermAtI(s + 1) = 0.0;
//                }
//
//                // compute the cell centroid
//                 if constexpr(std::is_same<tChemLib::exec_space,tChemLib::host_exec_space>::value) {
//                    PetscReal centroid[3];
//                    DMPlexComputeCellGeometryFVM(solutionDm, cell, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;
//
//                    // Output error information
//                    std::stringstream warningMessage;
//                    warningMessage << "Warning: Could not integrate chemistry at cell " << cell << " on rank " << rank << " at location " << utilities::VectorUtilities::Concatenate(centroid, dim)
//                                   << "\n";
//                    warningMessage << "dt: " << std::setprecision(16) << dt << "\n";
//                    warningMessage << "state: "
//                                   << "\n";
//                    for (std::size_t s = 0; s < stateAtI.size(); ++s) {
//                        warningMessage << "\t[" << s << "] " << stateAtI[s] << "\n";
//                    }
//
//                    std::cout << warningMessage.str() << std::endl;
//                }else{
//                    printf("Warning: Could not integrate chemistry at cell %d on rank %d\n", cell, rank );
//                }
//            }
//        });

    // copy the updated state back to host
    Kokkos::deep_copy(sourceTermsHost, sourceTermsDevice);
    EndEvent();
}
void ablate::eos::tChem::SourceCalculator::AddSource(const ablate::domain::Range& cellRange, Vec, Vec locFVec) {
    StartEvent("tChem::SourceCalculator::AddSource");
    // get access to the fArray
    PetscScalar* fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

    // Get the solution dm
    DM dm;
    VecGetDM(locFVec, &dm) >> utilities::PetscUtilities::checkError;

    // Use a parallel for loop to load up the tChem state
    Kokkos::parallel_for(
        "stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), [&](const auto i) {
            // get the host data from the petsc field
            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
            const std::size_t chemIndex = i - cellRange.start;

            // Get the current state variables for this cell
            PetscScalar* eulerSource = nullptr;
            DMPlexPointLocalFieldRef(dm, cell, eulerId, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
            PetscScalar* densityYiSource = nullptr;
            DMPlexPointLocalFieldRef(dm, cell, densityYiId, fArray, &densityYiSource) >> utilities::PetscUtilities::checkError;

            // cast the state at i to a state vector
            const auto sourceAtI = Kokkos::subview(sourceTermsHost, chemIndex, Kokkos::ALL());

            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += sourceAtI[0];
            for (std::size_t sp = 0; sp < numberSpecies; sp++) {
                densityYiSource[sp] += sourceAtI(sp + 1);
            }
        });

    // cleanup
    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    EndEvent();
}
