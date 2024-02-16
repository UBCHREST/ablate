#include "sourceCalculator2.hpp"
#include <TChem_ConstantVolumeIgnitionReactor.hpp>
#include <TChem_Impl_IgnitionZeroD_Problem.hpp>
#include <algorithm>
#include "constantVolumeIgnitionReactorTemperatureThreshold.hpp"
#include "eos/tChem2.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "ignitionZeroDTemperatureThreshold.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/stringUtilities.hpp"
#include <numeric>

#include "zerork_cfd_plugin.h"

ablate::eos::tChem2::SourceCalculator2::SourceCalculator2(const std::vector<domain::Field>& fields, const std::shared_ptr<TChem2> eosIn,
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
    internalEnergyRefDevice = real_type_1d_view("internalEnergyRefHost", numberCells);
    internalEnergyRefHost = Kokkos::create_mirror(internalEnergyRefDevice);
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
    timeAdvanceDefault._num_outer_time_iterations_per_interval = 1;  // This should always be one to prevent duplicate operations
    timeAdvanceDefault._jacobian_interval = constraints.jacobianInterval;

    // Copy the default values to device
    timeAdvanceDevice = time_advance_type_1d_view("timeAdvanceDevice", numberCells);
    Kokkos::deep_copy(timeAdvanceDevice, timeAdvanceDefault);
    Kokkos::deep_copy(dtViewDevice, 1E-4);

    // determine the number of equations
    ordinal_type numberOfEquations;
    switch (chemistryConstraints.reactorType) {
        case tChem::SourceCalculator::ReactorType::ConstantPressure:
            numberOfEquations = ::tChemLib::Impl::IgnitionZeroD_Problem<real_type, Tines::UseThisDevice<host_exec_space>::type>::getNumberOfTimeODEs(kineticModelGasConstData);
            break;
        case tChem::SourceCalculator::ReactorType::ConstantVolume:
            numberOfEquations = ::tChemLib::Impl::ConstantVolumeIgnitionReactor_Problem<real_type, Tines::UseThisDevice<host_exec_space>::type>::getNumberOfTimeODEs(kineticModelGasConstData);
            break;
    }

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




    int zerork_error_state = 0;
    zrm_handle = zerork_reactor_init();


    //We already parsed in reactor manager no need to have another log
    // TODO: Avoid parsing twice/having two mechanisms
    mech = std::make_unique<zerork::mechanism>("MMAReduced.inp", "MMAReduced.dat", cklogfilename);

//    zerork::mechanism mech("MMAReduced.inp", "MMAReduced.dat", cklogfilename);

//    zerork_status_t zerom_status = zerork_reactor_set_mechanism_files(eos::TChemBase::reactionFile.c_str(), eos::TChemBase::thermoFile.c_str(), zrm_handle);
    zerork_status_t zerom_status = zerork_reactor_set_mechanism_files("MMAReduced.inp", "MMAReduced.dat", zrm_handle);
    if(zerom_status != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;
    zerork_status_t zerom_status2 = zerork_reactor_set_string_option("mechanism_parsing_log_filename","mech.cklog",zrm_handle);
    zerork_status_t status_mech = zerork_reactor_load_mechanism(zrm_handle);
//    if(status_mech != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;
    zerork_status_t status_other = zerork_reactor_set_int_option("constant_volume", 0, zrm_handle);
    if(status_other != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;


    if (zerork_error_state!=0) {
        throw std::invalid_argument("ablate::eos::TChem2 can only read inp and dat files.");
    }



}

void ablate::eos::tChem2::SourceCalculator2::ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globFlowVec) {
    StartEvent("tChem2::SourceCalculator::ComputeSource");
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
    Kokkos::parallel_for("stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), [&](const auto i) {
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


//zerork state load up
    int nSpc=kineticModelGasConstDataDevice.nSpec;//ps the soot mech has C(s) // Number of Species
    int nState=nSpc+1;
    int nSpcStride = nSpc;

    int nReactors = numberCells;

    //Set up reactor initial states
    int nReactorsAlloc = nReactors;
    std::vector<double> reactorT(nReactors);
    std::vector<double> reactorP(nReactors);
    std::vector<double> density2(nReactors);
    std::vector<double> sensibleenergy2(nReactors);
    std::vector<double> velmag2(nReactors);
    std::vector<double> reactorMassFrac(nReactors*nSpc);
    std::vector<double> enthapyOfFormation(nSpc);
    std::vector<double> internalenergies(nReactors*nSpc);

    //load up current state from petsc
    const double refTemperature = 300.0;
    for(int k=0; k<nReactors; ++k) {
        const PetscInt cell = cellRange.points ? cellRange.points[k] : k;
        const std::size_t chemIndex = k - cellRange.start;

        const PetscScalar* eulerField = nullptr;
        DMPlexPointLocalFieldRead(solutionDm, cell, eulerId, flowArray, &eulerField) >> utilities::PetscUtilities::checkError;
        const PetscScalar* flowDensityField = nullptr;
        DMPlexPointLocalFieldRead(solutionDm, cell, densityYiId, flowArray, &flowDensityField) >> utilities::PetscUtilities::checkError;


        // get the current state at I
        auto density = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHO];
        auto densityE = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOE];
        density2[k]=density;

        real_type yiSum = 0.0;
        for (ordinal_type s = 0; s < nSpc - 1; s++) {
            reactorMassFrac[k * nSpc + s] = PetscMax(0.0, flowDensityField[k * nSpc + s] / density);
            reactorMassFrac[k * nSpc + s] = PetscMin(1.0, reactorMassFrac[k * nSpc + s]);
            yiSum += reactorMassFrac[k + s];
        }
        if (yiSum > 1.0) {
            for (PetscInt s = 0; s < nSpc - 1; s++) {
                // Limit the bounds
                reactorMassFrac[k * nSpc + s] /= yiSum;
            }
    //        reactorMassFrac[nSpc - 1] = 0.0;
        } else {
            reactorMassFrac[k * nSpc + nSpc - 1] = 1.0 - yiSum;
        }

        // Compute the internal energy from total ener
        PetscReal speedSquare = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            speedSquare += PetscSqr(eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
        }

        // compute the internal energy needed to compute temperature
        sensibleenergy2[k]= eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;

        double enthalpymix = mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[k*nSpc]);

        sensibleenergy2[k] += enthalpymix;

//        for (int s = 0; s < nSpc; s++) {
//
//
//        }

        double temp = mech->getTemperatureFromEY(sensibleenergy2[k], &reactorMassFrac[k*nSpc], 2000);

        std::vector<double> energytest(nSpc,0);
//        mech->getIntEnergy_RT(2000, &energytest[0]);
        double intener = mech->getMassIntEnergyFromTY(2000, &reactorMassFrac[k*nSpc],&energytest[0]);

        double totalenergy = accumulate(energytest.begin(),energytest.end(),0);

        reactorT[k] = temp;
        reactorP[k] = 101325;

    }

    //    reactorMassFrac[1] = 0.5;
    //    reactorMassFrac[4] = 0.5;
    std::vector<double> ys2 = reactorMassFrac;


    zerork_status_t flag = ZERORK_STATUS_SUCCESS;
    flag = zerork_reactor_solve(1, time, dt, nReactors, &reactorT[0], &reactorP[0],
                                &reactorMassFrac[0], zrm_handle);

    if(flag != ZERORK_STATUS_SUCCESS) printf("Oo something went wrong during zreork integration...");

    for (ordinal_type s = 0; s < nSpc - 1; s++) {
        std::vector<double> tempvec(nSpc,0.);
        tempvec[s]=1;
//        double* a = &tempvec[0];
        enthapyOfFormation[s] = mech->getMassEnthalpyFromTY(298.15, &tempvec[0]);
    }

    //Create source terms
    std::vector<double> sourceZeroRKAtI(nReactors * (nState));


    sourceZeroRKAtI[0] = 0.0;
    for(int k=0; k<nReactors; ++k) {

        for (int s = 0; s < nSpc; s++) {
//            sourceZeroRKAtI[k] += (ys2[k * nSpc + s] - reactorMassFrac[k * nSpc + s]) * enthalpyOfFormationLocal[k + s];
            sourceZeroRKAtI[k * nSpc] += (ys2[k * nSpc + s] - reactorMassFrac[k * nSpc + s]) * enthapyOfFormation[s];
        }

        for (int s = 0; s < nSpc; ++s) {
            // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
            sourceZeroRKAtI[k * nSpc + s + 1] = reactorMassFrac[k * nSpc + s] - ys2[k * nSpc + s];
        }

        // Now scale everything by density/dt
        for (int j = 0; j < nState; ++j) {
            // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
            sourceZeroRKAtI[k * nSpc + j] *= density2[k] / dt;
        }
    }









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

    auto timeAdvanceDeviceLocal = timeAdvanceDevice;
    auto dtViewDeviceLocal = dtViewDevice;
    auto chemistryConstraintsLocal = chemistryConstraints;
    auto timeViewDeviceLocal = timeViewDevice;

    double minimumPressure = 0;
    for (int attempt = 0; (attempt < chemistryConstraints.maxAttempts) && minimumPressure == 0; ++attempt) {
        auto factor = PetscPowInt(2, attempt);
        // Use a parallel for updating timeAdvanceDevice dt
        Kokkos::parallel_for(
            "timeAdvanceUpdate", Kokkos::RangePolicy<tChemLib::exec_space>(0, numberCells), KOKKOS_LAMBDA(const ordinal_type& i) {
                auto& tAdvAtI = timeAdvanceDeviceLocal(i);

                tAdvAtI._tbeg = time;
                tAdvAtI._tend = time + dt;
                tAdvAtI._dt = Kokkos::max(Kokkos::min(Kokkos::min(dtViewDeviceLocal(i) * chemistryConstraintsLocal.dtEstimateFactor, dt), tAdvAtI._dtmax) / factor, tAdvAtI._dtmin);
                // set the default time information
                timeViewDeviceLocal(i) = time;
            });

        auto chemistryFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(::tChemLib::exec_space(), numberCells, Kokkos::AUTO());

        // determine the required team size
        switch (chemistryConstraints.reactorType) {
            case tChem::SourceCalculator::ReactorType::ConstantPressure:
                chemistryFunctionPolicy.set_scratch_size(
                    1, Kokkos::PerTeam(::tChemLib::Scratch<real_type_1d_view>::shmem_size(::tChemLib::IgnitionZeroD::getWorkSpaceSize(kineticModelGasConstDataDevice))));
                break;
            case tChem::SourceCalculator::ReactorType::ConstantVolume:
                chemistryFunctionPolicy.set_scratch_size(
                    1, Kokkos::PerTeam(::tChemLib::Scratch<real_type_1d_view>::shmem_size(::tChemLib::ConstantVolumeIgnitionReactor::getWorkSpaceSize(solveTla, kineticModelGasConstDataDevice))));
                break;
        }




        // assume a constant pressure zero D reaction for each cell
        switch (chemistryConstraints.reactorType) {
            case tChem::SourceCalculator::ReactorType::ConstantPressure:
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
                    tChemLib::IgnitionZeroD::runDeviceBatch(chemistryFunctionPolicy,
                                                            tolNewtonDevice,
                                                            tolTimeDevice,
                                                            facDevice,
                                                            timeAdvanceDevice,
                                                            stateDevice,
                                                            timeViewDevice,
                                                            dtViewDevice,
                                                            endStateDevice,
                                                            kineticModelGasConstDataDevices);
                }
                break;
            case tChem::SourceCalculator::ReactorType::ConstantVolume:
                // These arrays are not used when solveTla is false
                real_type_3d_view state_z;
                if (chemistryConstraints.thresholdTemperature != 0.0) {
                    ablate::eos::tChem::ConstantVolumeIgnitionReactorTemperatureThreshold::runDeviceBatch(chemistryFunctionPolicy,
                                                                                                          solveTla,
                                                                                                          thetaTla,
                                                                                                          tolNewtonDevice,
                                                                                                          tolTimeDevice,
                                                                                                          facDevice,
                                                                                                          timeAdvanceDevice,
                                                                                                          stateDevice,
                                                                                                          state_z,
                                                                                                          timeViewDevice,
                                                                                                          dtViewDevice,
                                                                                                          endStateDevice,
                                                                                                          state_z,
                                                                                                          kineticModelGasConstDataDevices,
                                                                                                          chemistryConstraints.thresholdTemperature);
                } else {
                    ConstantVolumeIgnitionReactor::runDeviceBatch(chemistryFunctionPolicy,
                                                                  solveTla,
                                                                  thetaTla,
                                                                  tolNewtonDevice,
                                                                  tolTimeDevice,
                                                                  facDevice,
                                                                  timeAdvanceDevice,
                                                                  stateDevice,
                                                                  state_z,
                                                                  timeViewDevice,
                                                                  dtViewDevice,
                                                                  endStateDevice,
                                                                  state_z,
                                                                  kineticModelGasConstDataDevices);
                }

                break;
        }






        // check the output pressure, if it is zero the integration failed
        auto endStateDeviceLocal = endStateDevice;
        auto nSpecLocal = kineticModelGasConstDataDevice.nSpec;
        Kokkos::parallel_reduce(
            "pressureCheck",
            Kokkos::RangePolicy<typename tChemLib::exec_space>(0, numberCells),
            KOKKOS_LAMBDA(const int& chemIndex, double& pressureMin) {
                // cast the state at i to a state vector
                const auto stateAtI = Kokkos::subview(endStateDeviceLocal, chemIndex, Kokkos::ALL());
                Impl::StateVector<real_type_1d_view> stateVector(nSpecLocal, stateAtI);
                auto pressureAtI = stateVector.Pressure();
                if (pressureAtI < pressureMin) {
                    pressureMin = pressureAtI;
                }
            },
            Kokkos::Min<double>(minimumPressure));
    }

    // Get the local copies
    auto stateDeviceLocal = stateDevice;
    auto endStateDeviceLocal = endStateDevice;
    auto nSpecLocal = kineticModelGasConstDataDevice.nSpec;
    auto sourceTermsDeviceLocal = sourceTermsDevice;
    auto cellRangeStartLocal = cellRange.start;
    // Use a parallel for computing the source term
    auto enthalpyOfFormationLocal = eos->GetEnthalpyOfFormation();
    Kokkos::parallel_for(
        "sourceTermCompute", Kokkos::RangePolicy<typename tChemLib::exec_space>(cellRange.start, cellRange.end), KOKKOS_LAMBDA(const ordinal_type& i) {
            // get the host data from the petsc field
            const std::size_t chemIndex = i - cellRangeStartLocal;

            // cast the state at i to a state vector
            const auto stateAtI = Kokkos::subview(stateDeviceLocal, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> stateVector(nSpecLocal, stateAtI);
            const auto ys = stateVector.MassFractions();

            std::vector<double> ystart(nSpecLocal);
            for (int k = 0;k<nSpecLocal;k++){
                ystart[k]=ys(k);
            }
//            std::vector<double> ystart = stateVector.MassFractions();

            const auto endStateAtI = Kokkos::subview(endStateDeviceLocal, chemIndex, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> endStateVector(nSpecLocal, endStateAtI);
            const auto ye = endStateVector.MassFractions();

            std::vector<double> yend(nSpecLocal);
            std::vector<double> ydiff(nSpecLocal);
            for (int k = 0;k<nSpecLocal;k++){
                yend[k]=ye(k);
                ydiff[k]=ye(k)-ys(k);
            }


            // get the source term at this chemIndex
            const auto sourceTermAtI = Kokkos::subview(sourceTermsDeviceLocal, chemIndex, Kokkos::ALL());

            // the IgnitionZeroD::runDeviceBatch sets the pressure to zero if it does not converge
            if (endStateVector.Pressure() > 0) {
                // compute the source term from the change in the heat of formation
                sourceTermAtI(0) = 0.0;
                for (ordinal_type s = 0; s < stateVector.NumSpecies(); s++) {
                    sourceTermAtI(0) += (ys(s) - ye(s)) * enthalpyOfFormationLocal(s);
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

#ifndef KOKKOS_ENABLE_CUDA
                // compute the cell centroid
                PetscReal centroid[3];
                const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
                DMPlexComputeCellGeometryFVM(solutionDm, cell, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

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
#else
                        printf("Warning: Could not integrate chemistry at index %d on rank %d\n", (int)i, rank );
#endif
            }
        });

    // copy the updated state back to host
    Kokkos::deep_copy(sourceTermsHost, sourceTermsDevice);
    EndEvent();
}
void ablate::eos::tChem2::SourceCalculator2::AddSource(const ablate::domain::Range& cellRange, Vec, Vec locFVec) {
    StartEvent("tChem::SourceCalculator::AddSource");
    // get access to the fArray
    PetscScalar* fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

    // Get the solution dm
    DM dm;
    VecGetDM(locFVec, &dm) >> utilities::PetscUtilities::checkError;

    bool usetchemsources = true;
    auto numberCells = cellRange.end - cellRange.start;

    if (usetchemsources) {
        // Use a parallel for loop to load up the tChem state
        Kokkos::parallel_for("stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), [&](const auto i) {
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

            std::vector <double> sources(numberSpecies+1);

            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += sourceAtI[0];
            sources[0]=sourceAtI(0);
            for (std::size_t sp = 0; sp < numberSpecies; sp++) {
                densityYiSource[sp] += sourceAtI(sp + 1);
                sources[sp+1]=sourceAtI(sp + 1);
            }
            double a=1.;
        });
    } else{


        for (int k = 0 ; k<numberCells;k++) {
            const PetscInt cell = cellRange.points ? cellRange.points[k] : k;
            const std::size_t chemIndex = k - cellRange.start;

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
        }
    }

    // cleanup
    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    EndEvent();
}

std::ostream& ablate::eos::tChem2::operator<<(std::ostream& os, const ablate::eos::tChem2::SourceCalculator2::ReactorType& v) {
    switch (v) {
        case ablate::eos::tChem2::SourceCalculator2::ReactorType::ConstantPressure:
            return os << "ConstantPressure";
        case ablate::eos::tChem2::SourceCalculator2::ReactorType::ConstantVolume:
            return os << "ConstantVolume";
        default:
            return os;
    }
}

std::istream& ablate::eos::tChem2::operator>>(std::istream& is, ablate::eos::tChem2::SourceCalculator2::ReactorType& v) {
    std::string enumString;
    is >> enumString;

    // make the comparisons easier to converting to lower
    ablate::utilities::StringUtilities::ToLower(enumString);

    if (enumString == "constantvolume") {
        v = ablate::eos::tChem2::SourceCalculator2::ReactorType::ConstantVolume;
    } else {
        // default to constant pressure
        v = ablate::eos::tChem2::SourceCalculator2::ReactorType::ConstantPressure;
    }
    return is;
}