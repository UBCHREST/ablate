#include "tChemReactions.hpp"
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
    flow.GetCellRange(cellRange);

    // determine the number of required cells
    std::size_t numberCells = cellRange.end - cellRange.start;
    flow.RestoreRange(cellRange);

    // compute the required state dimension
    auto kineticModelGasConstData = TChem::createGasKineticModelConstData<typename Tines::UseThisDevice<host_exec_space>::type>(eos->GetKineticModelData());
    const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(kineticModelGasConstData.nSpec);

    // allocate the tChem memory
    stateHost = real_type_2d_view_host("stateVectorDevices", numberCells, stateVecDim);
    stateDevice = Kokkos::create_mirror_view(stateHost);
    endStateHost = real_type_2d_view_host("stateVectorDevicesEnd", numberCells, stateVecDim);
    endStateDevice = Kokkos::create_mirror_view(endStateHost);
    internalEnergyRefHost = real_type_1d_view_host("internalEnergyRefHost", numberCells);
    internalEnergyRefDevice = Kokkos::create_mirror_view(internalEnergyRefHost);
    perSpeciesScratchDevice = real_type_2d_view("perSpeciesScratchDevice", numberCells, kineticModelGasConstData.nSpec);

    // Create the default timeAdvanceObject
    time_advance_type timeAdvanceDefault;
    timeAdvanceDefault._tbeg = 0.0;
    timeAdvanceDefault._tend = 1.0;
    timeAdvanceDefault._dt = 1.0E-8;
    timeAdvanceDefault._dtmin = 1.0E-8;
    timeAdvanceDefault._dtmax = 1.0E-1;
    timeAdvanceDefault._max_num_newton_iterations = 100;
    timeAdvanceDefault._num_time_iterations_per_interval = 1E1;
    timeAdvanceDefault._num_outer_time_iterations_per_interval = 1;
    timeAdvanceDefault._jacobian_interval = 1;

    // Copy the default values to device
    timeAdvanceDevice = time_advance_type_1d_view("timeAdvanceDevice", numberCells);
    Kokkos::deep_copy(timeAdvanceDevice, timeAdvanceDefault);

    // setup the temperature function policy
    temperatureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(numberCells, Kokkos::AUTO());
    temperatureFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam((int)ablate::eos::tChem::Temperature::getWorkSpaceSize(kineticModelGasConstData.nSpec)));

    // setup the pressure function policy
    pressureFunctionPolicy = tChemLib::UseThisTeamPolicy<tChemLib::exec_space>::type(numberCells, Kokkos::AUTO());
    pressureFunctionPolicy.set_scratch_size(1, Kokkos::PerTeam((int)ablate::eos::tChem::Pressure::getWorkSpaceSize(kineticModelGasConstData.nSpec)));

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

PetscErrorCode ablate::finiteVolume::processes::TChemReactions::ChemistryFlowPreStage(TS flowTs, ablate::solver::Solver& flow, PetscReal stagetime) {
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
    solver::Range cellRange;
    flow.GetCellRange(cellRange);

    // get the dim
    PetscInt dim;
    PetscCall(DMGetDimension(flow.GetSubDomain().GetDM(), &dim));

    // store the current dt
    PetscReal dt;
    PetscCall(TSGetTimeStep(flowTs, &dt));

    // get access to the underlying data for the flow
    const auto& flowEulerId = flow.GetSubDomain().GetField("euler").id;
    const auto& flowDensityYiId = flow.GetSubDomain().GetField("densityYi").id;

    // get the flowSolution from the ts
    DM dm = flow.GetSubDomain().GetDM();
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));
    const PetscScalar* flowArray;
    PetscCall(VecGetArrayRead(globFlowVec, &flowArray));

    // Use a parallel for loop to load up the tChem state
    Kokkos::parallel_for(
        "stateLoadHost", Kokkos::RangePolicy<typename tChemLib::host_exec_space>(cellRange.start, cellRange.end), KOKKOS_LAMBDA(const auto i) {
            // get the host data from the petsc field
            const PetscInt cell = cellRange.points ? cellRange.points[i] : i;

            // Get the current state variables for this cell
            const PetscScalar* conserved = nullptr;
            DMPlexPointGlobalRead(dm, cell, flowArray, &conserved) >> checkError;

            // cast the state at i to a state vector
            const auto state_at_i = Kokkos::subview(stateHost, i, Kokkos::ALL());
            Impl::StateVector<real_type_1d_view> stateVector(kineticModelGasConstDataDevice.nSpec, state_at_i);

            // get the current state at I
            auto density = conserved[flowEulerId + ablate::finiteVolume::CompressibleFlowFields::RHO];
            stateVector.Density() = density;
            auto ys = stateVector.MassFractions();
            real_type yiSum = 0.0;
            for (ordinal_type s = 0; s < stateVector.NumSpecies(); s++) {
                ys[s] = PetscMax(0.0, conserved[flowDensityYiId + s] / density);
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
                speedSquare += PetscSqr(conserved[flowEulerId + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
            }

            // compute the internal energy needed to compute temperature
            internalEnergyRefHost[i] = conserved[flowEulerId + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
        });

    // copy from host to device
    Kokkos::deep_copy(internalEnergyRefDevice, internalEnergyRefHost);
    Kokkos::deep_copy(stateDevice, stateHost);

    // Compute temperature into the state field in the device
    ablate::eos::tChem::Temperature::runDeviceBatch(
        temperatureFunctionPolicy, stateDevice, internalEnergyRefDevice, perSpeciesScratchDevice, eos->GetReferenceSpeciesEnthalpy(), kineticModelGasConstDataDevice);

    // Compute the pressure into the state field in the device
    ablate::eos::tChem::Pressure::runDeviceBatch(pressureFunctionPolicy, stateDevice, kineticModelGasConstDataDevice);

    real_type_1d_view timeView("time", stateDevice.size());
    Kokkos::deep_copy(timeView, 0.0);
    real_type_1d_view dtView("delta time", stateDevice.size());
    Kokkos::deep_copy(dtView, 1E-10);

    // assume a constant pressure zero D reaction for each cell
    tChemLib::IgnitionZeroD::runDeviceBatch(
        chemistryFunctionPolicy, tolNewtonDevice, tolTimeDevice, facDevice, timeAdvanceDevice, stateDevice, timeView, dtView, endStateDevice, kineticModelGasConstDataDevices);

    // March over each cell
    //    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    //        // if there is a cell array, use it, otherwise it is just c
    //        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    //
    //        // Get the current state variables for this cell
    //        const PetscScalar* conserved = nullptr;
    //        ierr = DMPlexPointGlobalRead(flow.GetSubDomain().GetDM(), cell, flowArray, &conserved);
    //        CHKERRQ(ierr);
    //
    //        // If a real cell (not ghost)
    //        if (conserved) {
    //            // store the data for the chemistry ts (T, Yi...)
    //            PetscReal temperature;
    //            ierr = temperatureFunction.function(conserved, &temperature, temperatureFunction.context.get());
    //            CHKERRQ(ierr);
    //
    //            // get access to the point ode solver
    //            PetscScalar* pointArray;
    //            ierr = VecGetArray(pointData, &pointArray);
    //            CHKERRQ(ierr);
    //            pointArray[0] = temperature;
    //            for (std::size_t s = 0; s < numberSpecies; s++) {
    //                pointArray[s + 1] = PetscMin(PetscMax(0.0, conserved[flowDensityYiId.offset + s] / conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHO]), 1.0);
    //            }
    //
    //            // precompute some values with the point array
    //            double mwMix = NAN;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    //                                 //            TC_getMs2Wmix(pointArray + 1, (int)numberSpecies, &mwMix);
    //                                 //            TCCHKERRQ(err);
    //
    //            // compute the pressure as this node from T, Yi
    //            double R = 1000.0 * RUNIV / mwMix;
    //            PetscReal pressure = conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHO] * temperature * R;
    //            //            TC_setThermoPres(pressure);
    //
    //            // Compute the total energy sen + hof
    //            PetscReal hof = 0.0;
    //            //            err = eos::TChem::ComputeEnthalpyOfFormation((int)numberSpecies, pointArray, hof);//TODO: restore ComputeEnthalpyOfFormation
    //            //            TCCHKERRQ(err);
    //            PetscReal enerTotal =
    //                hof + conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    //
    //            ierr = VecRestoreArray(pointData, &pointArray);
    //            CHKERRQ(ierr);
    //
    //            // Do a soft reset on the ode solver
    //            ierr = TSSetTime(ts, time);
    //            CHKERRQ(ierr);
    //            ierr = TSSetMaxTime(ts, time + dt);
    //            CHKERRQ(ierr);
    //            ierr = TSSetTimeStep(ts, dtInit);
    //            CHKERRQ(ierr);
    //            ierr = TSSetStepNumber(ts, 0);
    //            CHKERRQ(ierr);
    //
    //            // solver for this point
    //            ierr = TSSolve(ts, pointData);
    //
    //            if (ierr != 0) {
    //                std::string error = "Could not solve chemistry ode, setting source terms to zero T,P (" + std::to_string(temperature) + ", " + std::to_string(pressure) + ") \n (euler, yi): ";
    //                for (PetscInt i = 0; i < dim + 2; i++) {
    //                    error += std::to_string(conserved[flowEulerId.offset + i]) + ", ";
    //                }
    //                for (std::size_t sp = 0; sp < numberSpecies; sp++) {
    //                    error += std::to_string(conserved[flowDensityYiId.offset + sp]) + ", ";
    //                }
    //                std::cout << error << std::endl;
    //
    //                // Use the updated values to compute the source terms for euler and species transport
    //                PetscScalar* fieldSource;
    //                ierr = DMPlexPointLocalRef(fieldDm, cell, sourceArray, &fieldSource);
    //                CHKERRQ(ierr);
    //
    //                fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
    //                fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0.0;
    //                for (PetscInt d = 0; d < dim; d++) {
    //                    fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
    //                }
    //                for (std::size_t sp = 0; sp < numberSpecies; sp++) {
    //                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
    //                    fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + dim + sp] = 0.0;
    //                }
    //
    //                continue;
    //            }
    //
    //            // Use the updated values to compute the source terms for euler and species transport
    //            PetscScalar* fieldSource;
    //            ierr = DMPlexPointLocalRef(fieldDm, cell, sourceArray, &fieldSource);
    //            CHKERRQ(ierr);
    //
    //            // get the array data again
    //            VecGetArray(pointData, &pointArray) >> checkError;
    //
    //            // Use the point array to compute the hof
    //            double updatedHof = 0.0;
    //            //            err = eos::TChem::ComputeEnthalpyOfFormation((int)numberSpecies, pointArray, updatedHof);//TODO: restore
    //            //            TCCHKERRQ(err);
    //            double updatedInternalEnergy = enerTotal - updatedHof;
    //
    //            // store the computed source terms
    //            fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
    //            fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = (conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHO] * updatedInternalEnergy -
    //                                                                               conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHOE]) /
    //                                                                              dt;
    //            for (PetscInt d = 0; d < dim; d++) {
    //                fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
    //            }
    //            for (std::size_t sp = 0; sp < numberSpecies; sp++) {
    //                // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
    //                fieldSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + dim + sp] =
    //                    (conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHO] * PetscMin(1.0, PetscMax(pointArray[sp + 1], 0.0)) - conserved[flowDensityYiId.offset +
    //                    sp]) / dt;
    //            }
    //
    //            VecRestoreArray(pointData, &pointArray);
    //        }
    //    }
    //
    //    // cleanup
    flow.RestoreRange(cellRange);
    //    ierr = VecRestoreArray(sourceVec, &sourceArray);
    //    CHKERRQ(ierr);
    //    ierr = VecRestoreArrayRead(globFlowVec, &flowArray);
    //    CHKERRQ(ierr);
    //
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::TChemReactions::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx) {
    PetscFunctionBegin;
    //    auto process = (ablate::finiteVolume::processes::TChemReactions*)ctx;
    //
    //    // get the cell range
    //    solver::Range cellRange;
    //    solver.GetCellRange(cellRange);
    //
    //    // get the dm for this
    //    PetscDS ds = nullptr;
    //    ierr = DMGetCellDS(dm, cellRange.points ? cellRange.points[cellRange.start] : cellRange.start, &ds);
    //    CHKERRQ(ierr);
    //
    //    // get access to the fArray
    //    PetscScalar* fArray;
    //    ierr = VecGetArray(locFVec, &fArray);
    //    CHKERRQ(ierr);
    //
    //    // hard code assuming only euler and density
    //    PetscInt totDim;
    //    ierr = PetscDSGetTotalDimension(ds, &totDim);
    //    CHKERRQ(ierr);
    //
    //    // get access to the source array in the solver
    //    const PetscScalar* sourceArray;
    //    ierr = VecGetArrayRead(process->sourceVec, &sourceArray);
    //    CHKERRQ(ierr);
    //
    //    // March over each cell
    //    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    //        // if there is a cell array, use it, otherwise it is just c
    //        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    //
    //        // read the global f
    //        PetscScalar* rhs;
    //        ierr = DMPlexPointLocalRef(dm, cell, fArray, &rhs);
    //        CHKERRQ(ierr);
    //
    //        // if a real cell
    //        if (rhs) {
    //            // read the source from the local calc
    //            const PetscScalar* source;
    //            ierr = DMPlexPointLocalRead(process->fieldDm, cell, sourceArray, &source);
    //            CHKERRQ(ierr);
    //
    //            // copy over and add to rhs
    //            for (PetscInt d = 0; d < totDim; d++) {
    //                rhs[d] += source[d];
    //            }
    //            CHKERRQ(ierr);
    //        }
    //    }
    //
    //    solver.RestoreRange(cellRange);
    //    ierr = VecRestoreArray(locFVec, &fArray);
    //    CHKERRQ(ierr);
    //    ierr = VecGetArrayRead(process->sourceVec, &sourceArray);
    //    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::TChemReactions, "reactions using the TChem library", ARG(ablate::eos::EOS, "eos", "the tChem v1 eos"),
         OPT(ablate::parameters::Parameters, "options", "any PETSc options for the chemistry ts"), OPT(std::vector<std::string>, "inertSpecies", "fix the Jacobian for any undetermined inertSpecies"),
         OPT(std::vector<double>, "massFractionBounds", "sets the minimum/maximum mass fraction passed to TChem Library. Must be a vector of size two [min,max] (default is no bounds)"));
