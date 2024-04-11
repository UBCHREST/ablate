#include "sourceCalculatorZeroRK.hpp"
#include <algorithm>
#include "eos/zerork.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/stringUtilities.hpp"
#include <math.h>

void ablate::eos::zerorkeos::SourceCalculator::ChemistryConstraints::Set(const std::shared_ptr<ablate::parameters::Parameters>& options) {
    if (options) {
        verbose = options->Get("verbose", verbose);
        timinglog = options->Get("timingLog", timinglog);
        sparseJacobian = options->Get("sparseJacobian", sparseJacobian);
        relTolerance = options->Get("relTolerance", relTolerance);
        absTolerance = options->Get("absTolerance", absTolerance);
        thresholdTemperature = options->Get("thresholdTemperature", thresholdTemperature);
        stepLimiter = options->Get("stepLimiter", stepLimiter);
        loadBalance = options->Get("loadBalance", loadBalance);
        useSEULEX = options->Get("useSEULEX", useSEULEX);
        iterative = options->Get("iterative", iterative);
        gpu = options->Get("gpu", gpu);
        maxiteration = options->Get("maxiteration", maxiteration);
        reactorType = options->Get("reactorType", ReactorType::ConstantVolume);
        errorhandle = options->Get("errorhandle",errorhandle);

#ifdef USE_MPI
    haveMPI=true;
#endif
    }
}

ablate::eos::zerorkeos::SourceCalculator::SourceCalculator(const std::vector<domain::Field>& fields, const std::shared_ptr<zerorkEOS> eosIn,
                                                           ablate::eos::zerorkeos::SourceCalculator::ChemistryConstraints constraints, const ablate::domain::Range& cellRange)
    : chemistryConstraints(constraints), eos(eosIn), numberSpecies(eosIn->GetSpeciesVariables().size()) {
    // determine the number of required cells
    std::size_t numberCells = cellRange.end - cellRange.start;

    // determine the source vector size
    sourceZeroRKAtI = std::vector<double>(numberCells * (eosIn->mech->getNumSpecies() + 1));

    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("ablate::eos::zerorkEOS::BatchSource requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }
    eulerId = eulerField->id;

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("ablate::eos::zerorkEOS::BatchSource requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
    }
    densityYiId = densityYiField->id;

    int zerork_error_state = 0;
    zrm_handle = zerork_reactor_init();
    // load in mechanism for the plugin
    zerork_status_t zerom_status = zerork_reactor_set_mechanism_files(eos->reactionFile.c_str(), eos->thermoFile.c_str(), zrm_handle);
    if (zerom_status != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_status_t status_cvode = zerork_reactor_set_int_option("integrator", chemistryConstraints.useSEULEX, zrm_handle);
    if (status_cvode != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    // verbose 0 is no output, max level is 4
    zerork_status_t status_verbose = zerork_reactor_set_int_option("verbosity", chemistryConstraints.verbose, zrm_handle);
    if (status_verbose != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    // set to  0 to turn it off
    zerork_status_t status_loadbalance = zerork_reactor_set_int_option("load_balance", chemistryConstraints.loadBalance, zrm_handle);
    if (status_loadbalance != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    // Set tolerances
    zerork_status_t status_abstol = zerork_reactor_set_double_option("abs_tol", chemistryConstraints.absTolerance, zrm_handle);
    if (status_abstol != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;
    zerork_status_t status_reltol = zerork_reactor_set_double_option("rel_tol", chemistryConstraints.relTolerance, zrm_handle);
    if (status_reltol != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    switch (chemistryConstraints.reactorType) {
        case ReactorType::ConstantVolume: {
            zerork_status_t status_constvolume = zerork_reactor_set_int_option("constant volume", 1, zrm_handle);
            if (status_constvolume != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;
            break;
        }
        case ReactorType::ConstantPressure: {
            zerork_status_t status_constpress = zerork_reactor_set_int_option("constant_volume", 0, zrm_handle);
            if (status_constpress != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;
            break;
        }
    }
    zerork_status_t status_alwaysSolveTemp = zerork_reactor_set_int_option("always_solve_temperature", 1, zrm_handle);
    if (status_alwaysSolveTemp != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    // Kinetic rate limiter
    zerork_status_t status_steplimiter = zerork_reactor_set_double_option("step_limiter", chemistryConstraints.stepLimiter, zrm_handle);
    if (status_steplimiter != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    // Kinetic rate limiter
    zerork_status_t status_gpu = zerork_reactor_set_int_option("gpu", chemistryConstraints.gpu, zrm_handle);
    if (status_gpu != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    if (chemistryConstraints.timinglog) {
        zerork_reactor_set_string_option("reactor_timing_log_filename", "timing.log", zrm_handle);
    }

    // Use sparse matrix math for the jacobian
    if (!chemistryConstraints.sparseJacobian) {
        zerork_status_t status_sparse = zerork_reactor_set_int_option("dense", 1, zrm_handle);
        if (status_sparse != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;
    }
    zerork_status_t status_iterative = zerork_reactor_set_int_option("iterative", chemistryConstraints.iterative, zrm_handle);
    if (status_iterative != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_status_t status_maxsteps = zerork_reactor_set_int_option("max_steps", 10, zrm_handle);
    if (status_maxsteps != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

//    zerork_status_t dump = zerork_reactor_set_int_option("dump_reactors", 1, zrm_handle);
//    if (dump != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_status_t status_mech = zerork_reactor_load_mechanism(zrm_handle);  // make sure this call is after gpu setup
    if (status_mech != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    if (zerork_error_state != 0) {
        throw std::invalid_argument("ablate::eos::zerork couldnt initialize, something is wrong...");
    }



}

void ablate::eos::zerorkeos::SourceCalculator::ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globFlowVec) {
    StartEvent("zerorkEOS::SourceCalculator::ComputeSource");
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

    // zerork state load up
    int nSpc = eos->mech->getNumSpecies();  // Number of Species
    int nState = nSpc + 1;
    int nReactors = numberCells;

    // Set up reactor initial states
    std::vector<double> reactorT(nReactors);
    std::vector<double> reactorP(nReactors);
    std::vector<double> density2(nReactors);
    std::vector<double> sensibleenergy(nReactors);
    std::vector<double> velmag2(nReactors);
    std::vector<double> reactorMassFrac(nReactors * nSpc);
    std::vector<double> enthalpyOfFormation(nSpc);
    std::vector<int> reactorEval(nReactors, 0);

    // Set up the vectors that are actually evaluated with the temperature threshold
    std::vector<double> reactorTEval(nReactors);
    std::vector<double> reactorPEval(nReactors);
    std::vector<double> reactorMassFracEval(nReactors * nSpc);

    // get the current state from petsc
    int p = 0;
    for (int i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
        const std::size_t k = i - cellRange.start;

        const PetscScalar* eulerField = nullptr;
        DMPlexPointLocalFieldRead(solutionDm, cell, eulerId, flowArray, &eulerField) >> utilities::PetscUtilities::checkError;
        const PetscScalar* flowDensityField = nullptr;
        DMPlexPointLocalFieldRead(solutionDm, cell, densityYiId, flowArray, &flowDensityField) >> utilities::PetscUtilities::checkError;

        // get the current state at I
        auto density = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHO];
        density2[k] = density;

        double yiSum = 0.0;
        for (int s = 0; s < nSpc - 1; s++) {
            reactorMassFrac[k * nSpc + s] = PetscMax(0.0, flowDensityField[s] / density);
            reactorMassFrac[k * nSpc + s] = PetscMin(1.0, reactorMassFrac[k * nSpc + s]);
            yiSum += reactorMassFrac[k * nSpc + s];
        }
        if (yiSum > 1.0) {
            for (PetscInt s = 0; s < nSpc - 1; s++) {
                // Limit the bounds
                reactorMassFrac[k * nSpc + s] /= yiSum;
            }
            reactorMassFrac[k * nSpc + nSpc - 1] = 0.0;
        } else {
            reactorMassFrac[k * nSpc + nSpc - 1] = 1.0 - yiSum;
        }

        // Compute the internal energy from total energy
        PetscReal speedSquare = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            speedSquare += PetscSqr(eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
        }

        // compute the internal energy needed to compute temperature
        sensibleenergy[k] = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;

        double enthalpymix = eos->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[k * nSpc]);

        sensibleenergy[k] += enthalpymix;

        reactorT[k] = eos->mech->getTemperatureFromEY(sensibleenergy[k], &reactorMassFrac[k * nSpc], 2000);
        reactorP[k] = eos->mech->getPressureFromTVY(reactorT[k], 1 / density, &reactorMassFrac[k * nSpc]);

        // Set up the vector that is actually being solved
        if (reactorT[k] > chemistryConstraints.thresholdTemperature) {
            reactorEval[k] = 1;
            reactorTEval[p] = reactorT[k];
            reactorPEval[p] = reactorP[k];
            for (int s = 0; s < nSpc; s++) {
                reactorMassFracEval[p * nSpc + s] = reactorMassFrac[k * nSpc + s];
            }
            p += 1;
        }
    }

    if(chemistryConstraints.haveMPI){
        std::cout << "STUFF    STUFF";
    }

    bool calcSourceterms=false;


    // mass fraction before the reactor
    std::vector<double> ys = reactorMassFracEval;

    if (chemistryConstraints.errorhandle==0) {
        // This case try to integrate if it fails, throw an error.
        try {
            auto nReactorsEval = std::reduce(reactorEval.begin(), reactorEval.end());
            zerork_status_t flag = ZERORK_STATUS_SUCCESS;
            flag = zerork_reactor_solve(1, time, dt, nReactorsEval, &reactorTEval[0], &reactorPEval[0], &reactorMassFracEval[0], zrm_handle);

            if (flag != ZERORK_STATUS_SUCCESS) {
                throw std::runtime_error("ablate::eos::zerorkEOS::Computesource zerork couldn't integrate the chemistry, considertightening the tolerances or restart with errorhandle: 2.");
            }
        }catch(const runtime_error& e){
            exit(1);
        }

    }else if(chemistryConstraints.errorhandle==1){
        //In this case try to integrate and if it fails create 0 source terms

        auto nReactorsEval = std::reduce(reactorEval.begin(), reactorEval.end());
        zerork_status_t flag = ZERORK_STATUS_SUCCESS;
        flag = zerork_reactor_solve(1, time, dt, nReactorsEval, &reactorTEval[0], &reactorPEval[0], &reactorMassFracEval[0], zrm_handle);

        if (flag != ZERORK_STATUS_SUCCESS) {
            std::cout << "Integration failed 0 source terms are used for rank "<< rank << "\n";
        }
        else{
            std::cout << "Integration successful for rank "<< rank << "\n";
        }

    }else if(chemistryConstraints.errorhandle==2){
        //In this case try to integrate and if you fail reduce the tolerances and integrate again on all ranks...
        calcSourceterms=true;

        //back up the state
        std::vector<double> Tbackup=reactorTEval;
        std::vector<double> Pbackup=reactorPEval;
        std::vector<double> Ybackup=reactorMassFracEval;

        // Solve the reactors
        auto nReactorsEval = std::reduce(reactorEval.begin(), reactorEval.end());
        zerork_status_t flag = ZERORK_STATUS_SUCCESS;
        flag = zerork_reactor_solve(1, time, dt, nReactorsEval, &reactorTEval[0], &reactorPEval[0], &reactorMassFracEval[0], zrm_handle);

        // Determine if it was successful everywhere
        int nranks_ = 1;
        MPI_Comm_size(MPI_COMM_WORLD,&(nranks_));
        int zerorklocalout = 0;
        std::vector<int> zerorkallout(nranks_,0);

        if (flag != ZERORK_STATUS_SUCCESS) {
            zerorklocalout = -1;
        }

        //We need to communicate to every rank to redo chemistry
        MPI_Allgather(&zerorklocalout,1,MPI_INT,
                      &zerorkallout[0],1,MPI_INT,MPI_COMM_WORLD);

        auto min = std::min_element(zerorkallout.begin(), zerorkallout.end());
        cout << "Minimum Element is:" << *min;
        std::min_element(std::begin(zerorkallout), std::end(zerorkallout));

        if (flag != ZERORK_STATUS_SUCCESS) {
            std::stringstream warningMessage;
            warningMessage << "Warning: Could not integrate chemistry on rank " << rank << "\n";
            warningMessage << "Current time is: " << std::setprecision(16) << time << "\n";
            warningMessage << "Requested timestep: " << std::setprecision(16) << dt << "\n";
            std::cout << warningMessage.str() << std::endl;

            int ii = 0;
            // Retrying the step
            while (flag != ZERORK_STATUS_SUCCESS) {
                ++ii;
                std::cout << "Tightening tolerances."
                          << "\n";
                zerork_reactor_set_double_option("abs_tol", chemistryConstraints.absTolerance * pow(0.01, ii), zrm_handle);
                zerork_reactor_set_double_option("rel_tol", chemistryConstraints.relTolerance * pow(0.5, ii), zrm_handle);
                flag = zerork_reactor_solve(2, time, dt, nReactorsEval, &Tbackup[0], &Pbackup[0], &Ybackup[0], zrm_handle);
                // If tigethening the
                if (ii == 5) {
                    break;
                }
            }
            if (flag != ZERORK_STATUS_SUCCESS) {
                std::cout << "Warning: Could not integrate chemistry after reducing the tolerances multiple times."
                          << "\n";
                std::cout << "Consider reducing the timestep and using steplimiter option "
                          << "\n";
                throw std::runtime_error("ablate::eos::zerorkEOS::Computesource zerork couldn't integrate the simulation.");
            }

            // reset original tolerances
            zerork_reactor_set_double_option("abs_tol", chemistryConstraints.absTolerance, zrm_handle);
            zerork_reactor_set_double_option("rel_tol", chemistryConstraints.relTolerance, zrm_handle);

            // Set the reactors to the new values
            reactorTEval = Tbackup;
            reactorPEval = Pbackup;
            reactorMassFracEval = Ybackup;
        }
    }



    std::cout << "start calculating sourceterms "<< rank << "\n";
    // Set all the sources to 0
    sourceZeroRKAtI.assign(nState * nReactors, 0);
    if (calcSourceterms) {
        for (int s = 0; s < nSpc - 1; s++) {
            std::vector<double> tempvec(nSpc, 0.);
            tempvec[s] = 1;
            enthalpyOfFormation[s] = eos->mech->getMassEnthalpyFromTY(298.15, &tempvec[0]);
        }

        //    // Recompute density for constant pressure reactors
        //    // Should be done but doesn't work with current coupling
        //    if(chemistryConstraints.reactorType==ReactorType::ConstantPressure){
        //        int q=0;
        //        for (int i=0;i<nReactors;i++){
        //            if (reactorEval[i]==1){
        //                density2[i]=eos->mech->getDensityFromTPY(reactorTEval[q], reactorPEval[q],&reactorMassFracEval[q]);
        //                q+=1;
        //            }
        //        }
        //    }

        int q = 0;
        for (int i = 0; i < nReactors; ++i) {
            if (reactorEval[i] != 1) {
                for (int j = 0; j < nState; ++j) {
                    // Set sourceterms to 0 if the reactor was evaluated
                    sourceZeroRKAtI[i * nState + j] = 0;
                }
                q += 1;
            } else {
                for (int s = 0; s < nSpc; s++) {
                    sourceZeroRKAtI[i * nState] += (ys[(i - q) * nSpc + s] - reactorMassFracEval[(i - q) * nSpc + s]) * enthalpyOfFormation[s];
                }

                for (int s = 0; s < nSpc; ++s) {
                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                    sourceZeroRKAtI[i * nState + s + 1] = reactorMassFracEval[(i - q) * nSpc + s] - ys[(i - q) * nSpc + s];
                }

                // Now scale everything by density/dt
                for (int j = 0; j < nState; ++j) {
                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                    sourceZeroRKAtI[i * nState + j] *= density2[i] / dt;
                }
            }
        }
    }
    EndEvent();
    std::cout << "finish calculating sourceterms "<< rank << "\n";
}
void ablate::eos::zerorkeos::SourceCalculator::AddSource(const ablate::domain::Range& cellRange, Vec, Vec locFVec) {
    StartEvent("zerorkEOS::SourceCalculator::AddSource");
    std::cout << "start adding sourceterms " << "\n";
    // get access to the fArray
    PetscScalar* fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

    // Get the solution dm
    DM dm;
    VecGetDM(locFVec, &dm) >> utilities::PetscUtilities::checkError;

    int nSpc = eos->mech->getNumSpecies();
    int nState = nSpc + 1;
    for (int i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt cell = cellRange.points ? cellRange.points[i] : i;
        const std::size_t k = i - cellRange.start;

        // Get the current state variables for this cell
        PetscScalar* eulerSource = nullptr;
        DMPlexPointLocalFieldRef(dm, cell, eulerId, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
        PetscScalar* densityYiSource = nullptr;
        DMPlexPointLocalFieldRef(dm, cell, densityYiId, fArray, &densityYiSource) >> utilities::PetscUtilities::checkError;

        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += sourceZeroRKAtI[k * nState];
        for (std::size_t sp = 0; sp < numberSpecies; sp++) {
            densityYiSource[sp] += sourceZeroRKAtI[k * nState + sp + 1];
        }
    }

    // cleanup
    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    EndEvent();
}

std::ostream& ablate::eos::zerorkeos::operator<<(std::ostream& os, const ablate::eos::zerorkeos::SourceCalculator::ReactorType& v) {
    switch (v) {
        case ablate::eos::zerorkeos::SourceCalculator::ReactorType::ConstantPressure:
            return os << "ConstantPressure";
        case ablate::eos::zerorkeos::SourceCalculator::ReactorType::ConstantVolume:
            return os << "ConstantVolume";
        default:
            return os;
    }
}

std::istream& ablate::eos::zerorkeos::operator>>(std::istream& is, ablate::eos::zerorkeos::SourceCalculator::ReactorType& v) {
    std::string enumString;
    is >> enumString;

    // make the comparisons easier to converting to lower
    ablate::utilities::StringUtilities::ToLower(enumString);

    if (enumString == "constantvolume") {
        v = ablate::eos::zerorkeos::SourceCalculator::ReactorType::ConstantVolume;
    } else if (enumString == "constantpressure") {
        // default to constant pressure
        v = ablate::eos::zerorkeos::SourceCalculator::ReactorType::ConstantPressure;
    } else {
        throw std::invalid_argument(
            " Unknown reactor type set. \n"
            " Acceptable reactor types: ConstantPressure, ConstantVolume. \n"
            " Default is Contstant volume");
    }
    return is;
}
