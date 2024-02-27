#include "sourceCalculatorZeroRK.hpp"
#include <algorithm>
#include "eos/zerork.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/stringUtilities.hpp"




ablate::eos::zerorkeos::SourceCalculator::SourceCalculator(const std::vector<domain::Field>& fields, const std::shared_ptr<zerorkEOS> eosIn,
                                                          ablate::eos::zerorkeos::SourceCalculator::ChemistryConstraints constraints, const ablate::domain::Range& cellRange)
    : chemistryConstraints(constraints), eos(eosIn), numberSpecies(eosIn->GetSpeciesVariables().size()) {
    // determine the number of required cells
    std::size_t numberCells = cellRange.end - cellRange.start;

    // determine the source vector size
    sourceZeroRKAtI = std::vector<double> (numberCells * (eosIn->mech->getNumSpecies()));

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
    // load in mechanism for the plugin
    zerork_status_t zerom_status = zerork_reactor_set_mechanism_files(eos->reactionFile.c_str(), eos->thermoFile.c_str(), zrm_handle);
//    zerork_status_t zerom_status = zerork_reactor_set_mechanism_files(eos->mech->mechFileStr.c_str(), eos->mech->thermFileStr, zrm_handle);
    if(zerom_status != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_status_t status_mech = zerork_reactor_load_mechanism(zrm_handle);
    if(status_mech != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_status_t status_other = zerork_reactor_set_int_option("constant_volume", 0, zrm_handle);
    if(status_other != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_status_t status_other2 = zerork_reactor_set_int_option("verbosity", 0, zrm_handle);
    if(status_other != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    zerork_reactor_set_int_option("always_solve_temperature", 1, zrm_handle);
    if(status_other != ZERORK_STATUS_SUCCESS) zerork_error_state += 1;

    if (zerork_error_state!=0) {
        throw std::invalid_argument("ablate::eos::zerork couldnt initialize, something is wrong...");
    }

}

void ablate::eos::zerorkeos::SourceCalculator::ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globFlowVec) {
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

    //zerork state load up
    int nSpc=eos->mech->getNumSpecies(); //Number of Species
    int nState=nSpc+1;

    int nReactors = numberCells;

    //Set up reactor initial states
    int nReactorsAlloc = nReactors;
    std::vector<double> reactorT(nReactors);
    std::vector<double> reactorP(nReactors);
    std::vector<double> density2(nReactors);
    std::vector<double> sensibleenergy(nReactors);
    std::vector<double> velmag2(nReactors);
    std::vector<double> reactorMassFrac(nReactors*nSpc);
    std::vector<double> enthapyOfFormation(nSpc);

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
        density2[k]=density;

        double yiSum = 0.0;
        for (int s = 0; s < nSpc - 1; s++) {
            reactorMassFrac[k + s] = PetscMax(0.0, flowDensityField[k + s] / density);
            reactorMassFrac[k + s] = PetscMin(1.0, reactorMassFrac[k + s]);
            yiSum += reactorMassFrac[k + s];
        }
        if (yiSum > 1.0) {
            for (PetscInt s = 0; s < nSpc - 1; s++) {
                // Limit the bounds
                reactorMassFrac[k + s] /= yiSum;
            }
    //        reactorMassFrac[nSpc - 1] = 0.0;
        } else {
            reactorMassFrac[k + nSpc - 1] = 1.0 - yiSum;
        }

        // Compute the internal energy from total energy
        PetscReal speedSquare = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            speedSquare += PetscSqr(eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
        }

        // compute the internal energy needed to compute temperature
        sensibleenergy[k]= eulerField[ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;

        std::vector<double> enthalpyOfFormationLocal(nSpc,0.);
        double enthalpymix = eos->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[k*nSpc]);

        sensibleenergy[k] += enthalpymix;

        reactorT[k] = eos->mech->getTemperatureFromEY(sensibleenergy[k], &reactorMassFrac[k*nSpc], 2000);

        // TODO change this so it is the right pressure!!!!
        reactorP[k] = 101325;
    }

    //mass fraction before the reactor
    std::vector<double> ys = reactorMassFrac;

    zerork_status_t flag = ZERORK_STATUS_SUCCESS;
    flag = zerork_reactor_solve(1, time, dt, nReactors, &reactorT[0], &reactorP[0],
                                &reactorMassFrac[0], zrm_handle);

    if(flag != ZERORK_STATUS_SUCCESS) printf("Oo something went wrong during zreork integration...");
    // TODO try to print the state for debugging if the integration fails

        std::cout << "zerork time is:" << time <<"\n";
        std::cout << "zerork temperature is:" << reactorT[0] <<"\n";
        std::cout << "zerork Yox is:" << reactorMassFrac[1]<<"\n";
        std::cout << "zerork Yfuel is:" << reactorMassFrac[63]<<"\n";

    // TODO make sure the the whole vector is 0;
    sourceZeroRKAtI[0] = 0.0;

    for (int s = 0; s < nSpc - 1; s++) {
        std::vector<double> tempvec(nSpc,0.);
        tempvec[s]=1;
        //        double* a = &tempvec[0];
        enthapyOfFormation[s] = eos->mech->getMassEnthalpyFromTY(298.15, &tempvec[0]);
    }


    for(int k=0; k<nReactors; ++k) {

        for (int s = 0; s < nSpc; s++) {
            //            sourceZeroRKAtI[k] += (ys2[k * nSpc + s] - reactorMassFrac[k * nSpc + s]) * enthalpyOfFormationLocal[k + s];
            sourceZeroRKAtI[k * nSpc] += (ys[k * nSpc + s] - reactorMassFrac[k * nSpc + s]) * enthapyOfFormation[s];
        }

        for (int s = 0; s < nSpc; ++s) {
            // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
            sourceZeroRKAtI[k * nSpc + s + 1] = reactorMassFrac[k * nSpc + s] - ys[k * nSpc + s];
        }

        // Now scale everything by density/dt
        for (int j = 0; j < nState; ++j) {
            // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
            sourceZeroRKAtI[k * nSpc + j] *= density2[k] / dt;
        }
    }
    EndEvent();
}
void ablate::eos::zerorkeos::SourceCalculator::AddSource(const ablate::domain::Range& cellRange, Vec, Vec locFVec) {
    StartEvent("zerork::SourceCalculator::AddSource");
    // get access to the fArray
    PetscScalar* fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

    // Get the solution dm
    DM dm;
    VecGetDM(locFVec, &dm) >> utilities::PetscUtilities::checkError;
    auto numberCells = cellRange.end - cellRange.start;

    for (int k = 0 ; k<numberCells;k++) {
        const PetscInt cell = cellRange.points ? cellRange.points[k] : k;
        const std::size_t chemIndex = k - cellRange.start;

        // Get the current state variables for this cell
        PetscScalar* eulerSource = nullptr;
        DMPlexPointLocalFieldRef(dm, cell, eulerId, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
        PetscScalar* densityYiSource = nullptr;
        DMPlexPointLocalFieldRef(dm, cell, densityYiId, fArray, &densityYiSource) >> utilities::PetscUtilities::checkError;


        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += sourceZeroRKAtI[k * numberSpecies];
        for (std::size_t sp = 0; sp < numberSpecies; sp++) {
            densityYiSource[sp] += sourceZeroRKAtI[k * numberSpecies + sp + 1];
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
    } else {
        // default to constant pressure
        v = ablate::eos::zerorkeos::SourceCalculator::ReactorType::ConstantPressure;
    }
    return is;
}