#ifndef ABLATELIBRARY_TCHEM_SOURCECALCULATORSOOT_HPP
#define ABLATELIBRARY_TCHEM_SOURCECALCULATORSOOT_HPP

#include <TChem_KineticModelGasConstData.hpp>
#include "eos/chemistryModel.hpp"
#include "eos/tChem/sourceCalculator.hpp"

namespace tChemLib = TChem;

namespace ablate::eos {
class TChemSoot;
}

namespace ablate::eos::tChemSoot {

/**
 * public class to to compute the source for each specified node
 */
class SourceCalculatorSoot : public ChemistryModel::SourceCalculator, private utilities::Loggable<SourceCalculatorSoot> {
   public:
    /**
     * create a batch source for this size specified in cellRange
     * @param tChemEos
     * @param constraints
     * @param cellRange
     */
    SourceCalculatorSoot(const std::vector<domain::Field>& fields, const std::shared_ptr<TChemSoot>& tChemEos, ablate::eos::tChem::SourceCalculator::ChemistryConstraints constraints,
                         const solver::Range& cellRange);

    /**
     * The compute source can be used as a prestep allowing the add source to be used at each stage without reevaluating
     */
    void ComputeSource(const solver::Range& cellRange, PetscReal time, PetscReal dt, Vec globalSolution) override;

    /**
     * Adds the source that was computed in the ComputeSource to the supplied vector
     */
    void AddSource(const solver::Range& cellRange, Vec localXVec, Vec localFVec) override;

   private:
    //! copy of constraints
    ablate::eos::tChem::SourceCalculator::ChemistryConstraints chemistryConstraints;

    /**
     * Hold access to the tchem eos needed to create eos
     */
    std::shared_ptr<eos::TChemSoot> eos;
    const size_t numberSpecies;

    //! the id for the required euler field
    PetscInt eulerId;

    //! the id for the required densityYi field
    PetscInt densityYiId;

    //! the id for the progress variable field
    PetscInt densityProgressId;
    PetscInt sootNumberDensityIndex;

    // tchem memory storage on host/device.  These will be sized for the number of active nodes in the domain
    real_type_2d_view stateDevice;
    real_type_2d_view_host stateHost;

    // store the end state for the device/host
    real_type_2d_view endStateDevice;

    // the time advance information
    time_advance_type_1d_view timeAdvanceDevice;
    time_advance_type timeAdvanceDefault{};

    // store host/device memory for computing state
    real_type_1d_view internalEnergyRefDevice;
    real_type_1d_view_host internalEnergyRefHost;
    real_type_1d_view totInternalEnergyRefDevice;
    real_type_1d_view_host totInternalEnergyRefHost;
    real_type_2d_view perSpeciesScratchDevice;
    real_type_2d_view perGasSpeciesScratchDevice;

    // store the source terms (density* energy + density*species)
    real_type_2d_view_host sourceTermsHost;
    real_type_2d_view sourceTermsDevice;

    // tolerance constraints
    real_type_2d_view tolTimeDevice;
    real_type_1d_view tolNewtonDevice;
    real_type_2d_view facDevice;

    // store the time and delta for the ode solver
    real_type_1d_view timeViewDevice;
    real_type_1d_view dtViewDevice;

    // store device specific kineticModelGasConstants
    tChemLib::KineticModelConstData<typename Tines::UseThisDevice<exec_space>::type> kineticModelGasConstDataDevice;
    kmd_type_1d_view_host kineticModelDataClone;
    Kokkos::View<KineticModelGasConstData<typename Tines::UseThisDevice<exec_space>::type>*, typename Tines::UseThisDevice<exec_space>::type> kineticModelGasConstDataDevices;
};

}  // namespace ablate::eos::tChemSoot

#endif  // ABLATELIBRARY_BATCHSOURCE_HPP
