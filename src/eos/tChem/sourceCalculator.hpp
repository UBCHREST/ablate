#ifndef ABLATELIBRARY_TCHEM_SOURCECALCULATOR_HPP
#define ABLATELIBRARY_TCHEM_SOURCECALCULATOR_HPP

#include <TChem_KineticModelGasConstData.hpp>
#include "eos/chemistryModel.hpp"

namespace tChemLib = TChem;

namespace ablate::eos {
class TChem;
}

namespace ablate::eos::tChem {

/**
 * public class to to compute the source for each specified node
 */
class SourceCalculator : public ChemistryModel::SourceCalculator, private utilities::Loggable<SourceCalculator> {
   public:
    //! hold a struct that can be used for chemistry constraints
    struct ChemistryConstraints {
        double dtMin = 1.0E-12;
        double dtMax = 1.0E-1;
        double dtDefault = 1E-4;
        double dtEstimateFactor = 1.5;
        double relToleranceTime = 1.0E-4;
        double absToleranceTime = 1.0E-8;
        double relToleranceNewton = 1.0E-6;
        double absToleranceNewton = 1.0E-10;

        int maxNumNewtonIterations = 100;
        int numTimeIterationsPerInterval = 100000;
        int jacobianInterval = 1;
        int maxAttempts = 4;

        // store an optional threshold temperature.  Only compute the reactions if the temperature is above thresholdTemperature
        double thresholdTemperature = 0.0;

        void Set(const std::shared_ptr<ablate::parameters::Parameters>&);
    };

    /**
     * create a batch source for this size specified in cellRange
     * @param tChemEos
     * @param constraints
     * @param cellRange
     */
    SourceCalculator(const std::vector<domain::Field>& fields, std::shared_ptr<TChem> tChemEos, ChemistryConstraints constraints, const ablate::domain::Range& cellRange);

    /**
     * The compute source can be used as a prestep allowing the add source to be used at each stage without reevaluating
     */
    void ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globalSolution) override;

    /**
     * The compute source can be used as a prestep allowing the add source to be used at each stage without reevaluating
     */
    static void ComputeSource(SourceCalculator& sourceCalculator, const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globalSolution);

    /**
     * Adds the source that was computed in the ComputeSource to the supplied vector
     */
    void AddSource(const ablate::domain::Range& cellRange, Vec localXVec, Vec localFVec) override;

   private:
    //! copy of constraints
    ChemistryConstraints chemistryConstraints;

    /**
     * Hold access to the tchem eos needed to create eos
     */
    std::shared_ptr<eos::TChem> eos;
    const size_t numberSpecies;

    //! the id for the required euler field
    PetscInt eulerId;

    //! the id for the required densityYi field
    PetscInt densityYiId;

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
    real_type_2d_view perSpeciesScratchDevice;

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

}  // namespace ablate::eos::tChem

#endif  // ABLATELIBRARY_BATCHSOURCE_HPP
