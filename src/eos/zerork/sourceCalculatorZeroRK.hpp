#ifndef ABLATELIBRARY_ZERORK_SOURCECALCULATOR_HPP
#define ABLATELIBRARY_ZERORK_SOURCECALCULATOR_HPP

#include "eos/chemistryModel.hpp"
#include "zerork_cfd_plugin.h"
#include "zerork/mechanism.h"
#include "zerork/utilities.h"

#include "zerork_cfd_plugin.h"

namespace ablate::eos {
class zerorkEOS;
}

namespace ablate::eos::zerorkeos {

/**
 * public class to to compute the source for each specified node
 */
class SourceCalculator : public ChemistryModel::SourceCalculator, private utilities::Loggable<SourceCalculator> {
   public:
    /**
     * Allow the user of TChem to set the reactor type
     */
    enum class ReactorType { ConstantPressure, ConstantVolume };

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

        // store the reactor type in the chemistry constrains
        ReactorType reactorType = ReactorType::ConstantPressure;

        // store an optional threshold temperature.  Only compute the reactions if the temperature is above thresholdTemperature
        double thresholdTemperature = 0.0;

//        void Set(const std::shared_ptr<ablate::parameters::Parameters>&);
    };
    /**
     * create a batch source for this size specified in cellRange
     * @param zerorkEos
     * @param constraints
     * @param cellRange
     */
    SourceCalculator(const std::vector<domain::Field>& fields, std::shared_ptr<zerorkEOS> zerorkEos, ablate::eos::zerorkeos::SourceCalculator::ChemistryConstraints constraints, const ablate::domain::Range& cellRange);

    /**
     * The compute source can be used as a prestep allowing the add source to be used at each stage without reevaluating
     */
    void ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globalSolution) override;

    /**
     * The compute source can be used as a prestep allowing the add source to be used at each stage without reevaluating
     */
    //    static void ComputeSource(SourceCalculator& sourceCalculator, const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec globalSolution);

    /**
     * Adds the source that was computed in the ComputeSource to the supplied vector
     */
    void AddSource(const ablate::domain::Range& cellRange, Vec localXVec, Vec localFVec) override;

   private:
    std::vector<double> sourceZeroRKAtI;
    zerork_handle zrm_handle;
    //! copy of constraints
    ablate::eos::zerorkeos::SourceCalculator::ChemistryConstraints chemistryConstraints;
    /**
     * Hold access to the tchem eos needed to create eos
     */
    std::shared_ptr<eos::zerorkEOS> eos;

    const size_t numberSpecies;

    //! the id for the required euler field
    PetscInt eulerId;

    //! the id for the required densityYi field
    PetscInt densityYiId;



};

/**
 * Support function for the TChemBase::ReactorType Enum
 * @param os
 * @param v
 * @return
 */
std::ostream& operator<<(std::ostream& os, const SourceCalculator::ReactorType& v);

/**
 * Support function for the TChemBase::ReactorType Enum
 * @param os
 * @param v
 * @return
 */
std::istream& operator>>(std::istream& is, SourceCalculator::ReactorType& v);

}  // namespace

#endif  // ABLATELIBRARY_BATCHSOURCE_HPP
