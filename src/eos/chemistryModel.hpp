#ifndef ABLATELIBRARY_CHEMISTRYMODEL_HPP
#define ABLATELIBRARY_CHEMISTRYMODEL_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <vector>
#include "eos/eos.hpp"
#include "solver/solver.hpp"

namespace ablate::eos {

/**
 * The ChemistryModel is an extension of the equation of state.  All ChemistryModels support computing source terms based upon conserved variables
 */
class ChemistryModel : public eos::EOS {
   public:
    /**
     * provide constructor to eos
     * @param name
     */
    explicit ChemistryModel(std::string name) : eos::EOS(name){};

    /**
     * The batch source interface can be used so solve multiple nodes simultaneously.
     * The batch interface is divided into two processes
     */
    class SourceCalculator {
       public:
        virtual ~SourceCalculator(){};
        /**
         * The compute source can be used as a prestep allowing the add source to be used at each stage without reevaluating
         */
        virtual void ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec solution) = 0;

        /**
         * Adds the source that was computed in the presetp to the supplied vector
         */
        virtual void AddSource(const ablate::domain::Range& cellRange, Vec solution, Vec source) = 0;
    };

    /**
     * Function to create the batch source specific to the provided cell range
     * @param cellRange
     * @return
     */
    virtual std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const ablate::domain::Range& cellRange) = 0;
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_CHEMISTRYMODEL_HPP
