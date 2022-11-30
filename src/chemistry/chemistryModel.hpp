#ifndef ABLATELIBRARY_CHEMISTRYMODEL_HPP
#define ABLATELIBRARY_CHEMISTRYMODEL_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "eos/eos.hpp"

namespace ablate::chemistry {

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
     * Single function to produce ChemistryFunction function based upon the available fields and sources.  This single point function is useful for unit level testing.
     * @param fields in the conserved/source arrays
     * @param property
     * @param fields
     * @return
     */
    virtual void ChemistrySource(const std::vector<domain::Field>& fields, const PetscReal conserved[], PetscReal* source) const = 0;
};
}  // namespace ablate::chemistry

#endif  // ABLATELIBRARY_CHEMISTRYMODEL_HPP
