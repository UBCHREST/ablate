#ifndef ABLATELIBRARY_HDF5CHEMTABINITIALIZER_HPP
#define ABLATELIBRARY_HDF5CHEMTABINITIALIZER_HPP

#include "domain/hdf5Initializer.hpp"
#include "eos/chemTab.hpp"

namespace ablate::eos::chemTab {

/**
 * Initializes the domain (assuming ChemTab) using a previous result stored in an hdf5 file.
 * The previous result should be run with standard chemistry (TChem).  The new domain should
 * be setup with ChemTab/progress variables.
 */
class Hdf5ChemTabInitializer : public domain::Hdf5Initializer {
   private:
    // keep a pointer to the chemTab eos
    std::shared_ptr<ablate::eos::ChemTab> chemTab;

   public:
    /**
     * Create the Hdf5ChemTabInitializer
     */
    explicit Hdf5ChemTabInitializer(std::filesystem::path hdf5Path, std::shared_ptr<ablate::eos::EOS> chemTab, std::shared_ptr<ablate::domain::Region> region = {});

    /**
     * Allow overriding of the GetFieldFunctions so that a chemTAb mapping function can be used
     */
    [[nodiscard]] std::vector<std::shared_ptr<mathFunctions::FieldFunction>> GetFieldFunctions(const std::vector<domain::Field>& fields) const override;

   private:
    /**
     * Private helper class to map from yis to progress variables
     */
    class Hdf5ChemTabMappingMathFunction : public Hdf5MathFunction {
       private:
        // store a reference to the chemTab eos
        const ablate::eos::ChemTab& chemTab;

        // Keep a Hdf5MathFunction so we can interpolate density
        std::shared_ptr<Hdf5MathFunction> eulerFunction = nullptr;

        // store the number of species
        PetscInt numberOfSpecies;

       public:
        /**
         * Load the hdf5 vector for Yi
         * @param baseMesh
         */
        Hdf5ChemTabMappingMathFunction(const std::shared_ptr<Hdf5Mesh>& baseMesh, const ablate::eos::ChemTab&);

       protected:
        /**
         * Private method that does the interpolation for the provided point.  Override this to allow interpolating progress variables
         * @param dim
         * @param x
         * @param Nf
         * @param u
         * @return
         */
        PetscErrorCode Eval(PetscInt dim, const PetscReal x[], PetscScalar* u) const override;
    };
};

}  // namespace ablate::eos::chemTab
#endif  // ABLATELIBRARY_HDF5CHEMTABINITIALIZER_HPP
