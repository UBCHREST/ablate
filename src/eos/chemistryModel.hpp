#ifndef ABLATELIBRARY_CHEMISTRYMODEL_HPP
#define ABLATELIBRARY_CHEMISTRYMODEL_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "eos/eos.hpp"
#include "solver/cellSolver.hpp"
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
    explicit ChemistryModel(std::string name) : eos::EOS(std::move(name)){};

    /**
     * The batch source interface can be used so solve multiple nodes simultaneously.
     * The batch interface is divided into two processes
     */
    class SourceCalculator {
       public:
        virtual ~SourceCalculator() = default;
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

    /**
     * Optional function to get a solution update
     */
    virtual std::vector<std::tuple<ablate::solver::CellSolver::SolutionFieldUpdateFunction, void*, std::vector<std::string>>> GetSolutionFieldUpdates() { return {}; }

    virtual inline double GetEnthalpyOfFormation(std::string_view speciesName) const {return {};};

    [[nodiscard]] virtual std::map<std::string, double> GetSpeciesMolecularMass() const = 0;

    [[nodiscard]] virtual std::map<std::string, double> GetElementInformation() const = 0;

    [[nodiscard]] virtual std::map<std::string, std::map<std::string, int>> GetSpeciesElementalInformation() const = 0;

    /**
     * a temperature thermodynamic function specific to
     */
    struct ThermodynamicTemperatureMassFractionFunction {
        //! function to be called
        PetscErrorCode (*function)(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx) = nullptr;
        //! optional context to pass into the function
        std::shared_ptr<void> context = nullptr;
        //! the property size being set
        PetscInt propertySize = 1;
    };

    [[nodiscard]] virtual ThermodynamicTemperatureMassFractionFunction GetThermodynamicTemperatureMassFractionFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const {return {};};
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_CHEMISTRYMODEL_HPP
