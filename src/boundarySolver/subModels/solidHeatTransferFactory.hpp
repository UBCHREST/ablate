#ifndef ABLATELIBRARY_SOLIDHEATTRANSFERFACTORY_HPP
#define ABLATELIBRARY_SOLIDHEATTRANSFERFACTORY_HPP

#include <memory>
#include "solidHeatTransfer.hpp"

namespace ablate::boundarySolver::subModels {

/**
 * A factory to create solid heat transfer instances
 */
class SolidHeatTransferFactory {
   private:
    // the heat transfer properties
    const std::shared_ptr<ablate::parameters::Parameters> properties;

    //  math function to initialize the temperature
    const std::shared_ptr<ablate::mathFunctions::MathFunction> initialization;

    // the petsc options for the solver/ts
    const std::shared_ptr<ablate::parameters::Parameters> options;

   public:
    /**
     * Create a single 1D solid model
     * @param properties the heat transfer properties
     * @param initialization, math function to initialize the temperature
     * @param options the petsc options for the solver/ts
     */
    explicit SolidHeatTransferFactory(const std::shared_ptr<ablate::parameters::Parameters> &properties, const std::shared_ptr<ablate::mathFunctions::MathFunction> &initialization,
                                      const std::shared_ptr<ablate::parameters::Parameters> &options = {});

    /**
     * Create a new instance of the solid heat transfer from the factory
     * @return
     */
    std::shared_ptr<SolidHeatTransfer> Create();
};

}  // namespace ablate::boundarySolver::subModels
#endif  // ABLATELIBRARY_SOLIDHEATTRANSFERFACTORY_HPP
