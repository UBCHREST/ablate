#include "solidHeatTransferFactory.hpp"
ablate::boundarySolver::subModels::SolidHeatTransferFactory::SolidHeatTransferFactory(const std::shared_ptr<ablate::parameters::Parameters>& properties,
                                                                                      const std::shared_ptr<ablate::mathFunctions::MathFunction>& initialization,
                                                                                      const std::shared_ptr<ablate::parameters::Parameters>& options)
    : properties(properties), initialization(initialization), options(options) {}

std::shared_ptr<ablate::boundarySolver::subModels::SolidHeatTransfer> ablate::boundarySolver::subModels::SolidHeatTransferFactory::Create() {
    return std::make_shared<SolidHeatTransfer>(properties, initialization, options);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::subModels::SolidHeatTransferFactory, ablate::boundarySolver::subModels::SolidHeatTransferFactory, "A factory to create solid heat transfer instances",
         ARG(ablate::parameters::Parameters, "properties", "the heat transfer properties (specificHeat, conductivity, density, maximumSurfaceTemperature, farFieldTemperature)"),
         ARG(ablate::mathFunctions::MathFunction, "initialization", " math function to initialize the temperature"),
         OPT(ablate::parameters::Parameters, "options", "the petsc options for the solver/ts"));
