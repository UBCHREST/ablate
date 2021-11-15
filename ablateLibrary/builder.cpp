#include "builder.hpp"
#include "monitors/monitor.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/petscOptions.hpp"
#include "version.h"

void ablate::Builder::Run(std::shared_ptr<ablate::parser::Factory> parser) {
    // get the global arguments
    auto globalArguments = parser->Get(parser::ArgumentIdentifier<std::map<std::string, std::string>>{.inputName = "arguments"});
    utilities::PetscOptionsUtils::Set(globalArguments);

    // create a time stepper
    auto timeStepper = parser->Get(parser::ArgumentIdentifier<solver::TimeStepper>{.inputName = "timestepper"});

    // Check to see if a single or multiple solvers were specified
    if (parser->Contains("solver")) {
        auto solver = parser->GetByName<solver::Solver>("solver");
        auto solverMonitors = parser->GetFactory("solver")->GetByName<std::vector<monitors::Monitor>>("monitors", std::vector<std::shared_ptr<monitors::Monitor>>());
        timeStepper->Register(solver, solverMonitors);
    }

    // Add in other solvers
    auto solverList = parser->GetByName<std::vector<solver::Solver>>("solvers", std::vector<std::shared_ptr<solver::Solver>>());
    if (!solverList.empty()) {
        auto solverFactorySequence = parser->GetFactorySequence("solvers");

        // initialize the flow for each
        for (std::size_t i = 0; i < solverFactorySequence.size(); i++) {
            auto& solver = solverList[i];
            auto solverMonitors = solverFactorySequence[i]->GetByName<std::vector<monitors::Monitor>>("monitors", std::vector<std::shared_ptr<monitors::Monitor>>());
            timeStepper->Register(solver, solverMonitors);
        }
    }

    timeStepper->Solve();
}

void ablate::Builder::PrintVersion(std::ostream& stream) { stream << ABLATECORE_VERSION; }

void ablate::Builder::PrintInfo(std::ostream& stream) {
    stream << "ABLATE: " << std::endl;
    stream << '\t' << "Documentation: https://ablate.dev" << std::endl;
    stream << '\t' << "Source: https://github.com/UBCHREST/ablate" << std::endl;
    stream << '\t' << "Version: " << ABLATECORE_VERSION << std::endl;
}
