#include "builder.hpp"
#include <yaml-cpp/yaml.h>
#include "monitors/monitor.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/stringUtilities.hpp"
#include "version.h"

std::shared_ptr<ablate::solver::TimeStepper> ablate::Builder::Build(const std::shared_ptr<cppParser::Factory>& parser) {
    // get the global arguments
    auto globalArguments = parser->GetByName<ablate::parameters::Parameters>("arguments");
    if (globalArguments) {
        globalArguments->Fill(nullptr);
    }

    // create a time stepper
    auto timeStepper = parser->Get(cppParser::ArgumentIdentifier<solver::TimeStepper>{.inputName = "timestepper", .description = "", .optional = false});

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

    return timeStepper;
}

void ablate::Builder::Run(const std::shared_ptr<cppParser::Factory>& parser) {
    // build the time stepper
    auto timeStepper = Build(parser);

    // advance the time stepper
    timeStepper->Solve();
}

void ablate::Builder::PrintVersion(std::ostream& stream) { stream << ABLATE_VERSION; }

void ablate::Builder::PrintInfo(std::ostream& stream) {
    // force this to print as yaml, so it is human and machine-readable
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "ABLATE";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "documentation";
    out << YAML::Value << "https://ablate.dev";
    out << YAML::Key << "source";
    out << YAML::Value << "https://github.com/UBCHREST/ablate";
    out << YAML::Key << "version";
    out << YAML::Value << ABLATE_VERSION;

    // build and print the petsc version number
    out << YAML::Key << "petscVersion";
    std::stringstream petscVersion;
    petscVersion << PETSC_VERSION_MAJOR << "." << PETSC_VERSION_MINOR << "." << PETSC_VERSION_SUBMINOR;
    out << YAML::Value << petscVersion.str();

    // Build the hash if needed, only limit the length if it is more than a hash
    out << YAML::Key << "petscGitCommit";
    auto petscGit = std::string(PETSC_VERSION_GIT);
    if (ablate::utilities::StringUtilities::Contains(petscGit, "-")) {
        size_t position = petscGit.find_last_of('-');
        petscGit = petscGit.substr(position + 1);
    } else {
        petscGit.resize(8);
    }
    out << YAML::Value << petscGit;
    out << YAML::EndMap << YAML::EndMap;

    // Pipe to the output stream
    stream << out.c_str();
}
