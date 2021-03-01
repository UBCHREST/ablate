#include "builder.hpp"
#include "flow/flow.hpp"
#include "particles/particles.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscOptions.hpp"
#include "version.h"

void ablate::Builder::Run(std::shared_ptr<ablate::parser::Factory> parser) {
    // get the global arguments
    auto globalArguments = parser->Get(parser::ArgumentIdentifier<std::map<std::string, std::string>>{"arguments"});
    utilities::PetscOptions::Set(globalArguments);

    // create a time stepper
    auto timeStepper = parser->Get(parser::ArgumentIdentifier<solve::TimeStepper>{"timestepper"});

    // assume one flow field right now
    auto flow = parser->Get(parser::ArgumentIdentifier<flow::Flow>{"flow"});
    flow->SetupSolve(timeStepper->GetTS());

    // get any particles that may be in the flow
    auto particleList = parser->Get(parser::ArgumentIdentifier<std::vector<particles::Particles>>{"particles"});

    // initialize the flow for each
    for (auto particle : particleList) {
        particle->InitializeFlow(flow, timeStepper);
    }

    // run
    timeStepper->Solve(flow);
}

void ablate::Builder::PrintVersion(std::ostream& stream) {
    stream << "ABLATE: " << std::endl;
    stream << '\t' << "Documentation: https://ubchrest.github.io/ablate/" << std::endl;
    stream << '\t' << "Source: https://github.com/UBCHREST/ablate" << std::endl;
    stream << '\t' << "Version: " <<ABLATECORE_VERSION << std::endl;

}
