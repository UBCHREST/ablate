#include "builder.hpp"
#include "utilities/petscOptions.hpp"
#include "solve/timeStepper.hpp"
#include "flow/flow.hpp"

void ablate::Builder::Run(std::shared_ptr<ablate::parser::Factory> parser) {

    // get the global arguments
    auto globalArguments = parser->Get(parser::ArgumentIdentifier<std::map<std::string, std::string>>{"arguments"});
    utilities::PetscOptions::Set(globalArguments);

    // create a time stepper
    auto timeStepper = parser->Get(parser::ArgumentIdentifier<solve::TimeStepper>{"timestepper"});

    // assume one flow field right now
    auto flow = parser->Get(parser::ArgumentIdentifier<flow::Flow>{"flow"});

    // run
    timeStepper->Solve(flow);
}
