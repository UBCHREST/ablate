#include "initializer.hpp"
ablate::domain::Initializer::Initializer(std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldFunctions) : fieldFunctions(std::move(fieldFunctions)) {}

#include "registrar.hpp"
REGISTER_DEFAULT_PASS_THROUGH(ablate::domain::Initializer, ablate::domain::Initializer, "Simple class used to produce the field functions for initialization",
                              std::vector<ablate::mathFunctions::FieldFunction>);
