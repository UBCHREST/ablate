#include "initializer.hpp"
#include "utilities/petscOptions.hpp"

ablate::particles::initializers::Initializer::Initializer(std::map<std::string, std::string> arguments) : arguments(arguments) { utilities::PetscOptions::Set(arguments); }
