#include "fieldDescriptor.hpp"
#include <map>
#include "parser/registrar.hpp"

static std::shared_ptr<ablate::domain::FieldDescriptor> FlowFieldDescriptorCreateFunction(std::shared_ptr<ablate::parser::Factory> factory) {
    auto name = factory->GetByName<std::string>("name");
    return std::shared_ptr<ablate::domain::FieldDescriptor>(
        new ablate::domain::FieldDescriptor{.name = name,
                                            .prefix = factory->Contains("prefix") ? factory->GetByName<std::string>("prefix") : name,
                                            .components = factory->GetByName<std::vector<std::string>>("components", std::vector<std::string>{"_"}),
                                            .type = factory->GetByName<ablate::parser::EnumWrapper<ablate::domain::FieldType>>("type", ablate::domain::FieldType::SOL),
                                            .adjacency = factory->GetByName<ablate::parser::EnumWrapper<ablate::domain::FieldAdjacency>>("adjacency", ablate::domain::FieldAdjacency::DEFAULT)});
}

REGISTER_FACTORY_FUNCTION_DEFAULT(ablate::domain::FieldDescriptor, ablate::domain::FieldDescriptor, "flow field description", FlowFieldDescriptorCreateFunction);
