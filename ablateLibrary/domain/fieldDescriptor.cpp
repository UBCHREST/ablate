#include "fieldDescriptor.hpp"
#include <map>
#include "parser/registrar.hpp"

static std::shared_ptr<ablate::domain::FieldDescriptor> FlowFieldDescriptorCreateFunction(std::shared_ptr<ablate::parser::Factory> factory) {
    return std::shared_ptr<ablate::domain::FieldDescriptor>(
        new ablate::domain::FieldDescriptor{.fieldName = factory->GetByName<std::string>("fieldName"),
                                            .fieldPrefix = factory->GetByName<std::string>("fieldPrefix"),
                                            .components = factory->GetByName<int>("components"),
                                            .componentNames = factory->GetByName<std::vector<std::string>>("componentNames", std::vector<std::string>{}),
                                            .fieldLocation = factory->GetByName<ablate::parser::EnumWrapper<ablate::domain::FieldLocation>>("fieldLocation", ablate::domain::FieldLocation::SOL)});
}

REGISTER_FACTORY_FUNCTION_DEFAULT(ablate::domain::FieldDescriptor, ablate::domain::FieldDescriptor, "flow field description", FlowFieldDescriptorCreateFunction);
