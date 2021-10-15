#include "fieldDescriptor.hpp"
#include <map>
#include "parser/registrar.hpp"

static std::shared_ptr<ablate::domain::FieldDescriptor> FlowFieldDescriptorCreateFunction(std::shared_ptr<ablate::parser::Factory> factory) {
    return std::shared_ptr<ablate::domain::FieldDescriptor>(
        new ablate::domain::FieldDescriptor{.name = factory->GetByName<std::string>("name"),
                                            .prefix = factory->GetByName<std::string>("prefix"),
                                            .components= factory->GetByName<std::vector<std::string>>("components", std::vector<std::string>{}),
                                            .fieldLocation = factory->GetByName<ablate::parser::EnumWrapper<ablate::domain::FieldType>>("fieldLocation", ablate::domain::FieldType::SOL)});
}

REGISTER_FACTORY_FUNCTION_DEFAULT(ablate::domain::FieldDescriptor, ablate::domain::FieldDescriptor, "flow field description", FlowFieldDescriptorCreateFunction);
