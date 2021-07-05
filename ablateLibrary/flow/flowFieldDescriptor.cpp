#include "flowFieldDescriptor.hpp"
#include <map>
#include "parser/registrar.hpp"

static const std::map<std::string, ablate::flow::FieldType> stringToFieldType = {
    {"fe", ablate::flow::FieldType::FE}, {"FE", ablate::flow::FieldType::FE}, {"fv", ablate::flow::FieldType::FV}, {"FV", ablate::flow::FieldType::FV}};

static std::shared_ptr<ablate::flow::FlowFieldDescriptor> FlowFieldDescriptorCreateFunction(std::shared_ptr<ablate::parser::Factory> factory) {
    return std::shared_ptr<ablate::flow::FlowFieldDescriptor>(
        new ablate::flow::FlowFieldDescriptor{.solutionField = factory->GetByName<bool>("solutionField", true),
                                              .fieldName = factory->GetByName<std::string>("fieldName"),
                                              .fieldPrefix = factory->GetByName<std::string>("fieldPrefix"),
                                              .components = factory->GetByName<PetscInt>("components"),
                                              .fieldType = factory->GetByName<ablate::parser::EnumWrapper<ablate::flow::FieldType>>("fieldType"),
                                              .componentNames = factory->GetByName<std::vector<std::string>>("componentNames", std::vector<std::string>{})});
}

std::istream& ablate::flow::operator>>(std::istream& is, ablate::flow::FieldType& v) {
    // get the key string
    std::string enumString;
    is >> enumString;
    v = stringToFieldType.at(enumString);
    return is;
}

REGISTER_FACTORY_FUNCTION_DEFAULT(ablate::flow::FlowFieldDescriptor, ablate::flow::FlowFieldDescriptor, "flow field description", FlowFieldDescriptorCreateFunction);
