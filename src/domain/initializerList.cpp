#include "initializerList.hpp"

ablate::domain::InitializerList::InitializerList(std::vector<std::shared_ptr<Initializer>> initializers) : initializers(std::move(initializers)) {}

std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> ablate::domain::InitializerList::GetFieldFunctions(const std::vector<domain::Field>& fields) const {
    std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> fieldFunctions;

    for (const auto& initializer : initializers) {
        auto initializerFieldFunctions = initializer->GetFieldFunctions(fields);

        fieldFunctions.insert(initializerFieldFunctions.begin(), initializerFieldFunctions.end(), fieldFunctions.end());
    }

    return fieldFunctions;
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::domain::Initializer, ablate::domain::InitializerList, "Allow multiple different initializer to be combined", std::vector<ablate::domain::Initializer>);