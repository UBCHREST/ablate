#ifndef ABLATELIBRARY_DMPLEX_HPP
#define ABLATELIBRARY_DMPLEX_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include "domain.hpp"

namespace ablate::domain {

class DMPlex : public Domain {
   private:
    static DM CreateDM(const std::string& name);

   public:
    DMPlex(std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, const std::string& name = "dmplex", std::vector<std::shared_ptr<modifiers::Modifier>> modifiers = {},
           std::shared_ptr<parameters::Parameters> options = {});
    ~DMPlex() override;
};
}  // namespace ablate::domain

#endif  // ABLATELIBRARY_DMPLEX_HPP
