#ifndef ABLATELIBRARY_DMTRANSFER_HPP
#define ABLATELIBRARY_DMTRANSFER_HPP
#include "domain.hpp"

namespace ablate::domain {
/**
 * Transfer ownership ownership of the specified domain and destroy it when complete.
 */
class DMTransfer : public ablate::domain::Domain {
   public:
    explicit DMTransfer(DM dm, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers = {},
                        std::shared_ptr<parameters::Parameters> options = {});
    ~DMTransfer() override;
};
}  // namespace ablate::domain

#endif  // ABLATELIBRARY_DMREFERENCE_HPP
