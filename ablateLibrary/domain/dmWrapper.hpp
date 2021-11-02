#ifndef ABLATELIBRARY_DMWRAPPER_HPP
#define ABLATELIBRARY_DMWRAPPER_HPP
#include "domain.hpp"

namespace ablate::domain {
class DMWrapper : public ablate::domain::Domain {
   public:
    explicit DMWrapper(DM dm, std::vector<std::shared_ptr<modifier::Modifier>> modifiers = {});
    ~DMWrapper() = default;
};
}  // namespace ablate::domain

#endif  // ABLATELIBRARY_DMREFERENCE_HPP
