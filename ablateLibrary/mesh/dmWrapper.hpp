#ifndef ABLATELIBRARY_DMWRAPPER_HPP
#define ABLATELIBRARY_DMWRAPPER_HPP
#include "mesh.hpp"

namespace ablate::mesh {
class DMWrapper : public ablate::mesh::Mesh {
   public:
    explicit DMWrapper(DM dm);
    ~DMWrapper() = default;
};
}  // namespace ablate::mesh

#endif  // ABLATELIBRARY_DMREFERENCE_HPP
