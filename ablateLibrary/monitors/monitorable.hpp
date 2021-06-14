#ifndef ABLATELIBRARY_MONITORABLE_HPP
#define ABLATELIBRARY_MONITORABLE_HPP
#include <petsc.h>
#include <memory>

namespace ablate::monitors {

class Monitorable {
   public:
    virtual const std::string& GetName() const = 0;
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_MONITORABLE_HPP
