#ifndef ABLATELIBRARY_RESTARABLE_HPP
#define ABLATELIBRARY_RESTARABLE_HPP
#include "restoreState.hpp"
#include "saveState.hpp"

namespace ablate::environment {
class Restartable {
   public:
    virtual ~Restartable() = default;
    virtual const std::string& GetName() const = 0;
    virtual void Save(SaveState&) const = 0;
    virtual void Restore(const RestoreState&)  = 0;
};
}

#endif  // ABLATELIBRARY_RESTARABLE_HPP
