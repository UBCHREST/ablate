#ifndef ABLATELIBRARY_SAVESTATE_HPP
#define ABLATELIBRARY_SAVESTATE_HPP

#include "parameters/parameters.hpp"
#include <sstream>
#include <string>

namespace ablate::environment {

class SaveState {
   public:
    virtual void Save(const std::string&, const std::string&) = 0;
    virtual void Save(const std::string&, Vec) = 0;

    template <typename T>
    void Save(const std::string& key, const T& value) {
        std::stringstream ss;
        ss << value;
        Save(key, ss.str());
    }
};
}  // namespace ablate::environment
#endif  // ABLATELIBRARY_SAVESTATE_HPP
