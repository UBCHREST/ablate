#ifndef ABLATELIBRARY_MAPUTILITIES_HPP
#define ABLATELIBRARY_MAPUTILITIES_HPP
#include <map>

namespace ablate::utilities {
class MapUtilities {
   public:
    template <class T, class U>
    inline const static std::map<T, U> Empty = {};

    MapUtilities() = delete;
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MAPUTILITIES_HPP
