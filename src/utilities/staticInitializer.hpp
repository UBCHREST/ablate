#ifndef ABLATELIBRARY_STATICINITIALIZER_HPP
#define ABLATELIBRARY_STATICINITIALIZER_HPP
#include <algorithm>
#include <string>
#include <string_view>

namespace ablate::utilities {
/**
 * Simple helper implementation that calls the static initializer just once before the constructor
 */
class StaticInitializer {
   private:
    static inline bool initialized = false;

   public:
    explicit StaticInitializer(std::function<void()> init) {
        if (!initialized) {
            init();
        }
        initialized = true;
    }
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
