#ifndef ABLATELIBRARY_VECTORUTILITIES_HPP
#define ABLATELIBRARY_VECTORUTILITIES_HPP
#include <vector>

namespace ablate::utilities {
class VectorUtilities {
   public:
    template <class T>
    static inline std::vector<T> Merge(const std::vector<T>& a, std::vector<T>& b) {
        if (a.empty()) {
            return b;
        }
        if (b.empty()) {
            return a;
        }
        std::vector<T> result(a.begin(), a.end());
        result.insert(result.end(), b.begin(), b.end());
        return result;
    }

    template <class T>
    static inline std::vector<T> Prepend(const std::vector<T>& a, T value) {
        std::vector<T> result(a.size());
        for (std::size_t i = 0; i < a.size(); i++) {
            result[i] = value + a[i];
        }
        return result;
    }

    /**
     * Makes a copy of the supplied vector components from the shared ptr
     * @tparam T
     * @param list
     * @return
     */
    template <class T>
    static inline std::vector<T> Copy(const std::vector<std::shared_ptr<T>>& vector) {
        std::vector<T> result;
        for(const auto& ptr : vector){
            result.push_back(*ptr);
        }
        return result;
    }

   private:
    VectorUtilities() = delete;
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
