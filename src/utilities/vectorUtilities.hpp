#ifndef ABLATELIBRARY_VECTORUTILITIES_HPP
#define ABLATELIBRARY_VECTORUTILITIES_HPP
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace ablate::utilities {
class VectorUtilities {
   public:
    template <class T>
    inline const static std::vector<T> Empty = {};

    template <class T>
    static inline std::vector<T> Merge(const std::vector<T>& a, const std::vector<T>& b) {
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
        for (const auto& ptr : vector) {
            result.push_back(*ptr);
        }
        return result;
    }

    /**
     * Copies vector of items to vector of shared pointers
     * @tparam T
     * @param vector
     * @return
     */
    template <class T>
    static inline std::vector<std::shared_ptr<T>> Copy(const std::vector<T>& vector) {
        std::vector<std::shared_ptr<T>> result;
        for (const auto& item : vector) {
            result.push_back(std::make_shared<T>(item));
        }
        return result;
    }

    /**
     * Fills an array based upon a key vector and map
     * @tparam T
     * @param list
     * @return
     */
    template <class K, class T>
    static inline std::vector<T> Fill(const std::vector<K>& keys, const std::map<K, T>& values, T defaultValue = {}) {
        std::vector<T> result(keys.size(), defaultValue);
        for (std::size_t i = 0; i < keys.size(); i++) {
            if (values.count(keys[i])) {
                result[i] = values.at(keys[i]);
            }
        }
        return result;
    }

   private:
    /**
     * helper function for Concatenate to string
     * @param value
     * @return
     */
    static std::string toString(const std::string& value) { return value; }

    /**
     * helper function for Concatenate to string
     * @param value
     * @return
     */
    template <class T>
    static std::string toString(const T& value) {
        return std::to_string(value);
    }

   public:
    /**
     * Concatenate a vector to strings
     * @tparam T
     * @param list
     * @return
     */
    template <class T>
    static inline std::string Concatenate(const std::vector<T>& vector, const std::string& delimiter = ", ") {
        using namespace std;
        return std::accumulate(std::begin(vector), std::end(vector), std::string(), [&delimiter](std::string& ss, auto& s) { return ss.empty() ? toString(s) : ss + delimiter + toString(s); });
    }

    /**
     * Concatenate an array to strings
     * @tparam T
     * @param list
     * @return
     */
    template <class T, class S>
    static inline std::string Concatenate(const T* vector, S size, const std::string& delimiter = ", ") {
        if (size <= 0) {
            return "";
        }

        std::string result = toString(vector[0]);

        for (S i = 1; i < size; ++i) {
            result += delimiter + toString(vector[i]);
        }

        return result;
    }

    /**
     * Finds the first item in list that is of type S
     * @tparam S the type of item to find
     * @tparam T
     * @param list
     * @return the first item of type S or null
     */
    template <class S, class T>
    static inline std::shared_ptr<S> Find(const std::vector<std::shared_ptr<T>>& list) {
        for (auto& item : list) {
            if (auto itemAsS = std::dynamic_pointer_cast<S>(item)) {
                return itemAsS;
            }
        }
        return {};
    }

    /**
     * Finds the first item in list that is of type S
     * @tparam L the vector or map
     * @tparam S the type of item to transform into
     * @tparam T the source list type
     * @param list
     * @param the transformation function
     * @return the transformed list
     */
    template <class L, class S, class T>
    static inline std::vector<S> Transform(const L& items, std::function<S(const T&)> function) {
        std::vector<S> transformed;
        transformed.reserve(items.size());

        for (const auto& item : items) {
            transformed.push_back(function(item));
        }

        return transformed;
    }

   private:
    VectorUtilities() = delete;
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
