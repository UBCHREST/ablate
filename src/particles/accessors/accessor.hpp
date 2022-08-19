#ifndef ABLATELIBRARY_PARTICLEACCESSOR_HPP
#define ABLATELIBRARY_PARTICLEACCESSOR_HPP

#include <petsc.h>
#include <functional>
#include <map>
#include <vector>
#include "particles/field.hpp"
#include "pointData.hpp"
#include "utilities/petscError.hpp"

namespace ablate::particles::accessors {
/**
 * Class responsible for computing point data locations for particle integration
 */
template <class DataType>
class Accessor {
   private:
    /**
     * Keep a map of point data that can be reused.
     */
    std::map<std::string, Data<DataType>> dataCache;

    /**
     * determine if we cache the data
     */
    const bool cachePointData;

    /**
     * Keep a vector of destructors to call
     */
    std::vector<std::function<void()>> destructors;

   protected:
    /**
     * protected call to create the instance of point data
     * @param fieldName
     * @return
     */
    virtual Data<DataType> CreateData(const std::string& fieldName) = 0;

   public:
    explicit Accessor(bool cachePointData) : cachePointData(cachePointData), destructors(cachePointData ? 5 : 0) {}

    virtual ~Accessor() {
        for (auto& function : destructors) {
            function();
        }
    };

    /**
     * Get the field data for this field
     * @param fieldName
     * @return
     */
    Data<DataType> operator[](const std::string& fieldName) { return GetData(fieldName); }

    /**
     * Get the field data for this field
     * @param fieldName
     * @return
     */
    inline Data<DataType> GetData(const std::string& fieldName) {
        if (cachePointData) {
            if (dataCache.count(fieldName)) {
                dataCache[fieldName] = CreateData(fieldName);
            }
            return dataCache.at(fieldName);

        } else {
            return CreateData(fieldName);
        }
    }

    /**
     * register a cleanup function
     * @param function
     */
    inline void RegisterCleanupFunction(const std::function<void()>& function) { destructors.push_back(function); }

    /**
     * prevent copy of this class
     */
    Accessor(const Accessor&) = delete;
};
}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_SWARMACCESSOR_HPP
