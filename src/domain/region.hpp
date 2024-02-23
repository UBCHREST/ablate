#ifndef ABLATELIBRARY_REGION_HPP
#define ABLATELIBRARY_REGION_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <utilities/petscUtilities.hpp>
#include <vector>
namespace ablate::domain {

class Region {
   public:
    inline const static std::shared_ptr<Region> ENTIREDOMAIN = {};

   private:
    const std::string name;
    const PetscInt value;
    std::size_t id;

   public:
    /**
     * Create a region that includes the label name and value
     * @param name label name
     * @param value label value
     */
    explicit Region(const std::string& name = {}, int value = 1);

    [[nodiscard]] inline const std::size_t& GetId() const { return id; }

    [[nodiscard]] inline const std::string& GetName() const { return name; }

    [[nodiscard]] inline const PetscInt& GetValue() const { return value; }

    [[nodiscard]] inline std::string ToString() const { return name + ":" + std::to_string(value); };

    /**
     * create and returns a label/region value
     * @param region
     * @param dm
     * @param regionLabel
     * @param regionValue
     */
    void CreateLabel(DM dm, DMLabel& regionLabel, PetscInt& regionValue) const;

    /**
     * static call to get a label from an optional region
     * @param region
     * @param dm
     * @param regionLabel
     * @param regionValue
     */
    static void GetLabel(const std::shared_ptr<Region>& region, DM dm, DMLabel& regionLabel, PetscInt& regionValue);

    /**
     * call to get a label from an a region
     * @param dm
     * @param regionLabel
     * @param regionValue
     */
    void GetLabel(DM dm, DMLabel& regionLabel, PetscInt& regionValue);

    /**
     * static call to see if a point is in region/or null
     * @param region
     * @param dm
     * @param point
     * @return
     */
    static bool InRegion(const std::shared_ptr<Region>& region, DM dm, PetscInt point);

    /**
     * Non static call to see if this point is in a given region
     * @param dm
     * @param point
     * @return
     */
    inline bool InRegion(DM dm, PetscInt point) const {
        PetscInt ptValue;
        DMGetLabelValue(dm, name.c_str(), point, &ptValue) >> utilities::PetscUtilities::checkError;
        return ptValue == value;
    }

    /**
     * throws exception if the label is not in the dm
     * @param region
     * @param dm
     */
    void CheckForLabel(DM dm) const;

    /**
     * throws exception if the label is not in the dm on any rank
     * @param region
     * @param dm
     */
    void CheckForLabel(DM dm, MPI_Comm comm) const;

    /**
     * provide simple comparsion == operator
     * @return
     */
    inline bool operator==(const Region& otherRegion) const { return id == otherRegion.id; }

    /**
     * simple comparsion operator allowing regions to be used in a map
     * @param otherRegion
     * @return
     */
    inline bool operator<(const Region& otherRegion) const { return id < otherRegion.id; }
};

std::ostream& operator<<(std::ostream& os, const Region& region);

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Region>& region);

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_REGION_HPP
