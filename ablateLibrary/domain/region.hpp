#ifndef ABLATELIBRARY_REGION_HPP
#define ABLATELIBRARY_REGION_HPP

#include <petsc.h>
#include <memory>
#include <string>
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
    Region(std::string name = {}, int = 1);

    inline const std::size_t& GetId() const { return id; }

    inline const std::string& GetName() const { return name; }

    inline const PetscInt& GetValue() const { return value; }

    inline const std::string ToString() const { return name + ":" + std::to_string(value); };
};

std::ostream& operator<<(std::ostream& os, const Region& region);

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Region>& region);

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_REGION_HPP
