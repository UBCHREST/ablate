#ifndef ABLATELIBRARY_REGION_HPP
#define ABLATELIBRARY_REGION_HPP

#include <string>
#include <vector>

namespace ablate::domain {

class Region {
   public:
    inline const static std::shared_ptr<Region> ENTIREDOMAIN = {};

   private:
    const std::string name;
    std::vector<int> values;
    std::size_t id;

   public:
    Region(std::string name = {}, std::vector<int> = {1});

    inline const std::size_t& GetId() const { return id; }

    inline const std::string& GetName() const { return name; }

    inline const std::vector<int>& GetValues() const { return values; }
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_REGION_HPP
