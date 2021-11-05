#ifndef ABLATELIBRARY_FIELDDESCRIPTOR_HPP
#define ABLATELIBRARY_FIELDDESCRIPTOR_HPP


#include <memory>
#include <vector>

namespace ablate::domain::fields {

// forward declare FieldDescription to prevent circular reference
struct FieldDescription;

/**
 * interface that lists the fields needed for the domain
 */
class FieldDescriptor {
   public:
    virtual std::vector<std::shared_ptr<FieldDescription>> GetFields() = 0;
    virtual ~FieldDescriptor() = default;
};

}  // namespace ablate::domain::fields

#endif  // ABLATELIBRARY_FIELDINITIALIZER_HPP
