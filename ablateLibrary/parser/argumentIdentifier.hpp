#ifndef ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
#define ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP

namespace ablate{
namespace parser{
template<typename Interface>
class ArgumentIdentifier{
   public:
    const std::string inputName;
    const std::string description;
};
}
}

#endif  // ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
