#ifndef ABLATELIBRARY_FACTORY_HPP
#define ABLATELIBRARY_FACTORY_HPP

#include <iostream>
#include "argumentIdentifier.hpp"

namespace ablate::parser {

class Factory;
template <typename Interface>
std::shared_ptr<Interface> ResolveAndCreate(Factory& factory);

class Factory {
   protected:
    /* return a factory that serves as the root of the requested item */
    virtual std::shared_ptr<Factory> GetFactory(const std::string& name) const = 0;

   public:
    /* gets the class type represented by this factory */
    virtual const std::string& GetClassType() const = 0;

    /* return a string*/
    virtual std::string Get(const ArgumentIdentifier<std::string>& identifier) const = 0;

    /* return an int for the specified identifier*/
    virtual int Get(const ArgumentIdentifier<int>& identifier) const = 0;

    /* produce a shared pointer for the specified interface and type */
    template <typename Interface>
    std::shared_ptr<Interface> Get(const ArgumentIdentifier<Interface>& identifier) const {
        auto childFactory = GetFactory(identifier.inputName);
        return ResolveAndCreate<Interface>(*childFactory);
    }
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_FACTORY_HPP