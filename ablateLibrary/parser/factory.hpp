#ifndef ABLATELIBRARY_FACTORY_HPP
#define ABLATELIBRARY_FACTORY_HPP

#include "argumentIdentifier.hpp"
#include <iostream>

namespace ablate{
namespace parser {

class Factory;
template <typename Interface>
std::unique_ptr<Interface> ResolveAndCreate(Factory& factory);

class Factory {
   public:
    const std::string type;

    Factory(std::string type) : type(type){}

    const std::string GetType() const {
        return type;
    }

    virtual std::string Get(const ArgumentIdentifier<std::string> identifier) = 0;

    //    virtual double GetDouble(const std::string& name) = 0;
    //    virtual int GetInt(const std::string& name) = 0;
    virtual std::shared_ptr<Factory> GetFactory(const std::string& name) = 0;

    template <typename Interface>
    std::unique_ptr<Interface> Get(const ArgumentIdentifier<Interface>& identifier) {
        auto childFactory = GetFactory(identifier.inputName);
        return ResolveAndCreate<Interface>(*childFactory);
    }
};

}
}

#endif  // ABLATELIBRARY_FACTORY_HPP