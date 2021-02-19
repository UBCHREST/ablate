#ifndef ABLATELIBRARY_REGISTRAR_HPP
#define ABLATELIBRARY_REGISTRAR_HPP

#include <map>
#include <stdexcept>
#include <string>
#include "argumentIdentifier.hpp"
#include "factory.hpp"
#include "listing.h"

namespace ablate::parser {

template <typename Interface>
class Registrar {
   public:
    Registrar() = delete;

    using TCreateMethod = std::function<std::shared_ptr<Interface>(Factory&)>;

    /* Register a class that has a constructor that uses a Factory instance */
    template <typename Class>
    static bool Register(const std::string&& className, const std::string&& description) {
        if (auto it = s_methods.find(className); it == s_methods.end()) {
            // Record the entry
            Listing::Get().RecordListing(Listing::ClassEntry{.interface = typeid(Interface).name(), .className = className, .description = description});

            // create method
            s_methods[className] = [](Factory& factory) { return std::make_shared<Class>(factory); };
        }
        return false;
    }

    /* Register a class with a function that takes argument identifiers */
    template <typename Class, typename... Args>
    static bool Register(const std::string&& className, const std::string&& description, ArgumentIdentifier<Args>&&... args) {
        if (auto it = s_methods.find(className); it == s_methods.end()) {
            // Record the entry
            Listing::Get().RecordListing(
                Listing::ClassEntry{.interface = typeid(Interface).name(),
                                    .className = className,
                                    .description = description,
                                    .arguments = std::vector({Listing::ArgumentEntry{.name = args.inputName, .description = args.description, .interface = typeid(Args).name()}...})});

            // create method
            s_methods[className] = [=](Factory& factory) { return std::make_shared<Class>(factory.Get(args)...); };
            return true;
        }
        return false;
    }

    static TCreateMethod GetCreateMethod(const std::string& className) {
        if (auto it = s_methods.find(className); it != s_methods.end()) return it->second;

        return nullptr;
    }

   private:
    inline static std::map<std::string, TCreateMethod> s_methods;
};

template <typename Interface>
std::shared_ptr<Interface> ResolveAndCreate(Factory& factory) {
    auto childType = factory.GetClassType();

    std::function<std::shared_ptr<Interface>(Factory&)> createMethod = Registrar<Interface>::GetCreateMethod(childType);
    if (!createMethod) {
        throw std::invalid_argument("unkown type " + childType);
    }

    return createMethod(factory);
}
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_REGISTRAR_HPP
