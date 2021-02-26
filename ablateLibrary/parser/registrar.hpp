#ifndef ABLATELIBRARY_REGISTRAR_HPP
#define ABLATELIBRARY_REGISTRAR_HPP

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include "argumentIdentifier.hpp"
#include "factory.hpp"
#include "listing.h"
#include "utilities/demangler.hpp"

/**
 * Helper macros for registering classes
 */

#define RESOLVE(interfaceTypeFullName, classFullName) \
    template <>                                       \
    std::shared_ptr<interfaceTypeFullName> ablate::parser::RegisteredInFactory<interfaceTypeFullName, classFullName>::Resolved = ablate::parser::ResolveAndCreate<interfaceTypeFullName>(nullptr)

#define REGISTER_FACTORY_CONSTRUCTOR(interfaceTypeFullName, classFullName, description)                                \
    template <>                                                                                                        \
    bool ablate::parser::RegisteredInFactory<interfaceTypeFullName, classFullName>::Registered =                       \
        ablate::parser::Registrar<interfaceTypeFullName>::Register<classFullName>(false, #classFullName, description); \
    RESOLVE(interfaceTypeFullName, classFullName)

#define REGISTER(interfaceTypeFullName, classFullName, description, ...)                                                            \
    template <>                                                                                                                     \
    bool ablate::parser::RegisteredInFactory<interfaceTypeFullName, classFullName>::Registered =                                    \
        ablate::parser::Registrar<interfaceTypeFullName>::Register<classFullName>(false, #classFullName, description, __VA_ARGS__); \
    RESOLVE(interfaceTypeFullName, classFullName)

#define REGISTER_FACTORY_CONSTRUCTOR_DEFAULT(interfaceTypeFullName, classFullName, description)                       \
    template <>                                                                                                       \
    bool ablate::parser::RegisteredInFactory<interfaceTypeFullName, classFullName>::Registered =                      \
        ablate::parser::Registrar<interfaceTypeFullName>::Register<classFullName>(true, #classFullName, description); \
    RESOLVE(interfaceTypeFullName, classFullName)

#define REGISTERDEFAULT(interfaceTypeFullName, classFullName, description, ...)                                                    \
    template <>                                                                                                                    \
    bool ablate::parser::RegisteredInFactory<interfaceTypeFullName, classFullName>::Registered =                                   \
        ablate::parser::Registrar<interfaceTypeFullName>::Register<classFullName>(true, #classFullName, description, __VA_ARGS__); \
    RESOLVE(interfaceTypeFullName, classFullName)

namespace ablate::parser {

template <typename Interface>
class Registrar {
   public:
    Registrar() = delete;

    using TCreateMethod = std::function<std::shared_ptr<Interface>(std::shared_ptr<Factory>)>;

    /* Register a class that has a constructor that uses a Factory instance */
    template <typename Class>
    static bool Register(bool defaultConstructor, const std::string&& className, const std::string&& description) {
        if (auto it = s_methods.find(className); it == s_methods.end()) {
            // Record the entry
            Listing::Get().RecordListing(Listing::ClassEntry{.interface = typeid(Interface).name(), .className = className, .description = description, .defaultConstructor = defaultConstructor});

            // create method
            s_methods[className] = [](std::shared_ptr<Factory> factory) { return std::make_shared<Class>(factory); };

            if (defaultConstructor) {
                if (!defaultCreationMethod) {
                    defaultCreationMethod = s_methods[className];
                } else {
                    throw std::invalid_argument("the default parameter for " + utilities::Demangler::Demangle(typeid(Interface).name()) + " is already set");
                }
            }
        }
        return false;
    }

    /* Register a class with a function that takes argument identifiers */
    template <typename Class, typename... Args>
    static bool Register(bool defaultConstructor, const std::string&& className, const std::string&& description, ArgumentIdentifier<Args>&&... args) {
        if (auto it = s_methods.find(className); it == s_methods.end()) {
            // Record the entry
            Listing::Get().RecordListing(
                Listing::ClassEntry{.interface = typeid(Interface).name(),
                                    .className = className,
                                    .description = description,
                                    .arguments = std::vector({Listing::ArgumentEntry{.name = args.inputName, .interface = typeid(Args).name(), .description = args.description}...}),
                                    .defaultConstructor = defaultConstructor});

            // create method
            s_methods[className] = [=](std::shared_ptr<Factory> factory) { return std::make_shared<Class>(factory->Get(args)...); };

            if (defaultConstructor) {
                if (!defaultCreationMethod) {
                    defaultCreationMethod = s_methods[className];
                } else {
                    throw std::invalid_argument("the default parameter for " + utilities::Demangler::Demangle(typeid(Interface).name()) + " is already set");
                }
            }

            return true;
        }
        return false;
    }

    static TCreateMethod GetCreateMethod(const std::string& className) {
        if (auto it = s_methods.find(className); it != s_methods.end()) return it->second;

        return nullptr;
    }

    static TCreateMethod GetDefaultCreateMethod() { return defaultCreationMethod; };

   private:
    inline static std::map<std::string, TCreateMethod> s_methods;

    inline static TCreateMethod defaultCreationMethod;
};

template <typename Interface>
std::shared_ptr<Interface> ResolveAndCreate(std::shared_ptr<Factory> factory) {
    if (factory == nullptr) {
        return nullptr;
    }
    auto childType = factory->GetClassType();

    if (!childType.empty()) {
        std::function<std::shared_ptr<Interface>(std::shared_ptr<Factory>)> createMethod = Registrar<Interface>::GetCreateMethod(childType);
        if (!createMethod) {
            throw std::invalid_argument("unknown type " + childType);
        }

        return createMethod(factory);
    } else {
        // check for a default
        std::function<std::shared_ptr<Interface>(std::shared_ptr<Factory>)> createMethod = Registrar<Interface>::GetDefaultCreateMethod();
        if (!createMethod) {
            throw std::invalid_argument("no default creator specified for interface " + utilities::Demangler::Demangle(typeid(Interface).name()));
        }

        return createMethod(factory);
    }
}
template <typename Interface, typename Class>
class RegisteredInFactory {
   public:
    static bool Registered;
    static std::shared_ptr<Interface> Resolved;
};

}  // namespace ablate::parser

#endif  // ABLATELIBRARY_REGISTRAR_HPP
