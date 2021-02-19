#ifndef ABLATELIBRARY_REGISTRAR_HPP
#define ABLATELIBRARY_REGISTRAR_HPP

#include <string>
#include "argumentIdentifier.hpp"
#include "factory.hpp"
#include <map>

namespace ablate{
namespace parser{

template <typename Interface>
class Registrar{
   private:
    Registrar() = delete;

   public:
    using TCreateMethod = std::function<std::unique_ptr<Interface>(Factory&)>;

    /* Register a class that has a constructor that uses a Factory instance */
    template <typename Class>
    static bool Register(const std::string className){
        if (auto it = s_methods.find(className); it == s_methods.end())
        { // C++17 init-if ^^
            s_methods[className] = [](Factory& factory) {
              return std::make_unique<Class>(factory);
            };
        }
        return false;
    }

    /* Register a class with a function that takes a factory */
    static bool Register(const std::string className, TCreateMethod funcCreate){
        if (auto it = s_methods.find(className); it == s_methods.end())
        {
            s_methods[className] = funcCreate;
            return true;
        }
        return false;
    }

    /* Register a class with a function that takes a factory */
    template<typename Class, typename... Args>
    static bool Register(const std::string className, ArgumentIdentifier<Args>&&... args){
        if (auto it = s_methods.find(className); it == s_methods.end())
        {
            s_methods[className] = [=](Factory& factory){
              return std::make_unique<Class>(factory.Get(args)...);
            };
            return true;
        }
        return false;
    }

    static TCreateMethod GetCreateMethod(const std::string& className){
        if (auto it = s_methods.find(className); it != s_methods.end())
            return it->second;

        return nullptr;
    }

   private:
    inline static std::map<std::string, TCreateMethod> s_methods;

};

template <typename Interface>
std::unique_ptr<Interface> ResolveAndCreate(Factory& factory){
    auto childType = factory.GetType();

    std::function<std::unique_ptr<Interface>(Factory&)> createMethod = Registrar<Interface>::GetCreateMethod(childType);
        if (!createMethod) {
            std::cout << "error missing type " << childType << std::endl;
        }

    return createMethod(factory);
}

}
}


#endif  // ABLATELIBRARY_REGISTRAR_HPP
