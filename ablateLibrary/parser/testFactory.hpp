#ifndef ABLATELIBRARY_TESTFACTORY_HPP
#define ABLATELIBRARY_TESTFACTORY_HPP

#include "argumentIdentifier.hpp"
#include <iostream>
#include "factory.hpp"

namespace ablate{
namespace parser {

class TestFactory :public Factory {
   public:

    TestFactory(std::string type) : Factory(type){}


    std::string Get(const ArgumentIdentifier<std::string> identifier) override {
        return " hi there";
    }

    //    virtual double GetDouble(const std::string& name) = 0;
    //    virtual int GetInt(const std::string& name) = 0;
    virtual std::shared_ptr<Factory> GetFactory(const std::string& name) override {
        return std::make_shared<TestFactory>(name == "talky" ? "item" : "Blue");
    }

};

}
}

#endif  // ABLATELIBRARY_FACTORY_HPP