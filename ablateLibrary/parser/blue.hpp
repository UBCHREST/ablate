//
// Created by Matt McGurn on 2/18/21.
//

#ifndef FACTORYPLAYGROUND_ITEM_HPP
#define FACTORYPLAYGROUND_ITEM_HPP
#include "ITalky.hpp"
#include <memory>
#include <string>
#include "factory.hpp"
#include "argumentIdentifier.hpp"

class Blue : public ITalky{
private:
    const std::string content;
    const std::shared_ptr<ITalky> talky;
public:
    Blue(std::string content, std::shared_ptr<ITalky> talky) : content(content), talky(talky){

    }
    void Talk();
};


#endif //FACTORYPLAYGROUND_ITEM_HPP
