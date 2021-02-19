//
// Created by Matt McGurn on 2/18/21.
//

#ifndef FACTORYPLAYGROUND_ITEM_HPP
#define FACTORYPLAYGROUND_ITEM_HPP
#include "ITalky.hpp"
#include <memory>
#include <string>
#include "factory.hpp"

class Item : public ITalky{
private:
    const std::string content;
public:

    Item(ablate::parser::Factory& factory): content(factory.Get(ablate::parser::ArgumentIdentifier<std::string>{.inputName ="whatToSay"})){

    }

    Item(std::string content) : content(content){

    }
    void Talk();
};


#endif //FACTORYPLAYGROUND_ITEM_HPP
