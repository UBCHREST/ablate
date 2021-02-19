//
// Created by Matt McGurn on 2/18/21.
//

#include "item.hpp"
#include <iostream>
#include "registrar.hpp"

static bool item_s_registered = ablate::parser::Registrar<ITalky>::Register<Item>("item", "all about the item");

void Item::Talk() {
    std::cout << "I'm talking here" << content << std::endl;
}
