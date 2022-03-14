//
// Created by owen on 3/14/22.
//
#include<radiation.cpp>
#include<iostream>

int main() {
    Irradiate cell(10, 20, 30);
    cell.rayTrace();
    std::cout << "Testing radiation solver:";
    std::cout << cell.radGain;
    return 0;
}