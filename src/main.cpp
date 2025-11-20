#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstdio>

#include "entities/enums.h"
#include "game.h"

int main() {
    // Force unbuffered output for Python subprocess communication
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
#ifndef DEBUG
    srand(time(0));
#endif
    Game game(BASIC_WIDTH / 2.0f, BASIC_HEIGHT / 2.0f, 60);
    game.run();
}
