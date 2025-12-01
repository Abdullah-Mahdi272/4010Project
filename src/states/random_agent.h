#pragma once
#include <random>

//possible actions
struct RandomAction {
    bool accel = false; 
    bool left  = false;
    bool right = false;
    bool brake = false;
    bool drift = false;
    bool item  = false;
};

class RandomAgent {
public:
    RandomAgent();
    //chooses random actions each frame
    void update(float dt);          
    const RandomAction& action() const { return act; }

private:
    //random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    RandomAction act;
};