#include "random_agent.h"

RandomAgent::RandomAgent()
    : rng(std::random_device{}()), dist(0.0f, 1.0f) {}

void RandomAgent::update(float dt) {
    //probabilites of each action

    //move forward and backwards
    act.accel = dist(rng) < 0.5f;
    act.brake = dist(rng) < 0.5f;
    
    //turn left and right
    act.left  = dist(rng) < 0.5f;
    act.right = dist(rng) < 0.5f;

    //drift and item
    act.drift = dist(rng) < 0.5f;
    act.item  = dist(rng) < 0.5f;
}
    
