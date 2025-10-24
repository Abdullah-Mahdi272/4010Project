#pragma once
#include <iostream>

class Agent {
public:
    Agent();
    // void doNothing(int x){std::cout<<x<<std::endl;};
    void updatePosition(float x,float y);
    void updateSpeed(float forward, float turn);
    float getSpeedForward() const;
    float getSpeedTurn() const;

private:
    float positionX;
    float positionY;
    float speedForward;
    float speedTurn;
};