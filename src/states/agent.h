#pragma once
#include <iostream>
#include <cmath>
class Agent {
public:
    Agent();
    void updatePosition(float x, float y);
    void updateSpeed(float forward, float turn);
    void updateAngle(float angle);
    float getSpeedForward();
    float getSpeedTurn();
    float getAngle();

private:
    float positionX;
    float positionY;
    float speedForward;
    float speedTurn;
    float angle;
};