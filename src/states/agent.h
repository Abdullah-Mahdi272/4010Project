#pragma once
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>
#include <map>
#include <string>
"""
Class representing our agent in the mario kart environment and its methods

"""
class Agent {
public:
    Agent();
    void updatePosition(float x, float y);
    void updateSpeed(float forward, float turn);
    void updateAngle(float angle);
    float getSpeedForward();
    float getSpeedTurn();
    float getAngle();
    void render();
    int getNextAction();

    void reset();

    std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> step(int);
    void setTerminated(bool t);
    void setTruncated(bool t);

private:
    float positionX;
    float positionY;
    float speedForward;
    float speedTurn;
    float angle;

    float prevPositionX;
    float prevPositionY;
    float prevSpeedForward;
    float prevSpeedTurn;
    float prevAngle;

    bool terminated;
    bool truncated;
};