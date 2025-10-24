#include "states/agent.h"

Agent::Agent()
    : positionX(0.0f), positionY(0.0f),
      speedForward(0.0f), speedTurn(0.0f) {}

void Agent::updatePosition(float x, float y) {
    positionX = x;
    positionY = y;
}

void Agent::updateSpeed(float forward, float turn) {
	speedForward = forward;
    speedTurn = turn;
	// std::cout<< "Agent Speed Update - Forward: " << forward << ", Turn: " << turn << std::endl;
}

float Agent::getSpeedForward() const { return speedForward; }
float Agent::getSpeedTurn() const { return speedTurn; }
