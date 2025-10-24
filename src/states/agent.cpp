#include "states/agent.h"

Agent::Agent()
    : positionX(0.0f),
      positionY(0.0f),
      speedForward(0.0f),
      speedTurn(0.0f),
      angle(0.0f) {}

void Agent::updatePosition(float x, float y) {
    positionX = x;
    positionY = y;
}

void Agent::updateSpeed(float forward, float turn) {
    speedForward = forward;
    speedTurn = turn;
}

void Agent::updateAngle(float a) {
	angle = a;
}

float Agent::getSpeedForward() { 
	return speedForward;
}

float Agent::getSpeedTurn() {
	return speedTurn; 
}

float Agent::getAngle(){ 
	return angle; 
}
