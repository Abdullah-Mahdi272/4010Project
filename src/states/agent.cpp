
#include "agent.h"

Agent::Agent(){positionX = 0; positionY = 0;}

void Agent::updatePosition(float x, float y) {
	positionX = x;
	positionY = y;
    std::cout << "Agent position updated to: (" << positionX << ", " << positionY << ")\n" << x << "," << y <<std::endl;
}

// void Agent::updateRanking(int r) {
// 	ranking = r;
// }

// int Agent::getPositionX() const {
// 	return positionX;
// }

// int Agent::getPositionY() const {
// 	return positionY;
// }
