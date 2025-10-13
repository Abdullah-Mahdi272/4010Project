#pragma once
#include <iostream>

class Agent {
    public:
	    Agent();
        void doNothing(int x){std::cout<<x<<std::endl;};
	    void updatePosition(float x,float y);
// 	void updateRanking(int ranking);
// 	int getPositionX() const;
// 	int getPositionY() const;

    private:
        float positionX;
        float positionY;
        // int ranking;
};