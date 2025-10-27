#include "gui.h"
#include <iostream>
using namespace std;

Gui Gui::instance;
Gui::Gui() { winSize = sf::Vector2u(0, 0); }

void Gui::setWindowSize(sf::Vector2u s) {
    instance.winSize = s;
    instance.timer.setWindowSize(s);
    instance.itemInd.setPosition(s, instance.timer.getItemPos());
    instance.others.setWindowSize(s);
    instance.effects.setWindowSize(s);
    instance.splitTimer.setWindowSize(s);
}

void Gui::setPowerUp(PowerUps power) { instance.itemInd.setItem(power); }

void Gui::addCoin(int ammount) { instance.others.addCoin(ammount); }

void Gui::setRanking(int r) { instance.others.setRanking(r); }

void Gui::thunder() { instance.effects.thunder(0.2); }

void Gui::speed(float time) { instance.effects.speed(time); }

void Gui::fade(float time, bool fromBlack) {
    instance.effects.blackFade(time, fromBlack);
}

bool Gui::canUseItem() { return !instance.itemInd.spinning; }

bool Gui::isBlackScreen(bool total) {
    if (total)
        return instance.effects.blackScreen.getFillColor().a > 252;
    else
        return instance.effects.blackScreen.getFillColor().a > 1;
}

void Gui::update(const sf::Time &deltaTime) {
    instance.timer.update(deltaTime);
    instance.itemInd.update(deltaTime);
    instance.others.update(deltaTime);
    instance.effects.update(deltaTime);
}

void Gui::draw(sf::RenderTarget &window, const sf::Color &timerColor) {
    instance.effects.draw(window);
    instance.timer.draw(window, timerColor);
    instance.itemInd.draw(window);
    instance.others.draw(window);
    instance.splitTimer.draw(window);
}

void Gui::endRace() { instance.others.setRanking(instance.others.rank, true); }

void Gui::reset(bool rankReset) {
    instance.timer.reset();
    instance.others.reset(rankReset);
    instance.itemInd.reset();
    instance.effects.reset();
    instance.splitTimer.reset();
}

void Gui::stopEffects() { instance.effects.stop(); }

void Gui::initializeSplits(int maxGradient) {
    instance.splitTimer.initializeCheckpoints(maxGradient);
}

void Gui::updateSplits(const sf::Time &currentRaceTime, int currentGradient, 
                      int currentLap, const sf::Time &deltaTime,
                      const sf::Vector2f &position, float speed, 
                      float turnSpeed, float angle) {
    instance.splitTimer.update(currentRaceTime, currentGradient, currentLap, 
                              deltaTime, position, speed, turnSpeed, angle);
}

void Gui::resetSplits() {
    instance.splitTimer.reset();
}
