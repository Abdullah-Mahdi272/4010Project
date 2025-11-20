#include "audio.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "map/enums.h"

// Static instance
Audio Audio::instance;

// Global flag to disable all audio operations
static bool g_audioDisabled = true;

// Stub implementations - all audio functions do nothing

SFX Audio::loadDing() {
    std::cout << "Audio system bypassed - running without sound" << std::endl;
    return SFX::MENU_INTRO_SCREEN_DING;
}

void Audio::loadAll() {
    // Do nothing
}

void Audio::loadCircuit(const std::string &folder) {
    // Do nothing
}

void Audio::load(const Music music, const std::string &filename) {
    // Do nothing
}

void Audio::load(const SFX sfx, const std::string &filename) {
    // Do nothing
}

void Audio::play(const Music music, bool loop) {
    // Do nothing
}

void Audio::play(const SFX sfx, bool loop) {
    // Do nothing
}

bool Audio::isPlaying(const SFX sfx) {
    return false;
}

void Audio::stop(const SFX sfx) {
    // Do nothing
}

void Audio::fadeOut(const Music music, const sf::Time &deltaTime,
                    const sf::Time &time) {
    // Do nothing
}

void Audio::pauseMusic() {
    // Do nothing
}

void Audio::pauseSFX() {
    // Do nothing
}

void Audio::resumeMusic() {
    // Do nothing
}

void Audio::resumeSFX() {
    // Do nothing
}

void Audio::stopSFX() {
    // Do nothing
}

void Audio::stopMusic() {
    // Do nothing
}

float Audio::logFunc(const float value) {
    float ret = -log10f(powf(1 - value * 0.9f, VOLUME_LOG_EXP));
    if (ret > 1.0f) {
        ret = 1.0f;
    }
    return ret;
}

void Audio::setVolume(const float musicVolumePct, const float sfxVolumePct) {
    instance.getMusicValue = musicVolumePct;
    instance.getSFXValue = sfxVolumePct;
    instance.musicVolumePct =
        logFunc(musicVolumePct) * 100.0f * VOLUME_MULTIPLIER;
    instance.sfxVolumePct = logFunc(sfxVolumePct) * 100.0f * VOLUME_MULTIPLIER;
}

void Audio::setPitch(const SFX sfx, const float sfxPitch) {
    // Do nothing
}

void Audio::playEngines(unsigned int playerIndex, bool raceMode) {
    instance.playerIndex = playerIndex;
    instance.raceMode = raceMode;
    instance.enginesPlaying = true;
}

void Audio::playEngines(bool playerOnly) {
    playEngines(instance.playerIndex, playerOnly);
}

void Audio::setEngineVolume(unsigned int i, float volume) {
    // Do nothing
}

void Audio::setEnginesVolume(float volume) {
    // Do nothing
}

void Audio::updateEngine(unsigned int i, sf::Vector2f position, float height,
                         float speedForward, float speedTurn) {
    // Do nothing
}

void Audio::updateEngine(sf::Vector2f position, float height,
                         float speedForward, float speedTurn) {
    // Do nothing
}

void Audio::updateListener(sf::Vector2f position, float angle, float height) {
    // Do nothing
}

void Audio::pauseEngines() {
    // Do nothing
}

void Audio::resumeEngines() {
    // Do nothing
}

void Audio::stopEngines() {
    instance.enginesPlaying = false;
}
