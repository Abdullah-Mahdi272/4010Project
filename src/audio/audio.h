#pragma once

#include <SFML/System.hpp>
#include <array>
#include <string>

#include "entities/enums.h"

enum class Music : int {
    MENU_TITLE_SCREEN,
    MENU_PLAYER_CIRCUIT,
    CIRCUIT_ANIMATION_START,
    CIRCUIT_NORMAL,
    CIRCUIT_LAST_LAP,
    CONGRATULATIONS_SCREEN,
    CIRCUIT_PLAYER_WIN,
    CIRCUIT_PLAYER_LOSE,
    __COUNT,
};

enum class SFX : int {
    MENU_INTRO_SCREEN_DING,
    MENU_SELECTION_ACCEPT,
    MENU_SELECTION_CANCEL,
    MENU_SELECTION_MOVE,
    CIRCUIT_LAKITU_SEMAPHORE,
    CIRCUIT_LAKITU_WARNING,
    CIRCUIT_COLLISION,
    CIRCUIT_COLLISION_PIPE,
    CIRCUIT_PASS_MOTOR,
    CIRCUIT_OVERTAKE_UP,
    CIRCUIT_OVERTAKE_DOWN,
    CIRCUIT_LAST_LAP_NOTICE,
    CIRCUIT_GOAL_END,
    CIRCUIT_END_VICTORY,
    CIRCUIT_END_DEFEAT,
    CIRCUIT_PLAYER_MOTOR,
    CIRCUIT_PLAYER_MOTOR_SPOOK,
    CIRCUIT_PLAYER_BRAKE,
    CIRCUIT_PLAYER_DRIFT,
    CIRCUIT_PLAYER_DRIFT_SPOOK,
    CIRCUIT_MATERIAL_GRASS,
    CIRCUIT_MATERIAL_WOOD,
    CIRCUIT_MATERIAL_SPOOK,
    CIRCUIT_PLAYER_JUMP,
    CIRCUIT_PLAYER_LANDING,
    CIRCUIT_PLAYER_FALL,
    CIRCUIT_PLAYER_FALL_WATER,
    CIRCUIT_PLAYER_FALL_LAVA,
    CIRCUIT_PLAYER_HIT,
    CIRCUIT_PLAYER_SMASH,
    CIRCUIT_PLAYER_GROW,
    CIRCUIT_PLAYER_SHRINK,
    CIRCUIT_COIN,
    CIRCUIT_ITEM_RANDOMIZING,
    CIRCUIT_ITEM_GET,
    CIRCUIT_ITEM_USE_LAUNCH,
    CIRCUIT_ITEM_USE_UP,
    CIRCUIT_ITEM_USE_DOWN,
    CIRCUIT_ITEM_COIN,
    CIRCUIT_ITEM_STAR,
    CIRCUIT_ITEM_MUSHROOM,
    CIRCUIT_ITEM_THUNDER,
    CIRCUIT_ITEM_RED_SHELL,
    RESULTS_POINTS_UPDATE,
    __COUNT,
};

class Audio {
   private:
    static constexpr const float VOLUME_MULTIPLIER = 0.8f;
    static constexpr const float VOLUME_LOG_EXP = 1.0f;
    
    std::array<int, (int)SFX::__COUNT> sfxLastIndex = {-1};
    unsigned int playerIndex = 0;
    bool raceMode = false;
    bool enginesPlaying = false;

    static Audio instance;
    float musicVolumePct, sfxVolumePct;
    float getMusicValue, getSFXValue;

    Audio() {
        musicVolumePct = logFunc(0.5f) * 100.0;
        sfxVolumePct = logFunc(0.5f) * 100.0;
        getMusicValue = 0.5f;
        getSFXValue = 0.5f;
    }
    
    static SFX loadDing();
    static void loadAll();
    static float logFunc(const float value);

    void load(const Music music, const std::string &filename);
    void load(const SFX sfx, const std::string &filename);

    friend class StateInitLoad;

   public:
    static void loadCircuit(const std::string &folder);
    static void play(const Music music, bool loop = true);
    static void play(const SFX sfx, bool loop = false);

    static bool isPlaying(const SFX sfx);

    static void fadeOut(const Music music, const sf::Time &deltaTime,
                        const sf::Time &time = sf::seconds(2.0f));

    static void pauseMusic();
    static void pauseSFX();

    static void resumeMusic();
    static void resumeSFX();

    static void stopSFX();
    static void stop(const SFX sfx);

    static void stopMusic();

    static void setVolume(const float musicVolumePct, const float sfxVolumePct);
    static float getMusicVolume() {
        return instance.getMusicValue;
    }
    static float getSfxVolume() {
        return instance.getSFXValue;
    }

    static void setPitch(const SFX sfx, const float sfxPitch);

    static void playEngines(unsigned int playerIndex, bool raceMode = true);
    static void playEngines(bool playerOnly = false);
    static void setEngineVolume(unsigned int i, float volume = 100.0f);
    static void setEnginesVolume(float volume = 100.0f);
    static void updateEngine(unsigned int i, sf::Vector2f position,
                             float height, float speedForward, float speedTurn);
    static void updateEngine(sf::Vector2f position, float height,
                             float speedForward, float speedTurn);
    static void updateListener(sf::Vector2f position, float angle,
                               float height);
    static void pauseEngines();
    static void resumeEngines();
    static void stopEngines();
};
