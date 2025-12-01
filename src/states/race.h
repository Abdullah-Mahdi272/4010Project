#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>

#include "ai/gradientdescent.h"
#include "entities/collisionhashmap.h"
#include "entities/driver.h"
#include "entities/item.h"
#include "entities/lakitu.h"
#include "gui/endranks.h"
#include "gui/gui.h"
#include "map/enums.h"
#include "map/map.h"
#include "states/racepause.h"
#include "states/agent.h"
#include "states/statebase.h"

#include "random_agent.h"
#include "MonteCarloAgent.h"





struct PlayerInfo{
  float progress;
  float speed;
  float angle;
  int rank;
  int lap;
  bool onGround;
  bool canDrive;
};

class StateRace : public State {
  private:
    static const sf::Time TIME_BETWEEN_ITEM_CHECKS;
    sf::Time nextItemCheck;

    const DriverPtr player;
    DriverArray drivers;
    DriverArray miniDrivers;
    RaceRankingArray& positions;

    bool pushedPauseThisFrame = false;
    bool raceFinished = false;
    bool driftPressed = false;
    bool splitsInitialized = false;

    //random 
    bool random_enabled = false;
    std::unique_ptr<RandomAgent> randomAI;

    //monte carlo
    bool mc_enabled = false;
    double lastProgress = 0.0;
    float stuckTimer = 0.0f;
    
    static std::unique_ptr<MonteCarloAgent> mcAgent;

    double lastRewardProgress = 0.0;
    bool rewardInit = false;
    bool collidedThisFrame = false;
    int noProgressFrames = 0;
    
    float maxEpisodeTimeSecs = 15.0f;
    
    //end of monte carlo

    // if player is in last place and all 7 AI finish, give some seconds to
    // the player and after that finish the game
    static const sf::Time WAIT_FOR_PC_LAST_PLACE;
    sf::Time waitForPCTime;

   public:
    PlayerInfo getPlayerInfo(const DriverPtr& driver);
    static sf::Time currentTime;
    static CCOption ccOption;
    Agent* agent = nullptr;

    StateRace(Game& game, const DriverPtr& _player, const DriverArray& _drivers,
              RaceRankingArray& _positions)
        : State(game),
          player(_player),
          drivers(_drivers),
          miniDrivers(_drivers),
          positions(_positions) {
        init();
    }

    ~StateRace(); // <-- added destructor to free agent

    void handleEvent(const sf::Event& event) override;
    bool fixedUpdate(const sf::Time& deltaTime) override;
    void draw(sf::RenderTarget& window) override;

    void init();

    inline std::string string() const override { return "Race"; }

    
};

