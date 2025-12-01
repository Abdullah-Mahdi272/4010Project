#include "initload.h"

sf::Texture StateInitLoad::shadowTexture;

const sf::Time StateInitLoad::DING_TIME = sf::seconds(0.75f);
const sf::Time StateInitLoad::END_TIME = sf::seconds(2.0f);

void StateInitLoad::loadAllGameTextures() {
    // Floor objects
    Zipper::loadAssets("assets/objects/floor/misc.png",
                       sf::IntRect(sf::Vector2i(0, 0), sf::Vector2i(16, 16)));
    QuestionPanel::loadAssets(
        "assets/objects/floor/question_panel.png",
        sf::IntRect(sf::Vector2i(0, 0), sf::Vector2i(16, 16)),
        sf::IntRect(sf::Vector2i(0, 16), sf::Vector2i(16, 16)));
    OilSlick::loadAssets(
        "assets/objects/floor/misc.png",
        sf::IntRect(sf::Vector2i(0, 16), sf::Vector2i(16, 16)));
    Coin::loadAssets("assets/objects/floor/misc.png",
                     sf::IntRect(sf::Vector2i(0, 32), sf::Vector2i(8, 8)));
    RampHorizontal::loadAssets(
        "assets/objects/floor/misc.png",
        sf::IntRect(sf::Vector2i(0, 40), sf::Vector2i(8, 8)));
    RampVertical::loadAssets(
        "assets/objects/floor/misc.png",
        sf::IntRect(sf::Vector2i(32, 40), sf::Vector2i(8, 8)));

    // Wall objects
    WallObject::loadAssets("assets/misc/shadow.png");
    Podium::loadAssets("assets/misc/congratulations.png",
                       sf::IntRect(417, 153, 104, 30));
    FloatingFish::loadAssets(
        "assets/misc/congratulations.png", sf::IntRect(2, 51, 90, 95),
        sf::IntRect(2, 148, 93, 95), sf::IntRect(508, 51, 90, 95),
        sf::IntRect(505, 148, 93, 95));
    Pipe::loadAssets("assets/objects/wall/misc.png",
                     sf::IntRect(sf::Vector2i(2, 53), sf::Vector2i(24, 32)),
                     sf::IntRect(sf::Vector2i(158, 53), sf::Vector2i(24, 32)));
    Thwomp::loadAssets(
        "assets/objects/wall/misc.png",
        sf::IntRect(sf::Vector2i(2, 20), sf::Vector2i(24, 32)),
        sf::IntRect(sf::Vector2i(158, 20), sf::Vector2i(24, 32)));

    Banana::loadAssets("assets/objects/wall/misc.png",
                       sf::IntRect(sf::Vector2i(2, 129), sf::Vector2i(16, 16)));

    GreenShell::loadAssets(
        "assets/objects/wall/misc.png",
        sf::IntRect(sf::Vector2i(83, 129), sf::Vector2i(16, 16)));

    RedShell::loadAssets(
        "assets/objects/wall/misc.png",
        sf::IntRect(sf::Vector2i(172, 129), sf::Vector2i(16, 16)));

    EffectCoin::loadAssets("assets/misc/coin.png", sf::IntRect(0, 0, 16, 16),
                           sf::IntRect(16, 0, 16, 16),
                           sf::IntRect(32, 0, 16, 16));
    EffectDrown::loadAssets(
        "assets/misc/particles.png", sf::IntRect(96, 61, 8, 16),
        sf::IntRect(105, 53, 24, 24), sf::IntRect(130, 61, 8, 16),
        sf::IntRect(79, 61, 16, 8), sf::IntRect(79, 52, 16, 8));
    EffectSparkles::loadAssets(
        "assets/misc/particles.png", sf::IntRect(1, 85, 24, 16),
        sf::IntRect(26, 85, 24, 16), sf::IntRect(51, 85, 24, 16),
        sf::IntRect(76, 85, 24, 16));

    // Audio/music assets
    Audio::loadAll();

    // Other menu assets
    EndRanks::loadAssets("assets/gui/ranking.png", sf::IntRect(1, 1, 14, 16),
                         sf::IntRect(18, 1, 14, 16), sf::IntRect(35, 1, 14, 16),
                         17, sf::IntRect(52, 1, 8, 16),
                         sf::IntRect(52, 18, 8, 16), sf::IntRect(52, 35, 8, 16),
                         9);

    TextUtils::loadAssets("assets/gui/letters.png",
                          "assets/gui/letters_alpha.png", sf::Vector2i(1, 1),
                          sf::Vector2i(1, 32));

    StateStart::loadBackgroundAssets("assets/menu/start/background.png",
                                     sf::IntRect(246, 16, 512, 224),
                                     sf::IntRect(6, 16, 234, 76));

    StatePlayerSelection::loadAssets(
        "assets/gui/player_selection.png", sf::IntRect(0, 0, 256, 224),
        sf::IntRect(281, 146, 38, 35), sf::IntRect(272, 24, 256, 16),
        sf::IntRect(352, 57, 16, 8), sf::IntRect(376, 48, 34, 10),
        sf::IntRect(376, 59, 34, 10));

    StateGPStandings::loadAssets("assets/menu/start/background.png",
                                 sf::IntRect(764, 16, 512, 256),
                                 sf::IntRect(887, 301, 256, 224));

    StateCongratulations::loadAssets(
        "assets/misc/congratulations.png", sf::IntRect(410, 2, 141, 14), 15,
        sf::IntRect(553, 2, 10, 15), sf::IntRect(564, 2, 12, 15),
        sf::IntRect(577, 2, 12, 15), 16, sf::IntRect(418, 192, 16, 16),
        sf::IntRect(418, 209, 16, 16), sf::IntRect(418, 227, 16, 16), 20,
        sf::IntRect(65, 2, 62, 22));
}

void StateInitLoad::init() {
    currentTime = sf::Time::Zero;

    // AGGRESSIVE DEBUG OUTPUT
    std::cerr << "========================================" << std::endl;
    std::cerr << "INITLOAD: Checking for RL_MODE..." << std::endl;
    std::cerr.flush();
    
    // Check for RL mode - WITH DEBUG OUTPUT
    const char* rl_env = std::getenv("RL_MODE");
    
    std::cerr << "INITLOAD: getenv(\"RL_MODE\") returned: " 
              << (rl_env ? rl_env : "NULL") << std::endl;
    std::cerr.flush();
    
    rlMode = (rl_env && std::string(rl_env) == "1");
    
    std::cerr << "INITLOAD: rlMode = " << (rlMode ? "TRUE" : "FALSE") << std::endl;
    std::cerr.flush();

    if (rlMode) {
        // OUTPUT TO BOTH STDOUT AND STDERR
        std::cout << "=== RL MODE DETECTED ===" << std::endl;
        std::cerr << "=== RL MODE DETECTED ===" << std::endl;
        std::cout.flush();
        std::cerr.flush();
        
        std::cout << "Skipping menus and loading directly into race..." << std::endl;
        std::cerr << "Skipping menus and loading directly into race..." << std::endl;
        std::cout.flush();
        std::cerr.flush();
        
        // Load the track
        Map::loadCourse(CIRCUIT_ASSET_NAMES[(unsigned int)RaceCircuit::MARIO_CIRCUIT_2]);
        Map::loadAI();
        
        // Set up race with 50cc
        CCOption selectedCC = CCOption::CC50;
        StateRace::ccOption = selectedCC;
        
        // Create drivers and positions
        VehicleProperties::setScaleFactor(1.0f, 1.0f);
        
        unsigned int modifiersIndexer[(unsigned int)MenuPlayer::__COUNT] = {0, 1, 2, 3, 4, 5, 6, 7};
        std::shuffle(std::begin(modifiersIndexer), std::end(modifiersIndexer), randGen);
        
        for (unsigned int i = 0; i < (unsigned int)MenuPlayer::__COUNT; i++) {
            DriverPtr driver(new Driver(
                DRIVER_ASSET_NAMES[i].c_str(), sf::Vector2f(0.0f, 0.0f),
                M_PI_2 * -1.0f, MAP_ASSETS_WIDTH, MAP_ASSETS_HEIGHT,
                DriverControlType::DISABLED, *DRIVER_PROPERTIES[i], MenuPlayer(i), rlPositions, false,
                FAR_VISIONS[(int)selectedCC][modifiersIndexer[i]],
                ITEM_PROB_MODS[(int)selectedCC][modifiersIndexer[i]],
                IMPEDIMENTS[(int)selectedCC][modifiersIndexer[i]]
            ));
            rlDrivers[i] = driver;
            rlPositions[i] = driver.get();
        }
        
        // Select player (e.g., MARIO as RL agent)
        MenuPlayer selectedPlayer = MenuPlayer::MARIO;
        
        // Apply player character multiplier to player vehicle
        rlDrivers[(int)selectedPlayer]->vehicle = &rlDrivers[(int)selectedPlayer]->vehicle->makePlayer();
        
        // Set all to AI_GRADIENT
        for (int i = 0; i < (int)MenuPlayer::__COUNT; i++) {
            rlDrivers[i]->controlType = DriverControlType::AI_GRADIENT;
        }
        
        // Move selected player to last position
        std::swap(rlPositions[(int)selectedPlayer], rlPositions[(int)MenuPlayer::__COUNT - 1]);
        
        // Shuffle first 7
        std::random_shuffle(rlPositions.begin(), rlPositions.begin() + ((int)MenuPlayer::__COUNT - 1));
        
        // Set player control
        rlDrivers[(int)selectedPlayer]->controlType = DriverControlType::PLAYER;
        
        // Improved reset before race
        Lakitu::reset();
        Gui::reset(false);
        Gui::resetSplits();
        EndRanks::reset(&rlPositions);
        StateRace::currentTime = sf::Time::Zero;
        for (unsigned int i = 0; i < rlPositions.size(); i++) {
            sf::Vector2f pos = Map::getPlayerInitialPosition(i + 1);
            rlPositions[i]->setPositionAndReset(sf::Vector2f(pos.x / MAP_ASSETS_WIDTH, pos.y / MAP_ASSETS_HEIGHT));
        }
        
        // Set real player
        Driver::realPlayer = rlDrivers[(int)selectedPlayer];
        
        std::cerr << "INITLOAD: About to push StateRace..." << std::endl;
        std::cerr.flush();
        
        // Push StateRace directly
        game.pushState(StatePtr(new StateRace(game, rlDrivers[(int)selectedPlayer], rlDrivers, rlPositions)));
        
        std::cerr << "INITLOAD: StateRace pushed successfully" << std::endl;
        std::cerr.flush();
        
        dingPlayed = true;
        return;
    }

    std::cerr << "INITLOAD: Normal mode - loading Nintendo logo" << std::endl;
    std::cerr.flush();

    nintendoLogoTexture.loadFromFile("assets/gui/nintendo_logo.png");

    audioDingId = Audio::loadDing();
    StateInitLoad::loadAllGameTextures();
}

bool StateInitLoad::update(const sf::Time& deltaTime) {
    currentTime += deltaTime;
    
    // RL mode: skip all animations and go straight to race
    if (rlMode && !dingPlayed) {
        // The code is already handled in init()
        return true;
    }
    
    // Normal mode: show Nintendo logo animation
    if (!dingPlayed) {
        Audio::play(audioDingId);
        dingPlayed = true;
    } else if (currentTime >= END_TIME) {
        game.pushState(StatePtr(new StateStart(game)));
    }

    return true;
}

void StateInitLoad::draw(sf::RenderTarget& window) {
    // In RL mode, just show black screen (race will start immediately)
    if (rlMode) {
        window.clear(sf::Color::Black);
        return;
    }
    
    // Normal mode: show Nintendo logo
    window.clear(sf::Color::Black);
    if (currentTime < END_TIME) {
        sf::Sprite nintendoLogo(nintendoLogoTexture);
        float scale =
            window.getSize().x / (float)nintendoLogoTexture.getSize().x;
        nintendoLogo.scale(scale, scale);
        if (currentTime >= DING_TIME) {
            // reduce opacity
            float pct = (currentTime - DING_TIME) / (END_TIME - DING_TIME);
            int opacity = std::max(255 * (1.00f - pct), 0.0f);
            nintendoLogo.setColor(sf::Color(255, 255, 255, opacity));
        }
        window.draw(nintendoLogo);
    }
}
