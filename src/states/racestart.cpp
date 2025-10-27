#define _USE_MATH_DEFINES
#include "racestart.h"

#include "map/map.h"

const sf::Time StateRaceStart::ANIMATION_FORWARD_TIME = sf::Time::Zero;
const sf::Time StateRaceStart::ANIMATION_TURN_TIME = sf::Time::Zero;
const float StateRaceStart::PROB_HIT_BY_CC[(int)CCOption::__COUNT] = {
    0.95, 0.975, 1.0};

void StateRaceStart::asyncLoad() {
    // assumes that course (map) has finished loading
    Map::loadAI();
    asyncLoadFinished = true;
}

void StateRaceStart::init(const sf::Vector2f& _playerPosition) {
    currentTime = sf::Time::Zero;
    accTime = sf::Time::Zero;
    playerPosition = sf::Vector2f(_playerPosition.x / MAP_ASSETS_WIDTH,
                                  _playerPosition.y / MAP_ASSETS_HEIGHT);
    pseudoPlayer = DriverPtr(new Driver(
        "assets/drivers/invisible.png",
        playerPosition + sf::Vector2f(0.0f, ANIMATION_FORWARD_DISTANCE * -1.0f),
        M_PI * -0.5f, MAP_TILES_WIDTH, MAP_TILES_HEIGHT,
        DriverControlType::DISABLED, VehicleProperties::ACCELERATION,
        MenuPlayer(1)));
    pseudoPlayer->setPositionAndReset(playerPosition);

    asyncLoadFinished = false;
    fadingMusic = false;
    loadingThread = std::thread(&StateRaceStart::asyncLoad, this);

    // OUTPUT EPISODE START FOR RL
    std::cout << "EPISODE_START" << std::endl;
}

void StateRaceStart::handleEvent(const sf::Event& event) {
    // REMOVED: No input skip for instant start
    if (Input::pressed(Key::PAUSE, event) && !pushedPauseThisFrame) {
        pushedPauseThisFrame = true;
        // call draw and store so we can draw it over the screen
        sf::RenderTexture render;
        sf::Vector2u windowSize = game.getWindow().getSize();
        render.create(windowSize.x, windowSize.y);
        fixedUpdate(sf::Time::Zero);
        draw(render);
        game.pushState(StatePtr(new StateRacePause(game, render)));
    }
}

bool StateRaceStart::update(const sf::Time& deltaTime) {
    currentTime += deltaTime;
    pushedPauseThisFrame = false;

    Lakitu::update(deltaTime);
    pseudoPlayer->position = playerPosition;
    pseudoPlayer->posAngle = M_PI * -0.5f;  // Instant position/angle

    if (currentTime > ANIMATION_FORWARD_TIME && !fadingMusic) {
        fadingMusic = true;
        // Audio::fadeOutMusic(ANIMATION_TURN_TIME - ANIMATION_FORWARD_TIME);
    }
    if (asyncLoadFinished) {
        game.popState();
    }
    if (loadingThread.joinable()) {
        loadingThread.join();
    }
    return true;
}

void StateRaceStart::draw(sf::RenderTarget& window) {
    // scale
    static constexpr const float NORMAL_WIDTH = 512.0f;
    const float scale = window.getSize().x / NORMAL_WIDTH;
    // Get textures from map
    sf::Texture tSkyBack, tSkyFront, tCircuit, tMap;
    Map::skyTextures(pseudoPlayer, tSkyBack, tSkyFront);
    Map::circuitTexture(pseudoPlayer, tCircuit);
    Map::mapTexture(tMap);
    // Create sprites and scale them accordingly
    sf::Sprite skyBack(tSkyBack), skyFront(tSkyFront), circuit(tCircuit),
        map(tMap);
    sf::Vector2u windowSize = game.getWindow().getSize();
    float backFactor = windowSize.x / (float)tSkyBack.getSize().x;
    float frontFactor = windowSize.x / (float)tSkyFront.getSize().x;
    skyBack.setScale(backFactor, backFactor);
    skyFront.setScale(frontFactor, frontFactor);
    // Sort them correctly in Y position and draw
    float currentHeight = 0;
    skyBack.setPosition(0.0f, currentHeight);
    skyFront.setPosition(0.0f, currentHeight);
    window.draw(skyBack);
    window.draw(skyFront);
    currentHeight += windowSize.y * Map::SKY_HEIGHT_PCT;
    circuit.setPosition(0.0f, currentHeight);
    window.draw(circuit);
    // Lakitu shadow
    Lakitu::drawShadow(window);
    // Circuit objects (must be before minimap)
    std::vector<std::pair<float, sf::Sprite*>> wallObjects;
    Map::getWallDrawables(window, pseudoPlayer, scale, wallObjects);
    Map::getDriverDrawables(window, pseudoPlayer, drivers, scale, wallObjects);
    std::sort(wallObjects.begin(), wallObjects.end(),
              [](const std::pair<float, sf::Sprite*>& lhs,
                 const std::pair<float, sf::Sprite*>& rhs) {
                  return lhs.first > rhs.first;
              });
    for (const auto& pair : wallObjects) {
        window.draw(*pair.second);
    }
    // Minimap
    currentHeight += windowSize.y * Map::CIRCUIT_HEIGHT_PCT;
    map.setPosition(0.0f, currentHeight);
    window.draw(map);
    // Minimap drivers
    std::sort(drivers.begin(), drivers.end(),
              [](const DriverPtr& lhs, const DriverPtr& rhs) {
                  return lhs->position.y < rhs->position.y;
              });
    for (const DriverPtr& driver : drivers) {
        sf::Sprite miniDriver =
            driver->animator.getMinimapSprite(M_PI * -0.5f, scale);
        sf::Vector2f mapPosition = Map::mapCoordinates(driver->position);
        miniDriver.setOrigin(miniDriver.getLocalBounds().width / 2.0f,
                             miniDriver.getLocalBounds().height * 0.9f);
        miniDriver.setPosition(mapPosition.x * windowSize.x,
                               mapPosition.y * windowSize.y);
        miniDriver.scale(0.5f, 0.5f);
        window.draw(miniDriver);
    }
    Lakitu::draw(window);
    Gui::draw(window);

}
