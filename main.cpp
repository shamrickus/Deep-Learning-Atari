#include <cstdio>
#include "SDL.h"
#include "ale/ale_interface.hpp"

int main() {
    ALEInterface ale;

    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", true);
    ale.loadROM("roms/Breakout.a26");

    auto actions = ale.getMinimalActionSet();

    while(!ale.game_over()){
        ale.act(actions[rand() % actions.size()]);
    }
}
