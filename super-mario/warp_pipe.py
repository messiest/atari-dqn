import os
import random

import gym
import gym_super_mario_bros as mario


class WarpPipe(object):
    def __init__(self, world, stage, version=2):
        assert world <= 8 and world > 0, f"Level {world}-{stage} does not exist"
        assert stage <= 4 and stage > 0, f"Level {world}-{stage} does not exist"
        self.__ver__ = version
        self.world = world
        self.stage = stage

        self.env = mario.make(
            f'SuperMarioBros-{self.world}-{self.stage}-v{self.__ver__}'
        )
        self.env.reset()

    def next(self):
        if self.stage = 4:
            print("I'm sorry Mario, but Peach isn't in this castle...")
            self.world += 1
            self.stage = 0
        else:
            self.stage += 1

        self.env = mario.make(
            f'SuperMarioBros-{self.world}-{self.stage}-v{self.__ver__}'
        )
        self.env.reset()

    def warp(self, world, stage, version=self.__ver__):
        assert world <= 8 and world > 0, f"Level {world}-{stage} does not exist"
        assert stage <= 4 and stage > 0, f"Level {world}-{stage} does not exist"
        self.__ver__ = version
        self.world = world
        self.stage = stage
        self.env = mario.make(
            f'SuperMarioBros-{self.world}-{self.stage}-v{self.__ver__}'
        )
        self.env.reset()
