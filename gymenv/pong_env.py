import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class PongEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }

    def __init__(self):
        self.width = 400
        self.height = 400

        self.ballwidth = 15.0
        self.ballheight = 15.0
        self.cartwidth = 60.0
        self.cartheight = 20.0

        self.cart_change_x = 10
        self.ball_change_x = 5
        self.ball_change_y = 5
        
        self.carttrans = None
        self.balltrans = None
        self.x_threshold = 2.4
        self.scores = 0
        self.collide = False
        
        # cart_x, ball_x, ball_y, x_diff, y_diff
        low = np.array([0,
                        0, 
                        0, 
                        np.finfo(np.float32).min, 
                        np.finfo(np.float32).min], 
                        dtype=np.float32)

        high = np.array([self.width,
                        self.width, 
                        self.height, 
                        np.finfo(np.float32).max, 
                        np.finfo(np.float32).max], 
                        dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        done = False
        x, y = self.carttrans.translation
        bx, by = self.balltrans.translation
        
        if action == 0:
            pass
        if action == 1:
            x = x - self.cart_change_x 
        elif action == 2:
            x = x + self.cart_change_x

        if x <= (self.cartwidth // 2):
            x = (self.cartwidth // 2)
        if x >= (self.width - (self.cartwidth // 2)):
            x = (self.width - (self.cartwidth // 2))

        self.carttrans.set_translation(x,y)

        if by <= -self.ballheight:
            done = True
        elif bx < (x - 50) or bx > (x + 50) and (by < self.cartheight) and self.collide:
            self.collide = False
        elif bx > (x - 50) and bx < (x + 50) and (by >= self.cartheight and  by <= self.cartheight+10):
            self.scores += 1
            self.collide = True

        self.state = (x, bx, by, bx-x, by-y)

        return np.array(self.state), self.scores, done, {}

    def reset(self):
        self.scores = 0
        self.wiewer = None
        self.render(update=False)
        self.state = self.np_random.uniform(low=-100, high=100, size=(5,))

        if self.carttrans and self.balltrans:
            self.carttrans.set_translation(self.width/2, self.cartheight/2)
            rand_x = np.random.randint(self.width/2 - 100, high=self.width/2 + 100, size=1)
            self.balltrans.set_translation(rand_x, 100)

            x, y = self.carttrans.translation
            bx, by = self.balltrans.translation
            self.state = (x, bx, by, bx-x, by-y)
            
        return np.array(self.state)

    def render(self, mode='human', update=True):
        carty = 10  # TOP OF CART
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # cart agent or paddle
            l, r, t, b = -self.cartwidth / 2, self.cartwidth / 2, self.cartheight, 0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform(translation=(self.width//2, self.cartheight//2))
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            # ball
            l, r, t, b = -self.ballwidth / 2, self.ballwidth / 2, self.ballheight, 0
            ball = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            ball.set_color(.9, 0.2, 0)

            rand_x = np.random.randint(self.width/2 - 100, high=self.width/2 + 100, size=1)
            self.balltrans = rendering.Transform(translation=(rand_x, 100))
            ball.add_attr(self.balltrans)
            self.viewer.add_geom(ball)

            # bottom line
            self.track = rendering.Line((0, carty), (self.width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            
        if update:
            x, y = self.carttrans.translation
            bx, by = self.balltrans.translation
            bx = bx + self.ball_change_x
            by = by + self.ball_change_y

            if bx > (x - 50) and bx < (x + 50) and (by >= self.cartheight and by <= self.cartheight+10):
                self.ball_change_y = abs(self.ball_change_y)
            elif bx <= (self.ballwidth // 2):
                bx = (self.ballwidth // 2)
                self.ball_change_x = self.ball_change_x * -1
            elif bx >= (self.width - (self.ballwidth // 2)):
                bx = (self.width - (self.ballwidth // 2))
                self.ball_change_x = self.ball_change_x * -1
            elif by >= (self.height - self.ballheight):
                by = (self.height - self.ballheight)
                self.ball_change_y = self.ball_change_y * -1
            elif by <= -self.ballheight:
                self.ball_change_y = self.ball_change_y * -1

            self.balltrans.set_translation(bx, by)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
