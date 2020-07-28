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

        self.cart_change_x = 5
        self.ball_change_x = 5
        self.ball_change_y = 5
        
        self.carttrans = None
        self.balltrans = None
        self.x_threshold = 2.4
        self.scores = 0
        
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

        self.action_space = spaces.Discrete(2)
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

        done = 0
        is_score = False

        x, y = self.carttrans.translation
        bx, by = self.balltrans.translation

        if action == 0:
            x = x - self.cart_change_x 
        elif action == 1:
            x = x + self.cart_change_x

        if x <= (self.cartwidth // 2):
            x = (self.cartwidth // 2)
        if x >= (self.width - (self.cartwidth // 2)):
            x = (self.width - (self.cartwidth // 2))

        bx = bx + self.ball_change_x
        by = by + self.ball_change_y

        if bx <= (self.ballwidth // 2):
            bx = (self.ballwidth // 2)
            self.ball_change_x = self.ball_change_x * -1
        elif bx >= (self.width - (self.ballwidth // 2)):
            bx = (self.width - (self.ballwidth // 2))
            self.ball_change_x = self.ball_change_x * -1
        elif by >= (self.height - self.ballheight):
            by = (self.height - self.ballheight)
            self.ball_change_y = self.ball_change_y * -1
        elif bx > (x - 50) and bx < (x + 50) and by == (self.cartheight+10):
            self.ball_change_y = self.ball_change_y * -1
            done = False
            is_score = True
        elif by <= -self.ballheight:
            self.ball_change_y = self.ball_change_y * -1
            done = True 
            self.scores = 0
        
        if is_score:
            reward = 1.0
            self.scores = self.scores + 1
            print('Rewarded 1.0')
            print('Scores', self.scores)
        elif self.steps_beyond_done is None:
            # ball touches the bottom
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        
        self.state =(x, bx, by, bx-x, by-y)
        
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-100, high=100, size=(5,))
        self.steps_beyond_done = None

        if self.carttrans and self.balltrans:
            self.carttrans.set_translation(self.width/2, self.cartheight/2)
            rand_x = np.random.randint(self.width/2 - 100, high=self.width/2 + 100, size=1)
            self.balltrans.set_translation(rand_x, 100)

            x, y = self.carttrans.translation
            bx, by = self.balltrans.translation
            self.state = (x, bx, by, bx-x, by-y)

        return np.array(self.state)

    def render(self, mode='human'):
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
            
            x, y = self.carttrans.translation
            bx, by = self.balltrans.translation
            self.state =(x, bx, by, bx-x, by-y)

        t = self.state
        y = self.carttrans.translation[1]
        self.carttrans.set_translation(t[0], y)
        self.balltrans.set_translation(t[1], t[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
