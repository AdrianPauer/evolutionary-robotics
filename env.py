import math
import numpy as np
import pygame



class Robot2DEnv:
    """Simple 2D robot environment with forward motion and rotation."""

    def __init__(self,
                 world_size=10.0,
                 max_speed=3.0,
                 max_steps=500,
                 render_size=(800, 600),
                 enable_render=False,
                 vision_mask=True,
                 spawn_in_center=False,
                 food_count=6,
                 obs_type='ray'):
        
        self.world_size = world_size
        self.half_size = world_size / 2.0
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.render_size = render_size
        self.enable_render = enable_render
        self.vision_mask = vision_mask
        self.spawn_in_center = spawn_in_center
        self.food_count = food_count

        self.steer_rate = 0.06
        self.accel_rate = 0.05
        self.friction = 0.95
        self.robot_radius = 0.3
        self.ray_count = 5
        self.ray_range = 3.5

        self.spawn_radius = self.world_size * 0.15
        self.initial_life = 120
        self.life_decay = 0.4
        self.life_gain = 60
        self.food_radius = 0.2

        self.screen = None
        self.clock = None
        self.font = None

        if obs_type != "ray":
            raise ValueError("Robot2DEnv only supports obs_type='ray'.")

        self.obs_type = obs_type

        self.reset()

        if self.enable_render:
            self._init_pygame()

    def _init_pygame(self):
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install it with 'pip install pygame'.")

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.render_size)
            pygame.display.set_caption("Robot2DEnv")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 22)

    def reset(self):
        if self.spawn_in_center:
            self.pos = np.array([0.0, 0.0], dtype=float)
        else:
            self.pos = self._spawn_position()
        self.angle = np.random.uniform(-math.pi, math.pi)
        self.velocity = 0.0
        self.life = self.initial_life
        self.food_eaten = 0
        self.step_count = 0
        self.done = False
        self._spawn_foods()
        return self._get_obs()

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot call step() after environment is done. Call reset() first.")

        steer, throttle = action
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

        self.angle += steer * self.steer_rate
        self.velocity += throttle * self.accel_rate
        self.velocity *= self.friction
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)

        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        new_pos = self.pos + direction * self.velocity * 0.1
        if self._is_outside(new_pos):
            self.velocity = 0.0
        else:
            self.pos = new_pos

        self.life -= self.life_decay
        if self.life < 0:
            self.life = 0

        self._check_food()

        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        self.done = (
            (self.life <= 0)
            or self._is_outside(new_pos)
            or (self.step_count >= self.max_steps)
        )
        info = {
            "step_count": self.step_count,
            "life": self.life,
            "food_eaten": self.food_eaten,
        }

        return obs, reward, self.done, info

    def _get_obs(self):
        return self._ray_distances()

    def _compute_reward(self, obs):
        return 0.01 + self.velocity / 20 + (1 - sum(obs) / len(obs)) * 10

    def _is_outside(self, point):
        x, y = point
        return abs(x) + self.robot_radius > self.half_size or abs(y) + self.robot_radius > self.half_size

    def _is_collision(self):
        return self._is_outside(self.pos)

    def _spawn_position(self):
        radius = np.random.uniform(0, self.spawn_radius)
        angle = np.random.uniform(0, 2 * math.pi)
        return np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=float)

    def _random_food_position(self):
        while True:
            pos = np.random.uniform(-self.half_size * 0.7, self.half_size * 0.7, size=2)
            if np.linalg.norm(pos) > self.spawn_radius + self.food_radius + 0.5:
                return pos

    def _spawn_foods(self):
        self.food = [self._random_food_position() for _ in range(self.food_count)]

    def _check_food(self):
        for index, food_pos in enumerate(self.food):
            if np.linalg.norm(self.pos - food_pos) <= self.robot_radius + self.food_radius:
                self.life = min(self.initial_life, self.life + self.life_gain)
                self.food_eaten += 1
                self.food[index] = self._random_food_position()
                break

    def _calculate_single_ray(self, ray_angle):
        # 1. Create the unit vector for this specific ray
        direction = np.array([math.cos(ray_angle), math.sin(ray_angle)])
        
        min_hit_dist = self.ray_range
        
        # 3. Check every food pellet against THIS specific ray
        for food_pos in self.food:
            rel = food_pos - self.pos
            proj = np.dot(rel, direction)

            if proj <= 0:
                continue
                
            closest_sq = np.dot(rel, rel) - proj * proj
            
            # Check if the ray actually passes through the food circle
            if closest_sq <= self.food_radius**2:
                # Calculate the exact 'hit' point on the edge of the food circle
                offset = math.sqrt(max(0.0, self.food_radius**2 - closest_sq))
                hit_dist = proj - offset
                
                # If this hit is closer than previous hits and within range, save it
                if 0 <= hit_dist < min_hit_dist:
                    min_hit_dist = hit_dist
                    
        return min_hit_dist

    def _ray_distances(self):

        direction = np.array([math.cos(self.angle), math.sin(self.angle)])

        # detect food only (ignore walls)
        food_dist = self.ray_range
        for food_pos in self.food:
            rel = food_pos - self.pos
            proj = np.dot(rel, direction)
            if proj <= 0:
                continue
            closest_sq = np.dot(rel, rel) - proj * proj
            if closest_sq <= self.food_radius * self.food_radius:
                offset = math.sqrt(max(0.0, self.food_radius * self.food_radius - closest_sq))
                hit = proj - offset
                if 0 <= hit < food_dist:
                    food_dist = hit

        #return min(food_dist, self.ray_range)
        fov = math.radians(30)  # Total arc of 30 degrees
        start_angle = self.angle - fov / 2
        step = fov / (self.ray_count - 1)

        vision_inputs = []
    
        for i in range(self.ray_count):
            ray_angle = start_angle + (i * step)

            # Reuse your projection logic but pass 'ray_angle'
            dist = self._calculate_single_ray(ray_angle) 
            
            # Normalize: 1.0 = nothing, 0.0 = right in front
            vision_inputs.append(dist / self.ray_range)
            
        return vision_inputs

    def render(self, mode="human"):
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install it with 'pip install pygame'.")

        if self.screen is None:
            self._init_pygame()

        self.screen.fill((20, 20, 30))
        self._draw_world()
        self._draw_foods()
        self._draw_robot()
        self._draw_rays()
        if self.vision_mask:
            self._draw_vision_mask()
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    def _world_to_screen(self, point):
        sx = int((point[0] + self.half_size) / self.world_size * self.render_size[0])
        sy = int((self.half_size - point[1]) / self.world_size * self.render_size[1])
        return sx, sy

    def _draw_world(self):
        rect = pygame.Rect(0, 0, *self.render_size)
        pygame.draw.rect(self.screen, (40, 40, 70), rect)
        border = pygame.Rect(0, 0, *self.render_size)
        pygame.draw.rect(self.screen, (90, 90, 140), border, 4)

        center = self._world_to_screen((0.0, 0.0))


    def _draw_robot(self):
        robot_center = self._world_to_screen(self.pos)
        heading = np.array([math.cos(self.angle), math.sin(self.angle)])
        nose = self.pos + heading * self.robot_radius * 1.5

        pygame.draw.circle(self.screen, (220, 220, 220), robot_center, int(self.robot_radius * self.render_size[1] / self.world_size))
        pygame.draw.line(
            self.screen,
            (255, 100, 100),
            robot_center,
            self._world_to_screen(nose),
            2,
        )

    def _draw_foods(self):
        for food_pos in self.food:
            food_center = self._world_to_screen(food_pos)
            pygame.draw.circle(self.screen, (120, 255, 120), food_center, int(self.food_radius * self.render_size[1] / self.world_size))

    def _draw_hud(self):
        bar_width = 240
        bar_height = 18
        margin = 12
        x = margin
        y = margin

        life_fraction = self.life / self.initial_life
        life_color = (180, 230, 120) if life_fraction > 0.4 else (235, 120, 120)

        pygame.draw.rect(self.screen, (60, 60, 80), (x, y, bar_width, bar_height), border_radius=6)
        pygame.draw.rect(self.screen, life_color, (x + 2, y + 2, int((bar_width - 4) * life_fraction), bar_height - 4), border_radius=6)
        pygame.draw.rect(self.screen, (180, 180, 200), (x, y, bar_width, bar_height), 2, border_radius=6)

        if self.font is not None:
            text = self.font.render(f"Life: {self.life} / {self.initial_life}", True, (240, 240, 240))
            self.screen.blit(text, (x + 8, y - 2))
            step_text = self.font.render(f"Step: {self.step_count}", True, (240, 240, 240))
            self.screen.blit(step_text, (x + 12, y + bar_height + 6))
            vision_text = self.font.render(f"Vision mask: {'ON' if self.vision_mask else 'OFF'}", True, (240, 240, 240))
            self.screen.blit(vision_text, (x + 12, y + bar_height + 28))

    def _draw_rays(self):
        num_rays = 5
        fov = math.radians(30)
        start_angle = self.angle - (fov / 2)
        step = fov / (num_rays - 1)

        x,y = self.pos
        for i in range(num_rays):
            angle = start_angle + (i * step)
            dist = self._calculate_single_ray(angle)
            
            origin = self._world_to_screen(self.pos)
            target = self._world_to_screen(self.pos + dist * np.array([math.cos(angle), math.sin(angle)]))
            
            # Determine color: Red if hitting food, Green if seeing nothing
            color = (255, 0, 0) if dist < self.ray_range else (0, 255, 0)
            
            # Draw the line from robot center to hit point
            pygame.draw.line(self.screen, color, origin, target, 1)

    def _ray_hit_points(self):
        fov = math.radians(30)
        start_angle = self.angle - (fov / 2)
        step = fov / (self.ray_count - 1)

        hit_points = []
        for i in range(self.ray_count):
            angle = start_angle + (i * step)
            dist = self._calculate_single_ray(angle)
            hit_points.append(self.pos + dist * np.array([math.cos(angle), math.sin(angle)]))
        return hit_points

    def _draw_vision_mask(self):
        overlay = pygame.Surface(self.render_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 255))

        polygon_points = [self._world_to_screen(self.pos)]
        polygon_points.extend(self._world_to_screen(point) for point in self._ray_hit_points())
        polygon_points.append(self._world_to_screen(self.pos))

        pygame.draw.polygon(overlay, (0, 0, 0, 0), polygon_points)
        pygame.draw.circle(
            overlay,
            (0, 0, 0, 0),
            self._world_to_screen(self.pos),
            max(8, int(self.robot_radius * self.render_size[1] / self.world_size * 1.5)),
        )
        self.screen.blit(overlay, (0, 0))



def demo():
    env = Robot2DEnv(enable_render=True, vision_mask=True, spawn_in_center=True, food_count=1, obs_type='ray')
    obs = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                env.vision_mask = not env.vision_mask

        keys = pygame.key.get_pressed()
        steer = 0.0
        throttle = 0.0
        if keys[pygame.K_LEFT]:
            steer = 1.0
        if keys[pygame.K_RIGHT]:
            steer = -1.0
        if keys[pygame.K_UP]:
            throttle = 1.0
        if keys[pygame.K_DOWN]:
            throttle = -1.0

        obs, reward, done, info = env.step((steer, throttle))

        print(obs, reward)
        env.render()

        if done:
            running = False

    env.close()


if __name__ == "__main__":
    demo()

