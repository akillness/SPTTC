import pygame
import math
import random
from pygame.math import Vector2

def limit_magnitude(vector, max_magnitude):
    magnitude = vector.length()
    if magnitude > max_magnitude:
        return vector.normalize() * max_magnitude
    return Vector2(vector)

class Boid:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        random_x = random.uniform(-1, 1)
        random_y = random.uniform(-1, 1)
        
        self.velocity = Vector2(random_x, random_y).normalize() * 3
        self.acceleration = Vector2(0, 0)
        self.max_force = 0.3
        self.max_speed = 0.1
        self.perception = 80

    def update(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        self.velocity = limit_magnitude(self.velocity, self.max_speed)
        self.acceleration *= 0

    def seek_target(self, target_pos):
        # 목표점을 향한 조향력 계산
        desired = target_pos - self.position
        desired = desired.normalize() * self.max_speed
        steering = desired - self.velocity
        return limit_magnitude(steering, self.max_force)

    def apply_rules(self, boids, target_pos):
        separation = self.separation(boids)
        alignment = self.alignment(boids)
        cohesion = self.cohesion(boids)
        seek = self.seek_target(target_pos)  # 목표점 추적
        
        # 가중치 조정
        self.acceleration += separation * 1.0
        self.acceleration += alignment * 0.5
        self.acceleration += cohesion * 0.5
        self.acceleration += seek * 1.5  # 목표점 추적 가중치

    def separation(self, boids):
        steering = Vector2(0, 0)
        total = 0
        
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < self.perception/2:
                diff = self.position - other.position
                diff /= distance**2
                steering += diff
                total += 1
                
        if total > 0:
            steering /= total
            steering = limit_magnitude(steering, self.max_speed)
            steering -= self.velocity
            steering = limit_magnitude(steering, self.max_force)
            
        return steering

    def alignment(self, boids):
        steering = Vector2(0, 0)
        total = 0
        
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < self.perception:
                steering += other.velocity
                total += 1
                
        if total > 0:
            steering /= total
            steering = limit_magnitude(steering, self.max_speed)
            steering -= self.velocity
            steering = limit_magnitude(steering, self.max_force)
            
        return steering

    def cohesion(self, boids):
        steering = Vector2(0, 0)
        total = 0
        
        for other in boids:
            distance = self.position.distance_to(other.position)
            if 0 < distance < self.perception:
                steering += other.position
                total += 1
                
        if total > 0:
            steering /= total
            steering -= self.position
            steering = limit_magnitude(steering, self.max_speed)
            steering -= self.velocity
            steering = limit_magnitude(steering, self.max_force)
            
        return steering

    def edges(self, width, height):
        margin = 50
        turn_factor = 0.2
        if self.position.x > width - margin:
            self.velocity.x -= turn_factor
        if self.position.x < margin:
            self.velocity.x += turn_factor
        if self.position.y > height - margin:
            self.velocity.y -= turn_factor
        if self.position.y < margin:
            self.velocity.y += turn_factor

class BoidsSimulation:
    def __init__(self, num_boids=50):
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.boids = [Boid(random.uniform(0, self.width), 
                         random.uniform(0, self.height)) 
                    for _ in range(num_boids)]
        
        # 목표점 초기화
        self.target_pos = Vector2(self.width/2, self.height/2)
        
        # 삼각형 크기 변경 및 중심 조정
        self.triangle = pygame.Surface((20, 30), pygame.SRCALPHA)
        pygame.draw.polygon(self.triangle, (255, 255, 255), 
                          [(0,0), (20,15), (0,30)])

    def run(self):
        running = True
        font = pygame.font.Font(None, 36)
        
        while running:
            self.screen.fill((0, 0, 30))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 마우스 클릭으로 목표점 이동
                    self.target_pos = Vector2(event.pos)

            # 목표점 표시
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(self.target_pos.x), int(self.target_pos.y)), 10)

            for boid in self.boids:
                boid.apply_rules(self.boids, self.target_pos)
                boid.update()
                boid.edges(self.width, self.height)
                
                angle = math.degrees(math.atan2(-boid.velocity.y, boid.velocity.x)) - 90
                rotated = pygame.transform.rotate(self.triangle, angle)
                self.screen.blit(rotated, 
                               rotated.get_rect(center=(int(boid.position.x), 
                                                      int(boid.position.y))))
            
            fps_text = font.render(f"FPS: {self.clock.get_fps():.1f}", True, (255,255,255))
            self.screen.blit(fps_text, (10, 10))
            
            # 목표점 위치 표시
            target_text = font.render("Click to set target", True, (255,255,255))
            self.screen.blit(target_text, (10, 50))
            
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    simulation = BoidsSimulation(num_boids=50)
    simulation.run() 