import pygame, random, math

class Surface(pygame.sprite.Sprite):

    def __init__(self, screen_dimension, location):
        pygame.sprite.Sprite.__init__(self)

        self.x = location[0]
        self.y = location[1]

        self.image = pygame.Surface([screen_dimension[0],screen_dimension[1]])
        self.image.fill((255,255,255))
        self.image.set_colorkey((255,255,255))

        # pygame.draw.rect(self.image, (255,0,0), [0,0,200,5])
        polygon_surface_points = self.random_ground(screen_dimension[1], screen_dimension[0], 8)
        # self.rect = pygame.draw.polygon(self.image, (0,0,0), [(0,0), (screen_dimension[0], 0), (screen_dimension[0], screen_dimension[1]), (0, screen_dimension[1])])        
        self.rect = pygame.draw.polygon(self.image, (0,0,0), polygon_surface_points)        
        
        # self.rect = pygame.draw.line(screen, (255,0,0), start_point, end_point, 5)
        # self.rect.x = start_point[0]
        # self.rect.y = start_point[1]

    def random_ground(self, screen_height, screen_width, spacing):
        # set out the boundaries
        highest_point = screen_height - (screen_height / 6)
        lowest_point = screen_height + 50
        left_most_point = -50
        right_most_point = screen_width + 50

        ans = [(right_most_point, highest_point/2),
               (right_most_point, lowest_point),
               (left_most_point, left_most_point),
               (left_most_point, highest_point/2)]

        number_of_points = screen_width / spacing
        i = 0
        while i < number_of_points:
            rand = random.random()
            rand = rand % 12
            last_y_point = ans[len(ans)-1][0]
            last_x_point = ans[len(ans)-1][1]
            ans.append((last_y_point + rand,last_x_point + spacing))
            i = i + 1
        return ans
