import pygame
import random
import sys
from pygame.locals import *

# 初始化pygame
pygame.init()

# 游戏常量
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SIDEBAR_WIDTH = 200

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
GRAY = (40, 40, 40)
LIGHT_GRAY = (100, 100, 100)
DARK_GRAY = (20, 20, 20)
BACKGROUND = (15, 15, 30)

# 方块形状定义
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # J
    [[1, 1, 1], [0, 0, 1]],  # L
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]  # Z
]

# 方块颜色
SHAPE_COLORS = [CYAN, YELLOW, PURPLE, BLUE, ORANGE, GREEN, RED]

# 设置游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("俄罗斯方块")

# 创建游戏时钟
clock = pygame.time.Clock()

# 创建字体
font_large = pygame.font.SysFont("Arial", 36, bold=True)
font_medium = pygame.font.SysFont("Arial", 24)
font_small = pygame.font.SysFont("Arial", 18)


class Tetromino:
    def __init__(self):
        self.shape_index = random.randint(0, len(SHAPES) - 1)
        self.shape = SHAPES[self.shape_index]
        self.color = SHAPE_COLORS[self.shape_index]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.rotation = 0

    def rotate(self):
        # 旋转方块
        rotated = list(zip(*reversed(self.shape)))
        return [list(row) for row in rotated]

    def get_positions(self):
        # 获取方块所有格子的位置
        positions = []
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    positions.append((self.x + x, self.y + y))
        return positions


class TetrisGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_speed = 0.5  # 方块下落速度（秒）
        self.fall_time = 0
        self.paused = False

    def valid_move(self, piece, x_offset=0, y_offset=0, rotation=None):
        # 检查移动是否有效
        shape = piece.shape if rotation is None else rotation
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x, new_y = piece.x + x + x_offset, piece.y + y + y_offset
                    if (new_x < 0 or new_x >= GRID_WIDTH or
                            new_y >= GRID_HEIGHT or
                            (new_y >= 0 and self.board[new_y][new_x])):
                        return False
        return True

    def lock_piece(self):
        # 将当前方块锁定到游戏板上
        for x, y in self.current_piece.get_positions():
            if y >= 0:  # 只锁定在游戏区域内的方块
                self.board[y][x] = self.current_piece.color

        # 检查是否有完整的行
        self.clear_lines()

        # 生成新方块
        self.current_piece = self.next_piece
        self.next_piece = Tetromino()

        # 检查游戏是否结束
        if not self.valid_move(self.current_piece):
            self.game_over = True

    def clear_lines(self):
        # 清除完整的行并计分
        lines_to_clear = []
        for y, row in enumerate(self.board):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(y)

        if not lines_to_clear:
            return

        # 计分
        cleared = len(lines_to_clear)
        self.lines_cleared += cleared
        self.score += [100, 300, 500, 800][min(cleared - 1, 3)] * self.level

        # 更新等级
        self.level = self.lines_cleared // 10 + 1
        self.fall_speed = max(0.05, 0.5 - (self.level - 1) * 0.05)

        # 清除行
        for line in lines_to_clear:
            del self.board[line]
            self.board.insert(0, [0 for _ in range(GRID_WIDTH)])

    def move(self, dx, dy):
        # 移动方块
        if not self.game_over and not self.paused:
            if self.valid_move(self.current_piece, dx, dy):
                self.current_piece.x += dx
                self.current_piece.y += dy
                return True
        return False

    def rotate_piece(self):
        # 旋转方块
        if not self.game_over and not self.paused:
            rotated = self.current_piece.rotate()
            if self.valid_move(self.current_piece, rotation=rotated):
                self.current_piece.shape = rotated
                return True
        return False

    def drop(self):
        # 快速下落
        if not self.game_over and not self.paused:
            while self.move(0, 1):
                pass
            self.lock_piece()

    def update(self, delta_time):
        # 更新游戏状态
        if self.game_over or self.paused:
            return

        self.fall_time += delta_time
        if self.fall_time >= self.fall_speed:
            if not self.move(0, 1):
                self.lock_piece()
            self.fall_time = 0

    def draw(self):
        # 绘制游戏界面
        screen.fill(BACKGROUND)

        # 绘制游戏区域边框
        pygame.draw.rect(screen, LIGHT_GRAY, (50, 50, GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE), 2)

        # 绘制网格
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(screen, GRAY,
                                 (50 + x * GRID_SIZE, 50 + y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)

        # 绘制已锁定的方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.board[y][x]:
                    pygame.draw.rect(screen, self.board[y][x],
                                     (50 + x * GRID_SIZE + 1, 50 + y * GRID_SIZE + 1,
                                      GRID_SIZE - 2, GRID_SIZE - 2))

        # 绘制当前方块
        for x, y in self.current_piece.get_positions():
            if y >= 0:  # 只绘制在游戏区域内的方块
                pygame.draw.rect(screen, self.current_piece.color,
                                 (50 + x * GRID_SIZE + 1, 50 + y * GRID_SIZE + 1,
                                  GRID_SIZE - 2, GRID_SIZE - 2))
                # 添加方块内部效果
                pygame.draw.rect(screen, WHITE,
                                 (50 + x * GRID_SIZE + 5, 50 + y * GRID_SIZE + 5,
                                  GRID_SIZE - 10, GRID_SIZE - 10), 1)

        # 绘制阴影（预测落点）
        shadow_y = self.current_piece.y
        while self.valid_move(self.current_piece, 0, shadow_y - self.current_piece.y + 1):
            shadow_y += 1

        for x, y in self.current_piece.get_positions():
            if y >= 0:  # 只绘制在游戏区域内的方块
                pygame.draw.rect(screen, (100, 100, 100, 100),
                                 (50 + x * GRID_SIZE + 1, 50 + (y + shadow_y - self.current_piece.y) * GRID_SIZE + 1,
                                  GRID_SIZE - 2, GRID_SIZE - 2), 1)

        # 绘制侧边栏
        sidebar_x = 50 + GRID_WIDTH * GRID_SIZE + 20

        # 绘制下一个方块预览
        next_text = font_medium.render("next", True, WHITE)
        screen.blit(next_text, (sidebar_x, 50))
        pygame.draw.rect(screen, DARK_GRAY, (sidebar_x, 80, 150, 150))

        # 绘制下一个方块
        next_shape = self.next_piece.shape
        next_color = self.next_piece.color
        shape_width = len(next_shape[0]) * GRID_SIZE
        shape_height = len(next_shape) * GRID_SIZE
        start_x = sidebar_x + (150 - shape_width) // 2
        start_y = 80 + (150 - shape_height) // 2

        for y, row in enumerate(next_shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, next_color,
                                     (start_x + x * GRID_SIZE + 1, start_y + y * GRID_SIZE + 1,
                                      GRID_SIZE - 2, GRID_SIZE - 2))
                    pygame.draw.rect(screen, WHITE,
                                     (start_x + x * GRID_SIZE + 5, start_y + y * GRID_SIZE + 5,
                                      GRID_SIZE - 10, GRID_SIZE - 10), 1)

        # 绘制分数
        score_text = font_medium.render(f"count: {self.score}", True, YELLOW)
        screen.blit(score_text, (sidebar_x, 250))

        # 绘制等级
        level_text = font_medium.render(f"level: {self.level}", True, GREEN)
        screen.blit(level_text, (sidebar_x, 290))

        # 绘制消除行数
        lines_text = font_medium.render(f"Eliminate the number of lines: {self.lines_cleared}", True, CYAN)
        screen.blit(lines_text, (sidebar_x, 330))

        # 绘制游戏说明
        controls_y = 400
        controls = [
            "Game Controls:",
            "← →: Move left and right",
            "↑ : rotation",
            "↓ : Accelerate the fall",
            "Space: Fall directly",
            "P: Pause/Continue",
            "R: Start over",
            "ESC: Exit"
        ]

        for i, text in enumerate(controls):
            ctrl_text = font_small.render(text, True, LIGHT_GRAY)
            screen.blit(ctrl_text, (sidebar_x, controls_y + i * 30))

        # 绘制游戏标题
        title = font_large.render("Tetris", True, WHITE)
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 5))

        # 绘制暂停状态
        if self.paused:
            pause_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pause_surface.fill((0, 0, 0, 150))
            screen.blit(pause_surface, (0, 0))
            pause_text = font_large.render("Game paused", True, YELLOW)
            screen.blit(pause_text, (SCREEN_WIDTH // 2 - pause_text.get_width() // 2,
                                     SCREEN_HEIGHT // 2 - pause_text.get_height() // 2))

        # 绘制游戏结束状态
        if self.game_over:
            game_over_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            game_over_surface.fill((0, 0, 0, 180))
            screen.blit(game_over_surface, (0, 0))
            game_over_text = font_large.render("Game Over!", True, RED)
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2,
                                         SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))

            restart_text = font_medium.render("Restart R", True, WHITE)
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2,
                                       SCREEN_HEIGHT // 2 + 50))


# 创建游戏实例
game = TetrisGame()

# 主游戏循环
running = True
last_time = pygame.time.get_ticks()

while running:
    # 计算时间增量
    current_time = pygame.time.get_ticks()
    delta_time = (current_time - last_time) / 1000.0  # 转换为秒
    last_time = current_time

    # 处理事件
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

            if not game.game_over:
                if event.key == K_LEFT:
                    game.move(-1, 0)
                elif event.key == K_RIGHT:
                    game.move(1, 0)
                elif event.key == K_DOWN:
                    game.move(0, 1)
                elif event.key == K_UP:
                    game.rotate_piece()
                elif event.key == K_SPACE:
                    game.drop()
                elif event.key == K_p:
                    game.paused = not game.paused

            if event.key == K_r:
                game.reset()

    # 更新游戏状态
    game.update(delta_time)

    # 绘制游戏
    game.draw()

    # 更新屏幕
    pygame.display.flip()

    # 控制游戏帧率
    clock.tick(60)

# 退出游戏
pygame.quit()
sys.exit()