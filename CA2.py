import pygame
import numpy as np
import random
import heapq
from collections import defaultdict

pygame.init()

WINDOW_SIZE = (832, 832)
TILE_SIZE = 32
GRID_SIZE = 26

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Jigsaw Puzzle")

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = defaultdict(int)
    for char in data:
        frequency[char] += 1
    
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(root, code="", mapping=None):
    if mapping is None:
        mapping = {}
    
    if root is None:
        return
    
    if root.char is not None:
        mapping[root.char] = code
    
    generate_huffman_codes(root.left, code + "0", mapping)
    generate_huffman_codes(root.right, code + "1", mapping)
    
    return mapping

def matrix_chain_multiplication(matrices):
    n = len(matrices)
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + matrices[i][0] * matrices[k][1] * matrices[j][1]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]

class Tile:
    def __init__(self, id, image):
        self.id = id
        self.image = image
        self.rotation = 0

    def rotate(self):
        self.rotation = (self.rotation + 90) % 360
        self.image = pygame.transform.rotate(self.image, 90)

class JigsawRecorder:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.moves = []
        self.huffman_codes = None
    
    def record_move(self, tile_id, old_x, old_y, new_x, new_y, rotation):
        move = (tile_id, old_x, old_y, new_x, new_y, rotation)
        self.moves.append(move)
    
    def compress_moves(self):
        move_data = "".join(str(m) for m in self.moves)
        huffman_tree = build_huffman_tree(move_data)
        self.huffman_codes = generate_huffman_codes(huffman_tree)
        
        compressed_data = ""
        for char in move_data:
            compressed_data += self.huffman_codes[char]
        
        return compressed_data
    
    def decompress_moves(self, compressed_data):
        reverse_mapping = {code: char for char, code in self.huffman_codes.items()}
        current_code = ""
        decompressed_data = ""
        
        for bit in compressed_data:
            current_code += bit
            if current_code in reverse_mapping:
                decompressed_data += reverse_mapping[current_code]
                current_code = ""
        
        return decompressed_data

    def get_moves(self):
        return self.moves

def create_puzzle(image_path):
    original_image = pygame.image.load(image_path)
    original_image = pygame.transform.scale(original_image, WINDOW_SIZE)
    tiles = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tile_surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
            tile_surface.blit(original_image, (0, 0), (j*TILE_SIZE, i*TILE_SIZE, TILE_SIZE, TILE_SIZE))
            tile = Tile(i*GRID_SIZE + j, tile_surface)
            if random.choice([True, False]):  # Randomly rotate some tiles
                tile.rotate()
            tiles.append(tile)
    random.shuffle(tiles)
    return tiles

def draw_puzzle(tiles):
    for i, tile in enumerate(tiles):
        x = (i % GRID_SIZE) * TILE_SIZE
        y = (i // GRID_SIZE) * TILE_SIZE
        screen.blit(tile.image, (x, y))
    pygame.display.flip()

def optimize_rotations(rotations):
    matrices = [(TILE_SIZE, TILE_SIZE) for _ in range(len(rotations))]
    return matrix_chain_multiplication(matrices)

def main():
    tiles = create_puzzle("test.png")  
    recorder = JigsawRecorder(tiles)

    selected_tile = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  
                    x, y = event.pos
                    tile_x, tile_y = x // TILE_SIZE, y // TILE_SIZE
                    index = tile_y * GRID_SIZE + tile_x
                    if selected_tile is None:
                        selected_tile = (index, tile_x, tile_y)
                    else:
                        old_index, old_x, old_y = selected_tile
                        tiles[old_index], tiles[index] = tiles[index], tiles[old_index]
                        recorder.record_move(tiles[index].id, old_x, old_y, tile_x, tile_y, tiles[index].rotation)
                        selected_tile = None
                elif event.button == 3: 
                    x, y = event.pos
                    tile_x, tile_y = x // TILE_SIZE, y // TILE_SIZE
                    index = tile_y * GRID_SIZE + tile_x
                    tiles[index].rotate()
                    recorder.record_move(tiles[index].id, tile_x, tile_y, tile_x, tile_y, tiles[index].rotation)

        screen.fill(BLACK)
        draw_puzzle(tiles)
        
        if selected_tile:
            x, y = selected_tile[1] * TILE_SIZE, selected_tile[2] * TILE_SIZE
            pygame.draw.rect(screen, RED, (x, y, TILE_SIZE, TILE_SIZE), 2)
        
        pygame.display.flip()

    pygame.quit()

    print("Recorded moves:")
    for move in recorder.get_moves():
        print(move)

    compressed_data = recorder.compress_moves()
    print("Compressed moves:", compressed_data)

    decompressed_data = recorder.decompress_moves(compressed_data)
    print("Decompressed moves:", decompressed_data)

    # rotations = [move[5] for move in recorder.get_moves() if move[2] == move[4] and move[3] == move[5]]  
    # min_operations = optimize_rotations(rotations)
    # print("Minimum operations for rotation calculations:", min_operations)

if __name__ == "__main__":
    main()
