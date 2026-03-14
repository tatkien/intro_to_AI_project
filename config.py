from enum import IntEnum

class Tile(IntEnum):
    EMPTY = 0
    WALL = 1
    MUD = 2

TILE_PROPS = {
    Tile.EMPTY: {"walkable": True, "cost": 1, "desc": "Road"},
    Tile.WALL: {"walkable": False, "cost": int(1e6), "desc": "Wall"},
    Tile.MUD: {"walkable": True, "cost": 3, "desc": "Mud"},
}

TILE_IMAGES = {
    Tile.EMPTY: "images/road.jpg",
    Tile.WALL: "images/wall.jpg",
    Tile.MUD: "images/mud.jpg",
}