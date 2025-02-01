import sqlite3
import json

KILTER = {"DB": "./data/KilterDB.sqlite3", "MIN_ASCEN": 5, "LAYOUTS": [1, 8], "FRAMES_COUNT": 1, "PRODUCTS": [1, 7], "QUALITY": 2.6}

PLACEMENTS_QUERY = """
SELECT placements.id, placements.set_id, holes.x, holes.y
FROM placements
INNER JOIN holes
ON placements.hole_id = holes.id
WHERE
placements.layout_id IN (1, 8)
"""

ROLES_QUERY = """
SELECT id, name, led_color FROM placement_roles
"""

LEDS_QUERY = """
SELECT holes.id, leds.product_size_id, leds.position
FROM holes
INNER JOIN leds
ON holes.id = leds.hole_id
WHERE holes.product_id IN (1, 7)
"""

def export_placements(placements):
    placement_dict = {}
    for placement in placements:
        placement_dict[placement[0]] = {'set': placement[1], 'x': placement[2], 'y': placement[3]}
    with open("./data/placements.json", 'w') as f:
        json.dump(placement_dict, f)

def export_roles(roles):
    roles_dict = {}
    for role in roles:
        roles_dict[role[0]] = {'name': role[1], 'color': role[2]}
    with open("./data/roles.json", 'w') as f:
        json.dump(roles_dict, f)

def export_leds(leds):
    leds_dict = {}
    for led in leds:
        if led[0] in leds_dict.keys():
            leds_dict[led[0]].append({'product_size_id': led[1], 'position': led[2]})
        else:
            leds_dict[led[0]] = [{'product_size_id': led[1], 'position': led[2]}]
    print("NEW :ED LEN: ", len(leds_dict))
    with open("./data/leds.json", 'w') as f:
        json.dump(leds_dict, f)

if __name__ == "__main__":
    cursor = sqlite3.connect(KILTER["DB"]).cursor()
    cursor.execute(PLACEMENTS_QUERY)
    placements = cursor.fetchall()
    export_placements(placements) # ID -> (set, x, y)
    cursor.execute(ROLES_QUERY)
    roles = cursor.fetchall()
    export_roles(roles) # ID -> (name, color)
    cursor.execute(LEDS_QUERY)
    leds = cursor.fetchall()
    export_leds(leds) # ID -> [(product_size_id, position)...]