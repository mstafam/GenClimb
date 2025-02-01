import sqlite3
import re
import json
import pandas as pd

KILTER = {"DB": "./data/KilterDB.sqlite3", "MIN_ASCEN": 5, "LAYOUTS": [1, 8], "FRAMES_COUNT": 1, "PRODUCTS": [1, 7], "QUALITY": 2.6}

CLIMBS_COUNT = f"""SELECT COUNT(*) 
FROM 
    climb_stats 
INNER JOIN 
    climbs 
ON 
    climb_stats.climb_uuid = climbs.uuid
WHERE 
    climb_stats.ascensionist_count >= {KILTER['MIN_ASCEN']} 
    AND climbs.frames_count = {KILTER['FRAMES_COUNT']}
    AND climb_stats.quality_average >= {KILTER['QUALITY']}
    AND climbs.layout_id IN ({','.join(str(i) for i in tuple(KILTER['LAYOUTS']))})"""

CLIMBS_QUERY = f"""SELECT 
    climbs.uuid, 
    climbs.layout_id, 
    climbs.hsm, 
    climbs.edge_left, 
    climbs.edge_right, 
    climbs.edge_bottom, 
    climbs.edge_top, 
    climb_stats.quality_average, 
    climbs.frames, 
    climb_stats.display_difficulty
FROM 
    climb_stats 
INNER JOIN 
    climbs 
ON 
    climb_stats.climb_uuid = climbs.uuid
WHERE 
    climb_stats.ascensionist_count >= {KILTER['MIN_ASCEN']} 
    AND climbs.frames_count = {KILTER['FRAMES_COUNT']}
    AND climb_stats.quality_average >= {KILTER['QUALITY']}
    AND climbs.layout_id IN ({','.join(str(i) for i in tuple(KILTER['LAYOUTS']))})
    AND climbs.is_listed = 1"""

LAYOUT_TO_PRODUCT_QUERY = f"SELECT product_id FROM layouts WHERE id = ?"

PRODUCT_SIZES_QUERY = f"SELECT name, description FROM product_sizes WHERE product_id = ? AND edge_left < ? AND edge_right > ? AND edge_bottom < ? AND edge_top > ?"

DIFFICULTY_QUERY = f"SELECT boulder_name FROM difficulty_grades WHERE difficulty = ?"

VOCAB_PLACEMENTS_QUERY = f"SELECT id FROM placements WHERE layout_id IN ({','.join(str(i) for i in tuple(KILTER['LAYOUTS']))})"

VOCAB_PLACEMENTS_ROLE_QUERY = f"SELECT id FROM placement_roles"

VOCAB_DIFFICULTY_GRADES = f"SELECT boulder_name FROM difficulty_grades"

VOCAB_WORDS = f"SELECT name, description FROM product_sizes WHERE product_id IN ({','.join(str(i) for i in tuple(KILTER['PRODUCTS']))})"

def product_sizes_and_format(cursor:sqlite3.Cursor, climbs):
    """
    Determines the compatible boards for each climb.
    """
    n_climbs = []
    for climb in climbs:
        # Determine compatible boards
        layout_id = str(climb[1])
        edge_left = str(climb[3])
        edge_right = str(climb[4])
        edge_bottom = str(climb[5])
        edge_top = str(climb[6])
        cursor.execute(LAYOUT_TO_PRODUCT_QUERY, layout_id)
        product_id = cursor.fetchall()
        cursor.execute(PRODUCT_SIZES_QUERY, (str(product_id[0][0]), edge_left, edge_right, edge_bottom, edge_top))
        product_sizes = [' '.join(board) for board in cursor.fetchall()]
        # Add boulder data
        climb = list(climb)
        climb.append(product_sizes)
        # Convert difficulty to grade
        cursor.execute(DIFFICULTY_QUERY, (str(round(climb[9])),))
        climb[9] = cursor.fetchall()[0][0]
        # Remove unused data
        n_climbs.append(climb[8:])
    return n_climbs

def build_vocab(cursor:sqlite3.Cursor):
    vocab = ["<SOS>", "<EOS>", "<PAD>"]
    cursor.execute(VOCAB_PLACEMENTS_QUERY)
    vocab.extend([f"p{placement[0]}" for placement in cursor.fetchall()])
    cursor.execute(VOCAB_PLACEMENTS_ROLE_QUERY)
    vocab.extend([f"r{placement[0]}" for placement in cursor.fetchall()])
    cursor.execute(VOCAB_DIFFICULTY_GRADES)
    vocab.extend([d[0] for d in cursor.fetchall()])
    cursor.execute(VOCAB_WORDS)
    vocab.extend([' '.join(board) for board in cursor.fetchall()])
    token_to_id = {t:i for i,t in enumerate(vocab)}
    id_to_token = {int(i):t for i,t in enumerate(vocab)}
    return token_to_id, id_to_token

def frames_regex(climbs:list, token_to_id:list):
    """
    Return the frame as list of tokens of placements and placement roles.
    """
    regex_expr = '(p\d{4}|r\d{2})'
    for climb in climbs:
            try:
                climb[-1] = [token_to_id[f] for f in re.split(regex_expr, climb[-1]) if f]
            except Exception as e:
                 print(e)
                 climbs.remove(climb)
    return climbs

def convert_data_to_tokens(climbs:list, token_to_id:list):
    n_climbs = []
    regex_expr = '(p\d{4}|r\d{2})'
    for climb in climbs:
        n_climb = [climb[0], climb[-1]]
        n_climb[0] = [token_to_id[i] for i in n_climb[0]]
        try:
            n_climb[-1] = [token_to_id[f] for f in re.split(regex_expr, n_climb[-1]) if f]
            n_climbs.append(n_climb)
        except Exception as e:
            print(e)
    return n_climbs

def convert_to_single_board(climbs):
    # [[BOARD, DIFFICULTY], [FRAMES]]
    n_climbs = []
    for climb in climbs:
        for board in climb[-1]:
            n_climb = [[board], climb[0]]
            n_climb[0].append(climb[1])
            n_climbs.append(n_climb)
    return n_climbs

def export_token_id_mappings(token_to_id, id_to_token):

    with open("./data/token_to_id.json", "w") as file:
        file.write(json.dumps(token_to_id))

    with open("./data/id_to_token.json", "w") as file:
        file.write(json.dumps(id_to_token))

def export_text_data(single_boards):
    rows1 = []

    for climb in single_boards:
        src = ','.join([str(i) for i in climb[0]])
        tgt = climb[1]
        rows1.append({'src': src, "tgt": tgt})
    df1 = pd.DataFrame(rows1)
    df1.to_csv("./data/boards.csv", header=False, index=False)

def export_tokens(single_board_tokens):
    srcs = []
    tgts = []
    
    for climb in single_board_tokens:
        srcs.append(climb[0])
        tgts.append(climb[1])

    with open('./data/board_tokens_srcs.json', 'w') as f:
        json.dump(srcs, f)

    with open('./data/board_tokens_tgts.json', 'w') as f:
        json.dump(tgts, f)

if __name__ == "__main__":
    cursor = sqlite3.connect(KILTER["DB"]).cursor()
    cursor.execute(CLIMBS_QUERY)
    climbs = cursor.fetchall()
    climbs = product_sizes_and_format(cursor, climbs)
    single_boards = convert_to_single_board(climbs=climbs)
    token_to_id, id_to_token = build_vocab(cursor=cursor)
    single_board_tokens = convert_data_to_tokens(climbs=single_boards, token_to_id=token_to_id)
    export_token_id_mappings(token_to_id=token_to_id, id_to_token=id_to_token)
    export_text_data(single_boards=single_boards)
    export_tokens(single_board_tokens=single_board_tokens)
    cursor.close()