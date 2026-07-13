import cv2
import numpy as np
import time
import multiprocessing as mp
import ctypes
import pickle
from pathlib import Path

BASE = Path(__file__).parent
MAX_DET = 100
TITLE_DST = "Block Blast Auto-Bot - press q to quit"

MAX_W, MAX_H = 1200, 1200

rng = np.random.default_rng(42)
PALETTE = rng.integers(80, 255, (100, 3)).tolist()

def colour(cls_id):
    if type(cls_id) is str:
        idx = sum(ord(c) for c in cls_id)
    else:
        idx = int(cls_id)
    c = PALETTE[idx % len(PALETTE)]
    return (int(c[0]), int(c[1]), int(c[2]))

def draw(frame, det_list, names):
    for (x1, y1, x2, y2, conf, cls_id) in det_list:
        if cls_id == "filled":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            continue
        if cls_id == "empty":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            continue
            
        label = f"{names.get(cls_id, str(cls_id))}"
        col = colour(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def scrcpy_worker(frame_buf, frame_seq, frame_w, frame_h, fps_q, frame_event, cmd_q):
    print("[scrcpy_worker] Process started.")
    import scrcpy
    client = scrcpy.Client(max_fps=30, bitrate=2000000, max_width=800)
    
    import threading
    def command_listener():
        while True:
            try:
                cmd = cmd_q.get()
                if cmd[0] == "swipe":
                    _, sx, sy, ex, ey = cmd
                    client.control.swipe(int(sx), int(sy), int(ex), int(ey), 5, 0.01)
                elif cmd[0] == "touch":
                    _, x, y, action = cmd
                    client.control.touch(int(x), int(y), action)
            except Exception as e:
                print(f"Control error: {e}")
                
    threading.Thread(target=command_listener, daemon=True).start()

    shared_array = np.ctypeslib.as_array(frame_buf).reshape(MAX_H, MAX_W, 3)

    fps_counter = 0
    fps_timer = time.perf_counter()

    def on_frame(frame):
        nonlocal fps_counter, fps_timer
        if frame is None:
            return

        h, w = frame.shape[:2]
        if h > MAX_H or w > MAX_W:
            print(f"[scrcpy_worker] Frame too large: {h}x{w}")
            return

        shared_array[:h, :w, :] = frame
        
        frame_w.value = w
        frame_h.value = h
        frame_seq.value += 1
        frame_event.set()

        fps_counter += 1
        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            if not fps_q.full():
                fps_q.put(("scrcpy", fps_counter / (now - fps_timer)))
            fps_counter = 0
            fps_timer = now

    client.add_listener(scrcpy.EVENT_FRAME, on_frame)
    print("starting scrcpy client")
    try:
        client.start(threaded=False)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"scrcpy client error: {e}")

def inference_worker(frame_buf, frame_seq, frame_w, frame_h, det_buf, det_len, fps_q, frame_event):
    print("[inference_worker] Process started (OpenCV CV Mode).")

    try:
        shared_array = np.ctypeslib.as_array(frame_buf).reshape(MAX_H, MAX_W, 3)
        det_array = np.ctypeslib.as_array(det_buf)

        last_seq = -1
        fps_counter = 0
        fps_timer = time.perf_counter()

        bg_color = np.array([141, 77, 59])

        while True:
            while frame_seq.value == last_seq:
                pass

            last_seq = frame_seq.value
            h, w = frame_h.value, frame_w.value
            if h == 0 or w == 0:
                continue

            frame = shared_array[:h, :w, :].copy()

            tray_y1 = int(h * 0.70)
            tray_y2 = h
            tray = frame[tray_y1:tray_y2, 0:w].copy()
            
            diff = np.abs(tray.astype(np.int32) - bg_color)
            mask = np.any(diff > 35, axis=-1).astype(np.uint8) * 255
            
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_boxes = []
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw > 15 and ch > 15:
                    valid_boxes.append((cx, cy, cw, ch))
                    
            valid_boxes = sorted(valid_boxes, key=lambda b: b[0])
            
            dets = []
            
            bx, by, bw, bh = 19, 187, 328, 332
            board_bg = np.array([65, 32, 29])
            
            by1, by2 = max(0, by), min(h, by + bh)
            bx1, bx2 = max(0, bx), min(w, bx + bw)
            
            board_crop = frame[by1:by2, bx1:bx2]
            if board_crop.size > 0:
                diff_board = np.abs(board_crop.astype(np.int32) - board_bg)
                mask_board = np.any(diff_board > 35, axis=-1).astype(np.uint8) * 255
                
                for r in range(8):
                    for c in range(8):
                        
                        cell_x1 = c * bw / 8.0
                        cell_y1 = r * bh / 8.0
                        cell_x2 = (c + 1) * bw / 8.0
                        cell_y2 = (r + 1) * bh / 8.0
                        
                        dw = (cell_x2 - cell_x1) * 0.25
                        dh = (cell_y2 - cell_y1) * 0.25
                        
                        samp_x1 = int(cell_x1 + dw)
                        samp_x2 = int(cell_x2 - dw)
                        samp_y1 = int(cell_y1 + dh)
                        samp_y2 = int(cell_y2 - dh)
                        
                        samp = mask_board[samp_y1:samp_y2, samp_x1:samp_x2]
                        is_filled = np.mean(samp) > 127
                        
                        draw_x1 = bx + samp_x1
                        draw_y1 = by + samp_y1
                        draw_x2 = bx + samp_x2
                        draw_y2 = by + samp_y2
                        
                        if is_filled:
                            dets.append((draw_x1, draw_y1, draw_x2, draw_y2, 1.0, "filled"))
                        else:
                            dets.append((draw_x1, draw_y1, draw_x2, draw_y2, 1.0, "empty"))
                            
            for (cx, cy, cw, ch) in valid_boxes:
                
                rx = cx
                ry = cy + tray_y1
                
                piece_mask = mask[cy:cy+ch, cx:cx+cw]
                
                grid_w = max(1, round(cw / 18.5))
                grid_h = max(1, round(ch / 18.5))
                
                small_mask = cv2.resize(piece_mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
                
                shape_rows = []
                for row in small_mask:
                    shape_rows.append("".join(["1" if val > 127 else "0" for val in row]))
                    
                shape_str = "_".join(shape_rows)
                dets.append((rx, ry, rx + cw, ry + ch, 1.0, shape_str))

            det_bytes = pickle.dumps(dets)
            length = len(det_bytes)
            if length < 65536:
                det_array[:length] = np.frombuffer(det_bytes, dtype=np.uint8)
                det_len.value = length

            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                if not fps_q.full():
                    fps_q.put(("infer", fps_counter / (now - fps_timer)))
                fps_counter = 0
                fps_timer = now
    except Exception as e:
        import traceback
        print(f"inference worker crashed: {e}")
        traceback.print_exc()

latest_dets = []
next_move = None

def get_best_move(board, pieces, failed_moves=None, piece_classes=None):
    def place(b, p, r, c):
        ph, pw = p.shape
        if r + ph > 8 or c + pw > 8: return None, 0
        if np.any(b[r:r+ph, c:c+pw] & p): return None, 0
        
        new_b = b.copy()
        new_b[r:r+ph, c:c+pw] |= p
        
        rows_to_clear = np.all(new_b, axis=1)
        cols_to_clear = np.all(new_b, axis=0)
        lines = np.sum(rows_to_clear) + np.sum(cols_to_clear)
        
        new_b[rows_to_clear, :] = False
        new_b[:, cols_to_clear] = False
        return new_b, lines

    memo = {}

    LINE_WEIGHT = 1_000_000
                                                                                 
    EARLY_CLEAR_WEIGHT = 50
                                                                               
    COMBO_WEIGHT = 20_000

    def dfs(current_board, remaining_pieces, current_score):
        if not remaining_pieces:
            row_sums = np.sum(current_board, axis=1)
            col_sums = np.sum(current_board, axis=0)
            clustering = np.sum(row_sums ** 2) + np.sum(col_sums ** 2)
            board_term = clustering - np.sum(current_board) * 100
                                                                              
            tiebreak = float(np.clip(board_term, -500, 500))
            eval_score = current_score * LINE_WEIGHT + tiebreak
            return eval_score, []
            
        board_hash = current_board.tobytes()
        rem_tuple = tuple(i for i, p, p_str in remaining_pieces)
        state_key = (board_hash, rem_tuple)
        
        if state_key in memo:
            return memo[state_key]
            
        max_eval = -float("inf")
        best_seq = []
        
        seen_pieces = set()
        depth_remaining = len(remaining_pieces) - 1                              
        
        for i, (orig_idx, p, p_str) in enumerate(remaining_pieces):
            p_hash = p.tobytes()
            if p_hash in seen_pieces:
                continue
            seen_pieces.add(p_hash)
            
            ph, pw = p.shape
            placed_any = False
            for r in range(8 - ph + 1):
                for c in range(8 - pw + 1):
                    if (p_str, r, c) in failed_moves:
                        continue
                    new_b, lines = place(current_board, p, r, c)
                    if new_b is not None:
                        placed_any = True
                        next_pieces = remaining_pieces[:i] + remaining_pieces[i+1:]
                        eval_score, seq = dfs(new_b, next_pieces, current_score + lines)
                        if lines > 0:
                                                                                  
                            eval_score += lines * EARLY_CLEAR_WEIGHT * (depth_remaining + 1)
                                                                                  
                            eval_score += (lines ** 2) * COMBO_WEIGHT
                        if eval_score > max_eval:
                            max_eval = eval_score
                            best_seq = [(orig_idx, r, c)] + seq
                            
            if not placed_any:
                memo[state_key] = (-float("inf"), [])
                return -float("inf"), []
                
        memo[state_key] = (max_eval, best_seq)
        return max_eval, best_seq

    rem = [(i, p, piece_classes[i]) for i, p in enumerate(pieces)]
    score, seq = dfs(board, rem, 0)
    if seq:
        return seq[0]
    return None

def bot_thread(frame_w, frame_h, cmd_q, frame_buf):
    import threading
    import subprocess
    import time
    import numpy as np
    global latest_dets, next_move
    
    shared_array = np.ctypeslib.as_array(frame_buf).reshape(MAX_H, MAX_W, 3)
    
    print("[BOT] Started.")
    failed_moves = set()
    last_num_pieces = 0
        
    while True:
        time.sleep(3.0)
        
        fw = frame_w.value
        fh = frame_h.value
        if fw == 0 or fh == 0:
            continue
        
        dets = list(latest_dets)
        if len(dets) == 0:
            continue
            
        board = np.zeros((8, 8), dtype=bool)
        pieces = []
        piece_coords = []
        piece_classes = []
        
        bx, by, bw, bh = 19, 187, 328, 332
        
        for (x1, y1, x2, y2, conf, cls_id) in dets:
            if cls_id == "filled":
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                c = int((cx - bx) / (bw / 8.0))
                r = int((cy - by) / (bh / 8.0))
                if 0 <= r < 8 and 0 <= c < 8:
                    board[r, c] = True
            elif cls_id != "empty":
                rows = cls_id.split("_")
                p_arr = [[True if ch == '1' else False for ch in row] for row in rows]
                p_np = np.array(p_arr, dtype=bool)
                pieces.append(p_np)
                piece_coords.append( ((x1+x2)/2.0, (y1+y2)/2.0) )
                piece_classes.append(cls_id)
                
        num_pieces = len(pieces)
        if last_num_pieces != num_pieces:
            failed_moves = set()
            last_num_pieces = num_pieces
            
        print(f"\n[BOT] Thinking... Board blocks: {np.sum(board)}, Pieces available: {num_pieces}")
        best = get_best_move(board, pieces, failed_moves, piece_classes)
        
        if best is not None:
            orig_idx, r, c = best
            p = pieces[orig_idx]
            
            global next_move
            next_move = (p, r, c)
            
            ph, pw = p.shape
            px, py = piece_coords[orig_idx]
            
            tx = bx + (c + pw / 2.0) * (bw / 8.0)
            ty = by + (r + ph / 2.0) * (bh / 8.0)
            
            p_top_x = np.where(p[0, :])[0]
            top_center_blocks = (np.min(p_top_x) + np.max(p_top_x)) / 2.0 + 0.5
            piece_center_blocks = pw / 2.0
            offset_pixels = (piece_center_blocks - top_center_blocks) * (bw / 8.0)
            
            target_tx = 0.762 * tx + 0.551 * pw + 0.285 * px - 7.693
            target_ty = 0.735 * ty - 2.368 * ph - 0.622 * py + 619.801
            
            scrcpy_tx = max(20, min(fw - 20, target_tx))
            scrcpy_ty = max(20, min(fh - 20, target_ty))
            
            print(f"--- BOARD STATE ---\n{board.astype(int)}\n-------------------")
            print(f"--- PIECE SHAPE ---\n{p.astype(int)}\n-------------------")
            print(f"[BOT] Found move: Piece {orig_idx} to grid ({r}, {c})")
            print(f"[BOT] Touch coordinates: px={px:.1f}, py={py:.1f} -> target_tx={scrcpy_tx:.1f}, target_ty={scrcpy_ty:.1f}")
            print(f"[BOT] Picking up piece and relying entirely on 1-shot regression formula!")
            
            cmd_q.put(("touch", px, py, 0)) 
            time.sleep(0.05)
            cmd_q.put(("touch", scrcpy_tx, scrcpy_ty, 2)) 
            time.sleep(0.1) 
                    
            print("[BOT] Dropping piece!")
            cmd_q.put(("touch", scrcpy_tx, scrcpy_ty, 1)) 
            
            next_move = None
            print("[BOT] Waiting for piece to lock and animations to clear...")
            time.sleep(1.0)
            
            new_dets = list(latest_dets)
            new_pieces = [d for d in new_dets if d[5] != "empty" and d[5] != "filled"]
            if len(new_pieces) == num_pieces:
                print("[BOT] Move was REJECTED by the game! Blacklisting this move.")
                failed_moves.add((piece_classes[orig_idx], r, c))
                
            print(f"[BOT] Post-drop: Pieces available is now {len(new_pieces)}.")
        else:
            print("[BOT] No valid moves!")
            time.sleep(3.0)

def main():
    frame_buf = mp.Array(ctypes.c_uint8, MAX_H * MAX_W * 3, lock=False)
    frame_seq = mp.Value('i', 0, lock=False)
    frame_w = mp.Value('i', 0, lock=False)
    frame_h = mp.Value('i', 0, lock=False)

    det_buf = mp.Array(ctypes.c_uint8, 65536, lock=False)
    det_len = mp.Value('i', 0, lock=False)

    frame_event = mp.Event()

    fps_q = mp.Queue(maxsize=10)
    cmd_q = mp.Queue()

    p_scrcpy = mp.Process(target=scrcpy_worker, args=(frame_buf, frame_seq, frame_w, frame_h, fps_q, frame_event, cmd_q))
    p_infer = mp.Process(target=inference_worker, args=(frame_buf, frame_seq, frame_w, frame_h, det_buf, det_len, fps_q, frame_event))

    p_scrcpy.daemon = True
    p_infer.daemon = True
    p_scrcpy.start()
    p_infer.start()

    cv2.namedWindow(TITLE_DST, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE_DST, 400, 800)

    shared_array = np.ctypeslib.as_array(frame_buf).reshape(MAX_H, MAX_W, 3)

    global latest_dets
    latest_dets = []
    
    import threading
    threading.Thread(target=bot_thread, args=(frame_w, frame_h, cmd_q, frame_buf), daemon=True).start()
    names = {}
    fps_stats = {"scrcpy": 0.0, "infer": 0.0}

    print("waiting for stream...")
    last_rendered_seq = -1

    try:
        while True:
            while not fps_q.empty():
                try:
                    src, val = fps_q.get_nowait()
                    if src == "names":
                        names = val
                    else:
                        fps_stats[src] = val
                except Exception:
                    pass

            dlen = det_len.value
            if dlen > 0:
                try:
                    det_bytes = np.ctypeslib.as_array(det_buf)[:dlen].tobytes()
                    latest_dets = pickle.loads(det_bytes)
                except Exception:
                    pass

            seq = frame_seq.value
            if seq == last_rendered_seq or frame_w.value == 0:
                cv2.waitKey(10)
                continue

            last_rendered_seq = seq
            h, w = frame_h.value, frame_w.value
            frame = shared_array[:h, :w, :].copy()

            draw(frame, latest_dets, names)
            
            if next_move is not None:
                p, r, c = next_move
                ph, pw = p.shape
                bx, by, bw, bh = 19, 187, 328, 332
                for pr in range(ph):
                    for pc in range(pw):
                        if p[pr, pc]:
                            cell_r = r + pr
                            cell_c = c + pc
                            cell_x1 = bx + int(cell_c * bw / 8)
                            cell_y1 = by + int(cell_r * bh / 8)
                            cell_x2 = bx + int((cell_c + 1) * bw / 8)
                            cell_y2 = by + int((cell_r + 1) * bh / 8)
                            
                            cv2.rectangle(frame, (cell_x1, cell_y1), (cell_x2, cell_y2), (255, 0, 0), 4)

            hud_text = (f"infer {fps_stats['infer']:.1f} fps  "
                        f"capture {fps_stats['scrcpy']:.1f} fps  "
                        f"detections {len(latest_dets)}")

            cv2.putText(frame, hud_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            
            cv2.imshow(TITLE_DST, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("interrupted")
    finally:
        print("shutting down worker processes")
        p_scrcpy.terminate()
        p_infer.terminate()
        cv2.destroyAllWindows()
        print("done")

if __name__ == "__main__":
    main()

# yes its vibe coded fuck you