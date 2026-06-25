"""
Pure ranking logic: dominance algorithm (block order + threshold).
No database or API dependencies.
"""
from typing import Dict, List, Optional, Tuple


def find_dominant_winner(
    miner_stats: List[Tuple[str, int, float, Optional[int]]],
    threshold: float,
) -> Optional[str]:
    """Find top-ranked miner by dominance: block order first; to dominate earlier miners,
    pass_rate must exceed theirs by threshold. miner_stats: (hotkey, passed_count, pass_rate, block)."""
    if not miner_stats:
        return None
    ordered = sorted(
        miner_stats,
        key=lambda x: (x[3] if x[3] is not None else (2**31), x[0]),
    )
    for i in range(len(ordered) - 1, -1, -1):
        hotkey, _wc, rate, _b = ordered[i]
        dominates_all = True
        for j in range(i):
            _h, _w, pred_rate, _ = ordered[j]
            if rate < pred_rate + threshold:
                dominates_all = False
                break
        if dominates_all:
            return hotkey
    return None


def compute_rank_from_miner_stats(
    miner_stats: List[Tuple[str, int, float, Optional[int]]],
    threshold: float,
) -> Tuple[Optional[str], List[dict]]:
    """From (hotkey, passed_count, pass_rate, block) list, compute top-ranked miner and full rank.
    Returns (winner_hotkey, rank_entries).
    
    If no dominant winner, the miner with highest passed_count gets rank 1 as fallback.
    """
    if not miner_stats:
        return None, []
    block_by_hotkey: Dict[str, Optional[int]] = {m[0]: m[3] for m in miner_stats}
    winner_hotkey = find_dominant_winner(miner_stats, threshold)
    by_passed_count = sorted(miner_stats, key=lambda x: (-x[1], x[0]))

    if not winner_hotkey and by_passed_count:
        winner_hotkey = by_passed_count[0][0]
    
    rank_entries: List[dict] = []
    if winner_hotkey:
        rank_entries.append({
            "miner_hotkey": winner_hotkey,
            "rank": 1,
            "passed_count": next(x[1] for x in miner_stats if x[0] == winner_hotkey),
            "pass_rate": next(x[2] for x in miner_stats if x[0] == winner_hotkey),
            "block": block_by_hotkey.get(winner_hotkey),
        })
    seen = {winner_hotkey} if winner_hotkey else set()
    r = 2
    for hotkey, passed_count, pass_rate, block in by_passed_count:
        if hotkey in seen:
            continue
        seen.add(hotkey)
        rank_entries.append({
            "miner_hotkey": hotkey,
            "rank": r,
            "passed_count": passed_count,
            "pass_rate": pass_rate,
            "block": block,
        })
        r += 1
    return winner_hotkey, rank_entries
