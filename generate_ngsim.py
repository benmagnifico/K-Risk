import os
import pandas as pd
import json
import re

#Ngsim的system message
ngsim_msg = """"

You are ChatGPT, a large language model trained by OpenAI. You are now acting as a mature driving assistant. And your task is to make the decision that assures the safety.

You have access to the same information as a real human driver, which includes the following:
1. Description of the environment
2. Your own vehicle's position, dynamic information.
3. Vehicles in your surroundings, including their position, dynamic information.

This scenario takes place at a U.S. highway. The positive x-coordinate points east, while the positive y-coordinate points south. The top-left corner is the origin (0,0). The velocity components follow the same convention: a negative vx indicates movement toward the west, while a negative vy indicates movement toward the north.

Your available actions include:
1. IDLE: Remain in the current lane with the current speed (Action ID: 1)
2. Turn Left: Change to the lane on the left of the current lane (Action ID: 2)
3. Turn Right: Change to the lane on the right of the current lane (Action ID: 3)
4. Acceleration: Increase vehicle speed (Action ID: 4)
5. Deceleration: Reduce vehicle speed (Action ID: 5)

A clear action ID must be chosen at the end of your reasoning process.
"""

def generate_ngsim(json_path, frame_step=5, start_frame=None, end_frame=None):
    """
    Generate a risk scene description from a NGSIM JSON file, mentioning TTC and preceding car.
    Supports optional frame filtering via start_frame and end_frame.
    """
    filename = os.path.basename(json_path)
    ego_match = re.search(r'car(\d+)', filename)
    if ego_match:
        ego_id = int(ego_match.group(1))
    else:
        return f"Error: Could not extract ego_id from filename: {filename}"

    if not os.path.exists(json_path):
        return f"Error: JSON file not found: {json_path}"

    # Load JSON
    with open(json_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        df = pd.DataFrame(raw)
    else:
        return "Error: Unknown JSON structure."

    # Standardize column names
    if "Frame_ID" not in df.columns or "Vehicle_ID" not in df.columns:
        return "Error: Missing 'Frame_ID' or 'Vehicle_ID' in JSON."

    df.rename(columns={
        "Frame_ID": "frame",
        "Vehicle_ID": "id",
        "Lane_ID": "laneId",
        "Local_X": "x",
        "Local_Y": "y",
        "v_Vel": "vVel",
        "v_Acc": "vAcc",
        "preceding_id": "precedingId",
        "TTC": "ttc"
    }, inplace=True)

    # === NEW: Filter by frame range ===
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]

    frames = sorted(df["frame"].unique())[::frame_step]
    descriptions = []

    surrounding_cols = {
        "precedingId": "ahead",
        "Following": "behind",
        "leftPrecedingId": "ahead in left lane",
        "leftFollowingId": "behind in left lane",
        "leftAlongsideId": "alongside in left lane",
        "rightPrecedingId": "ahead in right lane",
        "rightFollowingId": "behind in right lane",
        "rightAlongsideId": "alongside in right lane"
    }

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        ego_row = frame_df[frame_df["id"] == ego_id]
        if ego_row.empty:
            continue

        ego_row = ego_row.iloc[0]
        ego_lane = ego_row.get("laneId", "Unknown")
        ego_speed = ego_row.get("vVel", "N/A")
        ego_acc = ego_row.get("vAcc", "N/A")
        ttc_value = ego_row.get("ttc", None)
        preceding_id = int(ego_row.get("precedingId", -1)) if pd.notna(ego_row.get("precedingId")) else -1

        paragraph = f"""
        --------------------------------------------
        \nFrame {frame}:
        Ego Vehicle {ego_id} is driving in lane {ego_lane}.
        Speed: {ego_speed:.2f} m/s, Acceleration: {ego_acc:.2f} m/s².
        """

        # Add TTC info if available
        if pd.notna(ttc_value) and ttc_value > 0:
            if preceding_id != -1:
                paragraph += f"\n⚠️ TTC to preceding Vehicle {preceding_id}: {ttc_value:.2f} seconds."
            else:
                paragraph += f"\n⚠️ TTC available: {ttc_value:.2f} seconds (no preceding vehicle ID recorded)."

        surrounding_descriptions = []

        for col, desc in surrounding_cols.items():
            sur_id = ego_row.get(col, -1)
            if pd.isna(sur_id) or sur_id == -1:
                continue

            sur_row = frame_df[frame_df["id"] == int(sur_id)]
            if not sur_row.empty:
                sur_row = sur_row.iloc[0]
                sur_speed = sur_row.get("vVel", "N/A")
                sur_acc = sur_row.get("vAcc", "N/A")
                surrounding_descriptions.append(
                    f"Vehicle {int(sur_id)} is {desc} with speed {sur_speed:.2f} m/s and acceleration {sur_acc:.2f} m/s²."
                )

        if surrounding_descriptions:
            paragraph += "\nSurrounding Vehicles:\n" + "\n".join(surrounding_descriptions)
        else:
            paragraph += "\nNo surrounding vehicles detected."

        descriptions.append(paragraph)

    return "\n\n".join(descriptions)
