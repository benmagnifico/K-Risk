import os
import json
import pandas as pd

#InD的system message
ind_msg = """"

You are ChatGPT, a large language model trained by OpenAI. You are now acting as a mature driving assistant. And your task is to make the decision that assures the safety.

You have access to the same information as a real human driver, which includes the following:
1. Description of the environment
2. Your own vehicle's position, dynamic information.
3. Vehicles in your surroundings, including their position, dynamic information.

This scenario takes place at an urban intersection. The positive x-coordinate points east, while the positive y-coordinate points north. The top-left corner is the origin (0,0). The velocity components follow the same convention: a negative vx indicates movement toward the west, while a negative vy indicates movement toward the north.

Your available actions include:
1. IDLE: Remain in the current lane with the current speed (Action ID: 1)
2. Turn Left: Change to the lane on the left of the current lane (Action ID: 2)
3. Turn Right: Change to the lane on the right of the current lane (Action ID: 3)
4. Acceleration: Increase vehicle speed (Action ID: 4)
5. Deceleration: Reduce vehicle speed (Action ID: 5)

A clear action ID must be chosen at the end of your reasoning process.
"""


def generate_ind(json_path, frame_step=5, start_frame=None, end_frame=None):
    """
    Generate a risk scene description from a JSON file for inD or rounD dataset.
    Supports frame range filtering via start_frame and end_frame.
    """
    filename = os.path.basename(json_path)
    try:
        parts = filename.replace(".json", "").split("_")
        ego_id = int(parts[3])  # assuming ego ID is the 4th part: recording_X_ego_ID_frame
    except Exception:
        return f"Error: Could not extract ego id from filename: {filename}"

    if not os.path.exists(json_path):
        return f"Error: JSON file not found: {json_path}"

    with open(json_path, "r") as f:
        raw = json.load(f)

    # --- Parse JSON format ---
    rows = []
    for frame_entry in raw:
        frame_id = frame_entry.get("frame_id")
        for v in frame_entry.get("vehicles", []):
            v["frame"] = frame_id
            rows.append(v)

    df = pd.DataFrame(rows)

    if 'frame' not in df.columns or 'trackId' not in df.columns:
        return "Error: Missing essential fields 'frame' or 'trackId'."

    # === NEW: Filter by frame range ===
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]

    frames = sorted(df["frame"].unique())[::frame_step]
    descriptions = []

    # === Fully consistent urban wording ===
    categories = {
        "preceding_id": "driving in the preceding direction",
        "following_id": "driving in the following direction",
        "left_preceding_id": "driving in the left-preceding direction",
        "left_following_id": "driving in the left-following direction",
        "left_alongside_id": "driving in the left-alongside direction",
        "right_preceding_id": "driving in the right-preceding direction",
        "right_following_id": "driving in the right-following direction",
        "right_alongside_id": "driving in the right-alongside direction"
    }

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        ego_row = frame_df[frame_df["trackId"] == ego_id]
        if ego_row.empty:
            continue

        ego_row = ego_row.iloc[0]
        paragraph = f"""
        --------------------------------------------
        \nFrame {frame}:
        The ego vehicle is positioned at ({ego_row['xCenter']:.2f}, {ego_row['yCenter']:.2f}) meters.
        - Longitudinal Speed: {ego_row.get('lonVelocity', 0):.2f} m/s
        - Heading: {ego_row.get('heading', 0):.2f} degrees
        - Risk Value: {ego_row.get('risk_value', 0):.2f}
        """

        surrounding_descriptions = []
        for col, description in categories.items():
            sur_id = ego_row.get(col)
            if pd.isna(sur_id) or sur_id == -1:
                continue

            sur_row = frame_df[frame_df["trackId"] == int(sur_id)]
            if sur_row.empty:
                continue
            sur_row = sur_row.iloc[0]

            surrounding_descriptions.append(f"""
            Vehicle {int(sur_id)} is {description}.
            Position: ({sur_row['xCenter']:.2f}, {sur_row['yCenter']:.2f}) meters
            - Longitudinal Speed: {sur_row.get('lonVelocity', 0):.2f} m/s
            - Heading: {sur_row.get('heading', 0):.2f} degrees
            - Risk Value: {sur_row.get('risk_value', 0):.2f}
            """)

        if surrounding_descriptions:
            paragraph += "\nSurrounding Vehicles:\n" + "\n".join(surrounding_descriptions)
        else:
            paragraph += "\nNo surrounding vehicles detected."

        descriptions.append(paragraph)

    return "\n\n".join(descriptions)
