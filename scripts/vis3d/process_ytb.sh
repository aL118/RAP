set -euo pipefail

BASE="/fs/nexus-projects/sim2real/aliu/RAP"

NAME="blocker"
VIDEO_DIR="$BASE/data/YTB/$NAME"
HZ=2

# LINK="https://www.reddit.com/r/dashcams/comments/1uh2gbc/unsafe_at_any_speed/"

# mkdir -p "$VIDEO_DIR"

# # curl --fail -L -A "Mozilla/5.0" -o "$VIDEO_DIR/$NAME.mp4" "$LINK"

# # COOKIES="$BASE/scripts/vis3d/cookies.txt"
# yt-dlp --js-runtimes node --remote-components ejs:github -f "bv*[ext=mp4][vcodec^=avc1]" -o $VIDEO_DIR/$NAME.mp4 $LINK

python "$BASE/vis3d/process_ytb.py" \
--video "$VIDEO_DIR/$NAME.mp4" \
--save-dir "$VIDEO_DIR/frames" \
--hz "$HZ"

# freeway > arterial > residental
# - weather: clear | cloudy | rainy | snowy
# - time: day | dawn_dusk | night
# - sung_glare: true | false
# - land_use: urban | suburban | rural
# - road_type: freeway | arterial | residential | parking_lot
# - event_type: near_miss | collision | cut_in | hard_brake | ped_in_roadway | blocking
# - source_url
# - retrieved